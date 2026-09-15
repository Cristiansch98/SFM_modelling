"""Real-trajectory interface to the highD dataset (Krajewski et al., 2018).

60 drone recordings of German motorway traffic at 25 Hz, streamed straight from
`highD-dataset-v1.0.zip` (4.78 GB uncompressed, so nothing is ever extracted).

**highD contains no emergency vehicles.** What it can calibrate and validate is
the *host-traffic* behaviour the EV terms in `emv/forces.py` act upon - how real
drivers execute a lateral manoeuvre (duration, lateral speed and acceleration
envelope, lane-keeping discipline) and the traffic the EV drives through
(per-lane speed distributions, headways, TTC floor, truck share, geometry). The
yielding *response* itself - onset distance, compliance, corridor strength - is
not observable here; `emv/groundtruth.py`'s dashcam stream remains the only
EV-encounter reference. Used together the two bracket the model: highD measures
*discretionary* lane changes, the dashcam GT measures *EV-induced* ones, and the
latter are the gentler of the two.

Three normalisations are the whole bug surface, and all three are verified in
`tests/test_highd.py` (synthetic tracks) and in `docs/HIGHD_VALIDATION.md`
(recording 01):

  1. `x`, `y` in `tracks.csv` are the bounding-box **upper-left corner**, not the
     centre: `xc = x + width/2`, `yc = y + height/2`. (Recording 01, lane 5:
     mean |y + h/2 - lane centre| = 0.29 m versus |y - centre| = 1.04 m.) Note
     also that `width` is the *longitudinal* extent (vehicle length) and
     `height` the *lateral* one (vehicle width).
  2. The upper carriageway (`drivingDirection == 1`) travels **-x**. Mapping
     x, y, vx, vy -> -x, +y, -vx, +vy (and mirroring the marking array) puts
     travel along +x with lane 0 rightmost and +y to the left, i.e. the
     convention of `emv/road.py`.
  3. `laneId` is **global across both carriageways** (recording 01: {2,3} upper,
     {5,6} lower, id 4 = the median gap). Per-direction lane indices therefore
     come from `recordingMeta`'s marking arrays, never from `laneId` arithmetic.

Observables are emitted under the **same key names** as
`emv.metrics.behaviour_stats`, and lane changes are timed by calling the *same
function the model calls* (`emv.metrics.lane_change_from_track`, two-lane
occupancy after Thiemann et al. 2008), so a model number and a highD number with
the same name provably mean the same thing.

Reader design note: `np.loadtxt` on a narrow `usecols` subset was measured
*faster* than `pandas.read_csv` on this data (0.47 s versus 0.71 s for
`data/01_tracks.csv`), so this module is numpy-only and the repo gains no new
dependency - which also keeps the extractor runnable on the Windows box.
"""
from __future__ import annotations

import hashlib
import io
import os
import zipfile
from dataclasses import dataclass, field

import numpy as np

from . import metrics

#: Bump whenever an observable's *definition* changes; caches with an older
#: version are refused rather than silently pooled (see `cache_is_valid`).
EXTRACTOR_VERSION = 3

#: Traffic regimes, classified per **carriageway** (not per recording): one
#: direction can be jammed while the other flows freely, and recording 25 is
#: exactly that case, so a recording-level mean hides both regimes. `dense`
#: brackets the 1700 veh/h/lane of `scenarios.make_sumo_like`, giving the SUMO
#: mirror a real-world density twin.
REGIME_BOUNDS = {
    "freeflow":  dict(med_speed=(28.0, 99.0), flow=(0.0, 1300.0)),
    "dense":     dict(med_speed=(18.0, 99.0), flow=(1300.0, 2600.0)),
    "congested": dict(med_speed=(0.0, 18.0),  flow=(0.0, 3000.0)),
}


def classify_regime(med_speed: float, flow: float) -> str | None:
    for name, b in REGIME_BOUNDS.items():
        lo, hi = b["med_speed"]
        flo, fhi = b["flow"]
        if lo <= med_speed < hi and flo <= flow < fhi:
            return name
    return None

#: Model-side integration/recording step the data is decimated onto. 25 Hz / 3
#: is exactly 0.12 s, which is the `rec_dt` every experiment in this repo uses,
#: so no interpolation is involved.
REC_DT = 0.12
SOURCE_HZ = 25.0

_ZIP_CANDIDATES = (
    os.path.join(os.path.expanduser("~"), "highD-dataset-v1.0.zip"),
    os.path.join(os.path.expanduser("~"), "Downloads", "highD-dataset-v1.0.zip"),
    os.path.join(os.path.expanduser("~"), "Desktop", "highD-dataset-v1.0.zip"),
)


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[0] if paths else ""


#: Dataset location. Override with $EMV_HIGHD_ZIP or the --zip CLI flag.
HIGHD_ZIP = os.environ.get("EMV_HIGHD_ZIP") or _first_existing(_ZIP_CANDIDATES)

# tracks.csv column order (v1.0):
#  0 frame          1 id             2 x              3 y
#  4 width          5 height         6 xVelocity      7 yVelocity
#  8 xAcceleration  9 yAcceleration 10 frontSightDist 11 backSightDist
# 12 dhw           13 thw           14 ttc           15 precedingXVelocity
# 16..23 neighbour ids              24 laneId
_TRACK_USECOLS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 24)
_TRACK_NAMES = ("frame", "id", "x", "y", "len", "wid", "vx", "vy",
                "ax", "ay", "dhw", "thw", "ttc", "laneId")

_TMETA_NUMCOLS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)
_TMETA_NAMES = ("id", "len", "wid", "initialFrame", "finalFrame", "numFrames",
                "drivingDirection", "traveledDistance", "minXVelocity",
                "maxXVelocity", "meanXVelocity", "minDHW", "minTHW", "minTTC",
                "numLaneChanges")


# ---------------------------------------------------------------------------
# zip access
# ---------------------------------------------------------------------------
def _member(rid: int, what: str) -> str:
    return f"data/{rid:02d}_{what}.csv"


def _read_cols(zf: zipfile.ZipFile, name: str, usecols, dtype=np.float64):
    """Parse a comma-separated member into a (rows, len(usecols)) array.

    numpy-only by design (see the module docstring): a pandas fast path was
    measured and is *slower* for narrow column subsets on this data. If a
    wide-column need ever arises, this is the only function to change - nothing
    else in the package may import pandas (asserted in tests/test_highd.py).
    """
    with zf.open(name) as fh:
        return np.loadtxt(io.TextIOWrapper(fh, "utf-8"), delimiter=",",
                          skiprows=1, usecols=usecols, dtype=dtype, ndmin=2)


def recording_ids(zip_path: str = HIGHD_ZIP) -> list[int]:
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    return [r for r in range(1, 101) if _member(r, "tracks") in names]


def zip_fingerprint(zip_path: str = HIGHD_ZIP, full_sha: bool = False) -> dict:
    """Cheap integrity key: sha256 over the central directory's
    (name, crc32, size) triples - no decompression. `full_sha` adds the ~8 s
    whole-file hash."""
    with zipfile.ZipFile(zip_path) as zf:
        triples = sorted((i.filename, i.CRC, i.file_size) for i in zf.infolist())
    h = hashlib.sha256()
    for name, crc, size in triples:
        h.update(f"{name}|{crc}|{size}\n".encode())
    out = dict(zip=os.path.basename(zip_path),
               zip_bytes=os.path.getsize(zip_path),
               crc_digest=h.hexdigest(), zip_sha256=None)
    if full_sha:
        g = hashlib.sha256()
        with open(zip_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                g.update(chunk)
        out["zip_sha256"] = g.hexdigest()
    return out


# ---------------------------------------------------------------------------
# metadata pass (all 60 recordings in ~2 s, no tracks.csv touched)
# ---------------------------------------------------------------------------
@dataclass
class Recording:
    rid: int
    frame_rate: float
    location_id: int
    speed_limit: float                 # m/s; -1 = unrestricted
    duration_s: float
    driven_km: float
    driven_h: float
    n_vehicles: int
    n_cars: int
    n_trucks: int
    marks_upper: list = field(default_factory=list)
    marks_lower: list = field(default_factory=list)

    @property
    def n_lanes_upper(self) -> int:
        return max(0, len(self.marks_upper) - 1)

    @property
    def n_lanes_lower(self) -> int:
        return max(0, len(self.marks_lower) - 1)

    @property
    def lane_widths(self) -> list:
        return (list(np.diff(self.marks_upper)) + list(np.diff(self.marks_lower)))

    @property
    def lane_width_med(self) -> float:
        w = self.lane_widths
        return float(np.median(w)) if w else float("nan")

    def marks(self, direction: str) -> list:
        return self.marks_upper if direction == "upper" else self.marks_lower

    def n_lanes(self, direction: str) -> int:
        return (self.n_lanes_upper if direction == "upper"
                else self.n_lanes_lower)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("rid", "frame_rate", "location_id", "speed_limit", "duration_s",
              "driven_km", "driven_h", "n_vehicles", "n_cars", "n_trucks")}
        d.update(n_lanes_upper=self.n_lanes_upper,
                 n_lanes_lower=self.n_lanes_lower,
                 lane_width_med=round(self.lane_width_med, 3),
                 lane_widths=[round(float(w), 3) for w in self.lane_widths],
                 marks_upper=[float(m) for m in self.marks_upper],
                 marks_lower=[float(m) for m in self.marks_lower])
        return d


def read_recording_meta(zip_path: str, rid: int) -> Recording:
    with zipfile.ZipFile(zip_path) as zf:
        txt = zf.read(_member(rid, "recordingMeta")).decode("utf-8")
    head, row = (ln.strip() for ln in txt.splitlines()[:2])
    rec = dict(zip(head.split(","), row.split(",")))
    marks = lambda s: [float(v) for v in rec[s].split(";") if v.strip()]
    dur = float(rec["duration"])
    return Recording(
        rid=int(rec["id"]), frame_rate=float(rec["frameRate"]),
        location_id=int(rec["locationId"]), speed_limit=float(rec["speedLimit"]),
        duration_s=dur,
        driven_km=float(rec["totalDrivenDistance"]) / 1000.0,
        driven_h=float(rec["totalDrivenTime"]) / 3600.0,
        n_vehicles=int(rec["numVehicles"]), n_cars=int(rec["numCars"]),
        n_trucks=int(rec["numTrucks"]),
        marks_upper=marks("upperLaneMarkings"),
        marks_lower=marks("lowerLaneMarkings"))


def read_tracks_meta(zip_path: str, rid: int) -> dict:
    """Per-track metadata. `cls` is the only string column, read separately."""
    with zipfile.ZipFile(zip_path) as zf:
        num = _read_cols(zf, _member(rid, "tracksMeta"), _TMETA_NUMCOLS)
        with zf.open(_member(rid, "tracksMeta")) as fh:
            cls = np.loadtxt(io.TextIOWrapper(fh, "utf-8"), delimiter=",",
                             skiprows=1, usecols=(6,), dtype="U8", ndmin=1)
    out = {n: num[:, i] for i, n in enumerate(_TMETA_NAMES)}
    out["cls"] = cls
    return out


def recordings(zip_path: str = HIGHD_ZIP,
               rids: list | None = None) -> list[Recording]:
    return [read_recording_meta(zip_path, r)
            for r in (rids if rids is not None else recording_ids(zip_path))]


def fleet_stats(zip_path: str = HIGHD_ZIP, rids: list | None = None) -> dict:
    """Class-conditional vehicle dimensions and shares, from metadata alone.

    The model spawns one homogeneous car type (`L=4.5, W=1.8` in
    `state.blank_state`); highD's fleet is ~19 % trucks whose length is ~3x a
    car's, and they dominate the rightmost lane. `scenarios.make_highd_like`
    uses these to draw a realistic mix.
    """
    out = {}
    for cname, key in (("Car", "car"), ("Truck", "truck")):
        L, W = [], []
        for r in (rids if rids is not None else recording_ids(zip_path)):
            tm = read_tracks_meta(zip_path, r)
            m = np.char.lower(tm["cls"].astype(str)) == cname.lower()
            L.extend(tm["len"][m].tolist())
            W.extend(tm["wid"][m].tolist())
        L, W = np.asarray(L), np.asarray(W)
        out[key] = dict(n=int(L.size),
                        L_mean=round(float(L.mean()), 3),
                        L_sd=round(float(L.std()), 3),
                        L_p10=round(float(np.percentile(L, 10)), 3),
                        L_p90=round(float(np.percentile(L, 90)), 3),
                        W_mean=round(float(W.mean()), 3),
                        W_sd=round(float(W.std()), 3))
    n_tot = out["car"]["n"] + out["truck"]["n"]
    out["truck_share"] = round(out["truck"]["n"] / max(n_tot, 1), 4)
    return out


def select_recordings(recs: list, n_lanes: int | None = 3,
                      location: int | None = None) -> list[int]:
    """Recording ids whose *both* carriageways have `n_lanes` lanes."""
    out = []
    for r in recs:
        if n_lanes is not None and not (r.n_lanes_upper == n_lanes
                                        and r.n_lanes_lower == n_lanes):
            continue
        if location is not None and r.location_id != location:
            continue
        out.append(r.rid)
    return out


def meta_stats(zip_path: str = HIGHD_ZIP, rids: list | None = None) -> dict:
    """Dataset-level statistics from metadata alone (~2 s for all 60).

    This is where the lane-change *rate* comes from: `tracksMeta.numLaneChanges`
    summed over every released track, divided by `recordingMeta`'s total driven
    distance. It needs no trajectory file, so the headline number is reproducible
    anywhere in seconds.
    """
    recs = recordings(zip_path, rids)
    per, lc, km, nveh, hrs = {}, 0, 0.0, 0, 0.0
    minthw, minttc, widths = [], [], []
    for r in recs:
        tm = read_tracks_meta(zip_path, r.rid)
        n_lc = int(tm["numLaneChanges"].sum())
        lc += n_lc
        km += r.driven_km
        nveh += r.n_vehicles
        hrs += r.driven_h
        widths.extend(r.lane_widths)
        thw = tm["minTHW"][tm["minTHW"] > 0]
        ttc = tm["minTTC"][tm["minTTC"] > 0]
        minthw.extend(thw.tolist())
        minttc.extend(ttc.tolist())
        sp = np.abs(tm["meanXVelocity"])
        per[f"{r.rid:02d}"] = dict(
            n_lane_changes=n_lc, veh_km=round(r.driven_km, 3),
            lc_per_veh_km=round(n_lc / r.driven_km, 4) if r.driven_km else None,
            mean_speed=round(float(sp.mean()), 3),
            p15_speed=round(float(np.percentile(sp, 15)), 3),
            truck_share=round(r.n_trucks / max(r.n_vehicles, 1), 4),
            flow_veh_h_lane=round(r.n_vehicles / (r.duration_s / 3600.0) / 2.0
                                  / max(r.n_lanes_lower, 1), 1),
            **{k: r.to_dict()[k] for k in
               ("location_id", "n_lanes_upper", "n_lanes_lower",
                "lane_width_med", "duration_s", "speed_limit")})
    return dict(
        n_recordings=len(recs), n_vehicles=nveh, veh_km=round(km, 1),
        driven_h=round(hrs, 1), n_lane_changes=lc,
        lc_per_veh_km=round(lc / km, 4) if km else None,
        lane_width_med=round(float(np.median(widths)), 3),
        lane_width_min=round(float(np.min(widths)), 3),
        lane_width_max=round(float(np.max(widths)), 3),
        n_lane_widths=len(widths),
        min_thw_med=round(float(np.median(minthw)), 3) if minthw else None,
        min_ttc_p05=round(float(np.percentile(minttc, 5)), 3) if minttc else None,
        per_recording=per)


# ---------------------------------------------------------------------------
# trajectory pass
# ---------------------------------------------------------------------------
@dataclass
class Carriageway:
    """One carriageway of one recording, in the model's road frame.

    Flat per-frame arrays sorted by (track, frame), plus `slices` giving each
    track's contiguous span. `marks` is ascending with marks[0] = 0 at the
    right-hand edge, so lane 0 is the rightmost travel lane and +y is to the
    left - `emv/road.py`'s convention.
    """
    rid: int
    direction: str
    marks: np.ndarray
    dt: float
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    thw: np.ndarray
    ttc: np.ndarray
    tid: np.ndarray
    slices: list = field(default_factory=list)
    veh_len: dict = field(default_factory=dict)
    veh_wid: dict = field(default_factory=dict)
    veh_cls: dict = field(default_factory=dict)

    @property
    def n_lanes(self) -> int:
        return int(self.marks.size - 1)

    @property
    def lane_width_med(self) -> float:
        return float(np.median(np.diff(self.marks)))


def read_tracks(zip_path: str, rid: int) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        a = _read_cols(zf, _member(rid, "tracks"), _TRACK_USECOLS,
                       dtype=np.float32)
    return {n: a[:, i] for i, n in enumerate(_TRACK_NAMES)}


def carriageway(zip_path: str, rid: int, direction: str,
                rec: Recording | None = None, tmeta: dict | None = None,
                tracks: dict | None = None, dt: float = REC_DT,
                min_frames: int = 8) -> Carriageway:
    """Read one carriageway into the model's road frame (normalisations 1-3)."""
    rec = rec or read_recording_meta(zip_path, rid)
    tmeta = tmeta or read_tracks_meta(zip_path, rid)
    tracks = tracks or read_tracks(zip_path, rid)
    if direction not in ("upper", "lower"):
        raise ValueError("direction must be 'upper' or 'lower'")

    want = 1.0 if direction == "upper" else 2.0
    ids_dir = tmeta["id"][tmeta["drivingDirection"] == want]
    keep = np.isin(tracks["id"], ids_dir)

    # decimate 25 Hz -> dt exactly; frame numbering is global so tracks stay aligned
    stride = max(1, int(round(rec.frame_rate * dt)))
    keep &= (tracks["frame"].astype(np.int64) % stride == 1)

    fr = tracks["frame"][keep]
    tid = tracks["id"][keep].astype(np.int64)
    xc = tracks["x"][keep] + 0.5 * tracks["len"][keep]     # normalisation 1
    yc = tracks["y"][keep] + 0.5 * tracks["wid"][keep]
    vx, vy = tracks["vx"][keep], tracks["vy"][keep]
    ax, ay = tracks["ax"][keep], tracks["ay"][keep]

    marks = np.asarray(rec.marks(direction), dtype=float)
    if direction == "upper":                               # normalisation 2
        x_m, y_m = -xc, yc - marks.min()
        vx_m, vy_m, ax_m, ay_m = -vx, vy, -ax, ay
        marks_m = marks - marks.min()
    else:
        x_m, y_m = xc, marks.max() - yc
        vx_m, vy_m, ax_m, ay_m = vx, -vy, ax, -ay
        marks_m = (marks.max() - marks)[::-1]

    order = np.lexsort((fr, tid))
    fr, tid = fr[order], tid[order]
    x_m, y_m = np.asarray(x_m)[order], np.asarray(y_m)[order]
    vx_m, vy_m = np.asarray(vx_m)[order], np.asarray(vy_m)[order]
    ax_m, ay_m = np.asarray(ax_m)[order], np.asarray(ay_m)[order]
    thw, ttc = tracks["thw"][keep][order], tracks["ttc"][keep][order]
    t = fr.astype(float) / rec.frame_rate

    bounds = np.flatnonzero(np.r_[True, np.diff(tid) != 0, True])
    slices = [(int(tid[bounds[i]]), slice(int(bounds[i]), int(bounds[i + 1])))
              for i in range(bounds.size - 1)
              if bounds[i + 1] - bounds[i] >= min_frames]

    idx = {int(i): k for k, i in enumerate(tmeta["id"])}
    veh_len = {int(i): float(tmeta["len"][idx[int(i)]]) for i, _ in slices}
    veh_wid = {int(i): float(tmeta["wid"][idx[int(i)]]) for i, _ in slices}
    veh_cls = {int(i): str(tmeta["cls"][idx[int(i)]]) for i, _ in slices}

    return Carriageway(rid=rid, direction=direction, marks=marks_m, dt=dt,
                       t=t, x=x_m, y=y_m, vx=vx_m, vy=vy_m, ax=ax_m, ay=ay_m,
                       thw=thw, ttc=ttc, tid=tid, slices=slices,
                       veh_len=veh_len, veh_wid=veh_wid, veh_cls=veh_cls)


def carriageways(zip_path: str, rid: int, dt: float = REC_DT,
                 min_frames: int = 8) -> list[Carriageway]:
    rec = read_recording_meta(zip_path, rid)
    tmeta = read_tracks_meta(zip_path, rid)
    tracks = read_tracks(zip_path, rid)
    return [carriageway(zip_path, rid, d, rec, tmeta, tracks, dt, min_frames)
            for d in ("upper", "lower")]


# ---------------------------------------------------------------------------
# observables - same names, same definitions as emv.metrics.behaviour_stats
# ---------------------------------------------------------------------------
def _pct(a, q):
    a = np.asarray(a, dtype=float)
    return float(np.percentile(a, q)) if a.size else float("nan")


def lane_changes(cw: Carriageway, persist: float = 0.5,
                 settle: float = 0.6, fov_margin: float = 0.0) -> list:
    """Lane changes of every track, via `metrics.lane_change_from_track`.

    Each event additionally carries `rid`, `track`, `v_mean`, `cls` and
    `complete` - the last being True when the whole manoeuvre lies strictly
    inside the tracked section, which is the like-for-like counterpart of the
    published highD lane-change count (see `docs/HIGHD_VALIDATION.md`).
    """
    x_lo, x_hi = cw.x.min() + fov_margin, cw.x.max() - fov_margin
    out = []
    for tid, sl in cw.slices:
        y, vy, t = cw.y[sl], cw.vy[sl], cw.t[sl]
        evs = metrics.lane_change_from_track(
            t, y, vy, cw.veh_wid[tid], aware=None, persist=persist,
            settle=settle, veh=tid, marks=cw.marks)
        if not evs:
            continue
        xs, ts = cw.x[sl], t
        v_mean = float(np.abs(cw.vx[sl]).mean())
        for e in evs:
            i0 = int(np.searchsorted(ts, e["t_start"]))
            i1 = min(int(np.searchsorted(ts, e["t_end"])), xs.size - 1)
            e.update(rid=cw.rid, track=int(tid), v_mean=v_mean,
                     cls=cw.veh_cls[tid],
                     complete=bool(xs[i0] > x_lo and xs[i1] < x_hi
                                   and ts[0] < e["t_start"]
                                   and ts[-1] > e["t_end"]))
            out.append(e)
    return out


def track_pools(cw: Carriageway, persist: float = 0.5,
                settle: float = 0.6) -> dict:
    """Per-track scalars and per-frame value pools (sufficient statistics).

    Lateral acceleration is `|diff(vy)|/dt`, the *same* finite difference
    `metrics.behaviour_stats` applies to model output, not highD's own smoothed
    `yAcceleration` column - which is reported alongside as `alat_dataset` so the
    two conventions can be compared rather than conflated.

    Two measurement caveats found in the data and handled here:

    * **Lateral-acceleration quantization.** `yVelocity` is stored to 0.01 m/s,
      so `|dvy|/0.12` is quantized to 0.0833 m/s^2. The *median* per-track peak
      lands on exactly 2 quanta (0.1667) in every carriageway tested, and
      widening the difference to 0.48 s does not lift it, so the median is
      resolution-limited and must not be used as a calibration target - the p90
      and p99 (0.36-0.42) carry the signal.
    * **Longitudinal-acceleration bias.** highD's x scale drifts slightly per
      flight, which shows up as a direction-antisymmetric offset in forward
      acceleration (rec 01: +0.000 upper vs +0.083 lower). Longitudinal
      accelerations are therefore de-biased by this carriageway's median before
      any percentile is taken; the raw offset is returned as `accel_bias`.
    """
    dt = cw.dt
    peak_dec, peak_alat, peak_vy, mean_vx = [], [], [], []
    offs, n_frames, cls, vy_pool, alat_pool, off_pool = [], [], [], [], [], []
    alat_ds_pool, thw_pool, ttc_pool, dec_pool = [], [], [], []
    centres = 0.5 * (cw.marks[:-1] + cw.marks[1:])
    veh_km = 0.0
    for tid, sl in cw.slices:
        y, vy, vx = cw.y[sl], cw.vy[sl], cw.vx[sl]
        if vx.size < 2:
            continue
        veh_km += float(np.abs(np.diff(cw.x[sl])).sum()) / 1000.0
        a_lon = np.diff(vx) / dt
        a_lat = np.abs(np.diff(vy)) / dt
        peak_dec.append(float(-a_lon.min()))
        peak_alat.append(float(a_lat.max()))
        peak_vy.append(float(np.abs(vy).max()))
        mean_vx.append(float(vx.mean()))
        n_frames.append(int(vx.size))
        cls.append(1 if cw.veh_cls[tid].lower().startswith("t") else 0)
        # lane-keeping: distance to the current lane centre while the body is
        # NOT straddling a marking (a lane change is not lane keeping)
        li = np.clip(np.searchsorted(cw.marks, y, side="right") - 1,
                     0, cw.marks.size - 2)
        d_mark = np.min(np.abs(y[:, None] - cw.marks[None, :]), axis=1)
        keeping = d_mark >= 0.5 * cw.veh_wid[tid]
        if keeping.any():
            off = np.abs(y[keeping] - centres[li[keeping]])
            offs.append(float(off.mean()))
            off_pool.append(off.astype(np.float32))
        vy_pool.append(np.abs(vy).astype(np.float32))
        alat_pool.append(a_lat.astype(np.float32))
        alat_ds_pool.append(np.abs(cw.ay[sl]).astype(np.float32))
        dec_pool.append(a_lon.astype(np.float32))
        th, tt = cw.thw[sl], cw.ttc[sl]
        thw_pool.append(th[(th > 0) & (th < 20)].astype(np.float32))
        ttc_pool.append(tt[(tt > 0) & (tt < 60)].astype(np.float32))
    cat = lambda L: (np.concatenate(L) if L else np.zeros(0, np.float32))
    # de-bias longitudinal acceleration (see the docstring): the offset is a
    # property of the flight's x scale, not of driver behaviour
    accel = cat(dec_pool)
    bias = float(np.median(accel)) if accel.size else 0.0
    accel = (accel - bias).astype(np.float32)
    peak_dec = [pd + bias for pd in peak_dec]
    return dict(
        veh_km=veh_km, n_tracks=len(peak_dec), accel_bias=bias,
        trk_peak_decel=np.asarray(peak_dec, np.float32),
        trk_peak_alat=np.asarray(peak_alat, np.float32),
        trk_peak_vy=np.asarray(peak_vy, np.float32),
        trk_mean_vx=np.asarray(mean_vx, np.float32),
        trk_offset_abs=np.asarray(offs, np.float32),
        trk_n_frames=np.asarray(n_frames, np.int32),
        trk_class=np.asarray(cls, np.int8),
        vy=cat(vy_pool), alat=cat(alat_pool), alat_dataset=cat(alat_ds_pool),
        offset=cat(off_pool), accel=accel,
        thw=cat(thw_pool), ttc=cat(ttc_pool))


def behaviour_stats_highd(cw: Carriageway, persist: float = 0.5,
                          settle: float = 0.6, pools: dict | None = None,
                          lcs: list | None = None) -> dict:
    """highD counterpart of `metrics.behaviour_stats`, same keys, same rules.

    EV-dependent keys of the model function (`ev_*`, `yield_*`, `lc_yield_share`,
    `n_yielded`) are absent by construction: highD has no emergency vehicle.
    """
    p = pools if pools is not None else track_pools(cw, persist, settle)
    lcs = lcs if lcs is not None else lane_changes(cw, persist, settle)
    dur = np.array([e["duration"] for e in lcs], float)
    pvy = np.array([e["peak_vy"] for e in lcs], float)
    dur_c = np.array([e["duration"] for e in lcs if e["complete"]], float)
    n_complete = int(sum(1 for e in lcs if e["complete"]))
    # full-manoeuvre window, untruncated events only - the variant comparable
    # with dataset/literature lane-change durations
    full = [e for e in lcs if not e["truncated"] and e["complete"]]
    durf = np.array([e["duration_full"] for e in full], float)
    c2cf = np.array([e["dur_c2c_full"] for e in full], float)
    dyf = np.array([e["dy_full"] for e in full], float)
    pvyf = np.array([e["peak_vy_full"] for e in full], float)
    v_bg, a_bg = p["trk_mean_vx"], p["accel"]
    obs_dur = p["trk_n_frames"].astype(float) * cw.dt
    return dict(
        bg_med_speed=_pct(v_bg, 50), bg_p15_speed=_pct(v_bg, 15),
        accel_p99=_pct(a_bg, 99), decel_p01=-_pct(a_bg, 1),
        peak_decel_med=_pct(p["trk_peak_decel"], 50),
        peak_decel_p90=_pct(p["trk_peak_decel"], 90),
        peak_decel_p99=_pct(p["trk_peak_decel"], 99),
        peak_alat_med=_pct(p["trk_peak_alat"], 50),
        peak_alat_p90=_pct(p["trk_peak_alat"], 90),
        peak_alat_p99=_pct(p["trk_peak_alat"], 99),
        alat_p99=_pct(p["alat"], 99),
        alat_dataset_p99=_pct(p["alat_dataset"], 99),
        lat_speed_med=_pct(p["vy"], 50), lat_speed_p90=_pct(p["vy"], 90),
        n_lane_changes=len(lcs), n_lane_changes_complete=n_complete,
        lc_per_veh_km=(len(lcs) / p["veh_km"] if p["veh_km"] > 0
                       else float("nan")),
        lc_per_veh_km_complete=(n_complete / p["veh_km"] if p["veh_km"] > 0
                                else float("nan")),
        lc_dur_med=_pct(dur, 50), lc_dur_p10=_pct(dur, 10),
        lc_dur_p90=_pct(dur, 90),
        lc_dur_mean=float(dur.mean()) if dur.size else float("nan"),
        lc_dur_sd=float(dur.std()) if dur.size else float("nan"),
        lc_dur_complete_med=_pct(dur_c, 50),
        lc_peak_vy_med=_pct(pvy, 50), lc_peak_vy_p90=_pct(pvy, 90),
        lc_dur_full_med=_pct(durf, 50), lc_dur_full_p10=_pct(durf, 10),
        lc_dur_full_p90=_pct(durf, 90),
        lc_dur_full_mean=float(durf.mean()) if durf.size else float("nan"),
        lc_dur_full_sd=float(durf.std()) if durf.size else float("nan"),
        lc_c2c_med=_pct(c2cf, 50), lc_c2c_p90=_pct(c2cf, 90),
        lc_dy_full_med=_pct(dyf, 50),
        lc_peak_vy_full_med=_pct(pvyf, 50),
        lc_peak_vy_full_p90=_pct(pvyf, 90),
        n_lane_changes_full=int(durf.size),
        lc_truncated_share=(float(np.mean([e["truncated"] for e in lcs]))
                            if lcs else 0.0),
        accel_bias=float(p.get("accel_bias", 0.0)),
        lane_offset_mean=(float(p["trk_offset_abs"].mean())
                          if p["trk_offset_abs"].size else float("nan")),
        lane_offset_p90=_pct(p["trk_offset_abs"], 90),
        thw_med=_pct(p["thw"], 50), thw_p15=_pct(p["thw"], 15),
        thw_p85=_pct(p["thw"], 85), ttc_p01=_pct(p["ttc"], 1),
        ttc_p05=_pct(p["ttc"], 5),
        truck_share=(float(p["trk_class"].mean())
                     if p["trk_class"].size else float("nan")),
        obs_dur_med=_pct(obs_dur, 50), n_tracks=int(p["n_tracks"]),
        veh_km=float(p["veh_km"]), lane_width_med=cw.lane_width_med,
        n_lanes=cw.n_lanes, **_demand(cw, p))


def _demand(cw: Carriageway, p: dict) -> dict:
    """Flow, density and regime label of one carriageway.

    Flow is tracks per hour per lane over the observed span; density follows from
    flow and the pooled median speed. Both are needed to place the carriageway in
    a regime and to set `scenarios.make_highd_like`'s spacing.
    """
    span = float(cw.t.max() - cw.t.min()) if cw.t.size else 0.0
    v = float(np.median(p["trk_mean_vx"])) if p["trk_mean_vx"].size else float("nan")
    flow = (p["n_tracks"] / (span / 3600.0) / cw.n_lanes
            if span > 0 else float("nan"))
    dens = flow / (v * 3.6) * 1000.0 / 1000.0 if v and np.isfinite(v) and v > 0.5 \
        else float("nan")
    return dict(flow_veh_h_lane=flow, density_veh_km_lane=dens,
                span_s=span, regime=classify_regime(v, flow) or "other")


def lane_speed_profile(cw: Carriageway) -> dict:
    """Per-lane desired-speed statistics (lane 0 = rightmost), for
    `scenarios.make_highd_like`: real motorways are strongly lane-stratified
    (right lane truck-dominated and ~15 m/s slower than the left)."""
    li_of_track, v_of_track, cls_of_track = [], [], []
    for tid, sl in cw.slices:
        y = cw.y[sl]
        li = np.clip(np.searchsorted(cw.marks, y, side="right") - 1,
                     0, cw.marks.size - 2)
        li_of_track.append(int(np.bincount(li, minlength=cw.n_lanes).argmax()))
        v_of_track.append(float(np.abs(cw.vx[sl]).mean()))
        cls_of_track.append(1 if cw.veh_cls[tid].lower().startswith("t") else 0)
    li = np.asarray(li_of_track)
    v = np.asarray(v_of_track, float)
    c = np.asarray(cls_of_track)
    out = dict(lane_mean=[], lane_sd=[], lane_med=[], truck_share=[], n=[])
    for k in range(cw.n_lanes):
        m = li == k
        out["lane_mean"].append(float(v[m].mean()) if m.any() else float("nan"))
        out["lane_sd"].append(float(v[m].std()) if m.sum() > 1 else float("nan"))
        out["lane_med"].append(_pct(v[m], 50))
        out["truck_share"].append(float(c[m].mean()) if m.any() else 0.0)
        out["n"].append(int(m.sum()))
    return out


def window_records(cw: Carriageway, win: float = 60.0) -> list:
    """Per-time-window regime records for one carriageway.

    Windowing is per carriageway on purpose: one direction can be jammed while
    the other flows freely (recording 25 is the standing example), so a
    recording-level mean speed hides both regimes.
    """
    if cw.t.size == 0:
        return []
    t0 = float(cw.t.min())
    out = []
    wid = np.floor((cw.t - t0) / win).astype(int)
    lane_w = cw.lane_width_med
    for w in np.unique(wid):
        m = wid == w
        v = np.abs(cw.vx[m])
        if v.size < 50:
            continue
        n_tracks = int(np.unique(cw.tid[m]).size)
        out.append(dict(
            rid=cw.rid, direction=cw.direction, window=int(w),
            t0=round(t0 + w * win, 2), n_obs=int(v.size), n_tracks=n_tracks,
            med_speed=round(float(np.median(v)), 3),
            p15_speed=round(float(np.percentile(v, 15)), 3),
            frac_below_10=round(float((v < 10).mean()), 4),
            flow_veh_h_lane=round(n_tracks / (win / 3600.0) / cw.n_lanes, 1),
            n_lanes=cw.n_lanes, lane_width_med=round(lane_w, 3)))
    return out


def recording_stats(zip_path: str, rid: int, dt: float = REC_DT,
                    persist: float = 0.5, settle: float = 0.6,
                    win: float = 60.0, min_frames: int = 8) -> dict:
    """Full per-recording block: observables per carriageway + pooled + windows."""
    rec = read_recording_meta(zip_path, rid)
    tmeta = read_tracks_meta(zip_path, rid)
    tracks = read_tracks(zip_path, rid)
    obs, pools, lcs_all, wins, lanes = {}, {}, [], [], {}
    for d in ("upper", "lower"):
        cw = carriageway(zip_path, rid, d, rec, tmeta, tracks, dt, min_frames)
        if not cw.slices:
            continue
        p = track_pools(cw, persist, settle)
        lc = lane_changes(cw, persist, settle)
        obs[d] = behaviour_stats_highd(cw, persist, settle, pools=p, lcs=lc)
        # event-level pools, so the distributions behind the lane-change
        # percentiles survive into out/highd_pools.npz and can be plotted or
        # used for a distributional distance without re-reading the CSVs
        keep = [e for e in lc if not e["truncated"] and e["complete"]]
        f32 = lambda a: np.asarray(a, dtype=np.float32)
        p.update(lc_dur=f32([e["duration"] for e in lc]),
                 lc_peak_vy=f32([e["peak_vy"] for e in lc]),
                 lc_dur_full=f32([e["duration_full"] for e in keep]),
                 lc_c2c=f32([e["dur_c2c_full"] for e in keep]),
                 lc_peak_vy_full=f32([e["peak_vy_full"] for e in keep]))
        pools[d] = p
        lanes[d] = lane_speed_profile(cw)
        lcs_all.extend(lc)
        wins.extend(window_records(cw, win))
    return dict(rid=rid, meta=rec.to_dict(), observables=obs,
                lane_profile=lanes, windows=wins, pools=pools, events=lcs_all)


# ---------------------------------------------------------------------------
# pooling
# ---------------------------------------------------------------------------
def pool_pools(pool_list: list) -> dict:
    keys = set().union(*(set(p) for p in pool_list)) if pool_list else set()
    out = {}
    for k in keys:
        vals = [p[k] for p in pool_list if k in p]
        if isinstance(vals[0], np.ndarray):
            out[k] = np.concatenate(vals)
        else:
            out[k] = float(np.sum(vals))
    return out


def bootstrap_ci(a, n: int = 2000, q=(2.5, 97.5), seed: int = 7):
    """Percentile bootstrap of the mean (same design as
    `experiments/run_param_stats.py:boot_ci`, kept here so this module has no
    experiment-script dependency)."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n, a.size))
    means = a[idx].mean(axis=1)
    return (float(a.mean()), float(np.percentile(means, q[0])),
            float(np.percentile(means, q[1])))


def cache_is_valid(cached: dict, config: dict, fingerprint: dict) -> bool:
    prov = cached.get("provenance", {})
    return (prov.get("extractor_version") == EXTRACTOR_VERSION
            and prov.get("crc_digest") == fingerprint.get("crc_digest")
            and cached.get("config") == config)
