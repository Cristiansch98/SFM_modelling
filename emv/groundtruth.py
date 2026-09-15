"""Ground-truth behaviour interface to the emergency-vehicle-dataset-pipeline.

The pipeline (``emergency-vehicle-dataset-pipeline-main`` on the Desktop)
processes dashcam video from German ambulances on real emergency runs into
one JSON per second: per-vehicle ego-frame kinematics plus a manually
reviewed behaviour label (normal / yielded / braked_abruptly /
failed_to_yield).  Those labelled interactions are the behavioural ground
truth this framework is compared against (and can be finetuned on).

Everything here is a convention-exact port of the pipeline so simulated
traffic is measured by EXACTLY the rules that measured the real data:

* :class:`HeuristicAnnotator` - verbatim port of the pipeline's
  ``annotator.py`` (same thresholds, same rule order, same history
  semantics: counters update even beyond the 50 m annotation range).
* :class:`EgoKinematics` - the pipeline's finite-difference chain
  (``homography.estimate_relative_velocity/acceleration/jerk`` at 1 Hz,
  including its rounding and first-frame zeros) plus its symmetric-road
  lane / lateral-offset assignment.
* :func:`fingerprint` / :func:`fingerprint_distance` - reduce any labelled
  ego-frame stream (real or simulated) to comparable behaviour statistics
  and score sim-vs-real distance.

Ego-frame convention (pipeline README): ``x`` = lateral metres, + = right
of the EV; ``y`` = forward metres ahead of the EV; speeds are RELATIVE to
the EV (per-second position differences).  The SUMO bridge's ego recorder
(`emv/sumo/bridge.py`) emits the same frame.
"""
import glob
import json
import os

import numpy as np

#: default location of the pipeline's processed output (one dir per video)
PIPELINE_OUT = os.path.join(
    os.path.expanduser("~"), "Desktop",
    "emergency-vehicle-dataset-pipeline-main",
    "emergency-vehicle-dataset-pipeline-main", "output")

LABELS = ("normal", "yielded", "braked_abruptly", "failed_to_yield")


# ---------------------------------------------------------------------------
# 1. annotator - verbatim port of pipeline annotator.py (thresholds cited
#    there: Pierson et al. 2019, Krajewski et al. 2018, Cortes & Stefoni 2023)
# ---------------------------------------------------------------------------
class HeuristicAnnotator:
    """Labels one vehicle observation per frame; keeps per-track history.

    Port notes: rule order (brake > sustained lateral > cumulative drift >
    failed_to_yield), the centre dead band, and the fact that history and
    run counters update BEFORE the 50 m proximity gate are all preserved
    from the pipeline source.
    """

    YIELD_LATERAL_SPEED = 0.5      # m/s
    YIELD_PERSIST = 2              # consecutive frames
    CENTRE_DEAD_BAND = 0.5         # m
    YIELD_CUMULATIVE = 0.8         # m
    CUMULATIVE_WINDOW = 3          # s (= frames at 1 Hz)
    ABRUPT_BRAKE_THRESHOLD = -2.5  # m/s^2
    BRAKE_ONSET_ACCEL = -1.5       # m/s^2
    BRAKE_ONSET_JERK = -3.0        # m/s^3
    PROXIMITY_THRESHOLD = 50.0     # m
    FAILED_YIELD_PROXIMITY = 20.0  # m
    MIN_OBSERVED_FRAMES = 3        # frames

    def __init__(self):
        self.lateral_history = {}
        self.frames_seen = {}
        self.lateral_run = {}

    def annotate(self, vehicle: dict, emergency_active: bool = True) -> str:
        if not emergency_active:
            return "normal"

        tid = vehicle["id"]
        lateral_spd = vehicle.get("lateral_speed_ms", 0.0)
        x_pos = vehicle.get("x_meters", 0.0)
        acceleration = vehicle.get("acceleration", 0.0)
        jerk = vehicle.get("jerk", 0.0)
        distance = vehicle.get("distance_to_ego", 999.0)
        curr_lateral = vehicle.get("lateral_offset", 0.0)

        self.frames_seen[tid] = self.frames_seen.get(tid, 0) + 1

        self.lateral_history.setdefault(tid, []).append(curr_lateral)
        if len(self.lateral_history[tid]) > self.CUMULATIVE_WINDOW:
            self.lateral_history[tid].pop(0)

        # directional filter: outside the dead band, motion toward the
        # EV's path (toward x = 0) is not a yield
        if (abs(x_pos) > self.CENTRE_DEAD_BAND
                and np.sign(lateral_spd) != np.sign(x_pos)):
            yield_lateral = 0.0
        else:
            yield_lateral = abs(lateral_spd)

        if yield_lateral >= self.YIELD_LATERAL_SPEED:
            self.lateral_run[tid] = self.lateral_run.get(tid, 0) + 1
        else:
            self.lateral_run[tid] = 0

        if distance > self.PROXIMITY_THRESHOLD:
            return "normal"

        if acceleration <= self.ABRUPT_BRAKE_THRESHOLD:
            return "braked_abruptly"
        if (acceleration <= self.BRAKE_ONSET_ACCEL
                and jerk <= self.BRAKE_ONSET_JERK):
            return "braked_abruptly"

        if self.lateral_run.get(tid, 0) >= self.YIELD_PERSIST:
            return "yielded"

        if len(self.lateral_history[tid]) >= self.CUMULATIVE_WINDOW:
            history = self.lateral_history[tid]
            total = abs(history[-1] - history[0])
            direction = history[-1] - history[0]
            consistent = all(
                (history[i + 1] - history[i]) * direction >= 0
                for i in range(len(history) - 1))
            if total >= self.YIELD_CUMULATIVE and consistent:
                return "yielded"

        if (distance <= self.FAILED_YIELD_PROXIMITY
                and self.frames_seen[tid] >= self.MIN_OBSERVED_FRAMES):
            return "failed_to_yield"

        return "normal"


# ---------------------------------------------------------------------------
# 2. kinematics - port of the pipeline's finite-difference chain (dt = 1 s)
# ---------------------------------------------------------------------------
class EgoKinematics:
    """Derives the pipeline's kinematic fields from 1 Hz ego-frame positions.

    Mirrors homography.py exactly: first observation of a track gets zero
    speed/accel/jerk; speeds are rounded to 2 decimals BEFORE feeding the
    acceleration difference (the pipeline chains the rounded values); the
    road is assumed symmetric about the EV for lane / lateral-offset.
    """

    def __init__(self, n_lanes: int = 3, lane_width: float = 3.75):
        self.n_lanes = n_lanes
        self.lane_width = lane_width
        self.prev_pos = {}
        self.prev_speed = {}
        self.prev_accel = {}

    def step(self, tid, x: float, y: float, dt: float = 1.0) -> dict:
        prev = self.prev_pos.get(tid)
        self.prev_pos[tid] = (x, y)
        if prev is None:
            fwd, lat, kmh = 0.0, 0.0, 0.0
        else:
            dx, dy = x - prev[0], y - prev[1]
            lat = round(dx / dt, 2)
            fwd = round(dy / dt, 2)
            kmh = round(float(np.hypot(dx, dy)) / dt * 3.6, 2)

        acc = round((fwd - self.prev_speed.get(tid, fwd)) / dt, 3)
        self.prev_speed[tid] = fwd
        jerk = round((acc - self.prev_accel.get(tid, acc)) / dt, 3)
        self.prev_accel[tid] = acc

        half = 0.5 * self.n_lanes * self.lane_width
        lane = min(max(int((x + half) / self.lane_width) + 1, 1), self.n_lanes)
        lane_centre = -half + (lane - 0.5) * self.lane_width

        return dict(
            x_meters=round(x, 2), y_meters=round(y, 2),
            forward_speed_ms=fwd, lateral_speed_ms=lat, speed_kmh=kmh,
            acceleration=acc, jerk=jerk,
            distance_to_ego=round(float(np.hypot(x, y)), 2),
            lane_id=lane, lateral_offset=round(x - lane_centre, 2))


def annotate_stream(frames: list, n_lanes: int = 3,
                    lane_width: float = 3.5) -> list:
    """Label a simulated ego-frame stream exactly like the pipeline.

    ``frames`` = ``[{"t": .., "vehicles": [{"id", "x", "y"}, ..]}, ..]``
    (the SUMO bridge's ``ego_frames``).  Returns new frames whose vehicles
    carry the full pipeline field set plus ``behaviour``.
    """
    kin = EgoKinematics(n_lanes=n_lanes, lane_width=lane_width)
    ann = HeuristicAnnotator()
    out = []
    for fr in frames:
        vehs = []
        for v in fr["vehicles"]:
            rec = kin.step(v["id"], v["x"], v["y"])
            rec["id"] = v["id"]
            rec["behaviour"] = ann.annotate(rec, emergency_active=True)
            vehs.append(rec)
        out.append(dict(t=fr["t"], vehicles=vehs))
    return out


def ego_stream(h, fov: float = 120.0, rate: float = 1.0) -> list:
    """1 Hz ego-frame stream from a `simulate.History`, in the annotator's frame.

    Moved verbatim from `experiments/run_surrogate_sumo.py` (2026-08-06) so the
    standalone model can be scored against the dashcam ground truth without a
    SUMO installation - the same behaviour-safe relocation session 5 made with
    `make_sumo_like`. It reproduces the bridge's recorder
    (`emv/sumo/bridge.py:_record_ego`): `x` is lateral (+ = right of the EV) and
    `y` is forward from the EV's front bumper to the target's rear.
    """
    ev = h.ev
    half = 0.5 * h.road.n_lanes * h.road.lane_width
    rec_dt = float(h.t[1] - h.t[0])
    step = max(1, int(round(rate / rec_dt)))
    frames = []
    for k in range(0, h.n_frames, step):
        fwd = (h.x[k] - 0.5 * h.L) - (h.x[k, ev] + 0.5 * h.L[ev])
        lat = h.y[k, ev] - h.y[k]
        keep = (fwd > 0.0) & (fwd <= fov) & (np.abs(lat) <= half + 3.0)
        keep[ev] = False
        frames.append(dict(
            t=float(h.t[k]),
            vehicles=[dict(id=int(i), x=round(float(lat[i]), 2),
                           y=round(float(fwd[i]), 2))
                      for i in np.flatnonzero(keep)]))
    return frames


# ---------------------------------------------------------------------------
# 3. ground-truth loading
# ---------------------------------------------------------------------------
def gt_videos(out_dir: str = PIPELINE_OUT) -> list:
    return sorted(d for d in glob.glob(os.path.join(out_dir, "*"))
                  if os.path.isdir(d)
                  and glob.glob(os.path.join(d, "t*.json")))


def load_gt_stream(video_dir: str) -> list:
    """Load one processed video into the frame-stream format.

    Keeps the pipeline's stored kinematics and (manually reviewed) labels -
    those ARE the ground truth; we do not re-derive them.
    """
    frames = []
    for path in sorted(glob.glob(os.path.join(video_dir, "t*.json"))):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if not d.get("emergency_active", False):
            continue
        vehs = []
        for v in d.get("vehicles", []):
            vehs.append(dict(
                id=v["id"], x_meters=v.get("x_meters", 0.0),
                y_meters=v.get("y_meters", 0.0),
                lateral_speed_ms=v.get("lateral_speed_ms", 0.0) or 0.0,
                forward_speed_ms=v.get("forward_speed_ms", 0.0) or 0.0,
                acceleration=v.get("acceleration", 0.0) or 0.0,
                jerk=v.get("jerk", 0.0) or 0.0,
                distance_to_ego=v.get("distance_to_ego", 999.0),
                lateral_offset=v.get("lateral_offset", 0.0) or 0.0,
                behaviour=v.get("behaviour", "normal"),
                reliable=bool(v.get("position_reliable", True))))
        frames.append(dict(t=d.get("timestamp", len(frames)), vehicles=vehs))
    return frames


def reannotate(frames: list) -> list:
    """Relabel a stream with the pure kinematic rules (fresh annotator).

    On ground-truth data this strips the manual-review corrections (~20 %
    of labels, almost all upgrades to 'yielded'), giving the apples-to-
    apples reference for simulated streams, which only ever get rule
    labels. Kinematics are left untouched.
    """
    ann = HeuristicAnnotator()
    out = []
    for fr in frames:
        vehs = []
        for v in fr["vehicles"]:
            w = dict(v)
            w["behaviour"] = ann.annotate(w, emergency_active=True)
            vehs.append(w)
        out.append(dict(t=fr["t"], vehicles=vehs))
    return out


# ---------------------------------------------------------------------------
# 4. behaviour fingerprint + distance
# ---------------------------------------------------------------------------
def fingerprint(frames: list) -> dict:
    """Reduce a labelled ego-frame stream to comparable behaviour statistics.

    All shares are computed over observations within the annotator's 50 m
    range (beyond it every label is 'normal' by construction).  Track-level
    shares use only "assessable" tracks (>= MIN_OBSERVED_FRAMES frames in
    range) because tracker ID churn fragments real tracks; obs-level shares
    are robust to fragmentation and get more weight in the distance.
    """
    R = HeuristicAnnotator.PROXIMITY_THRESHOLD
    obs = []                      # in-range observations
    tracks = {}                   # tid -> dict(n, labels, onset_dist, order)
    for fr in frames:
        for v in fr["vehicles"]:
            if v.get("distance_to_ego", 999.0) > R:
                continue
            obs.append(v)
            tr = tracks.setdefault(v["id"], dict(n=0, labels=set(),
                                                 onset=None))
            tr["n"] += 1
            tr["labels"].add(v["behaviour"])
            if v["behaviour"] == "yielded" and tr["onset"] is None:
                tr["onset"] = v["distance_to_ego"]

    n_obs = len(obs)
    counts = {lb: sum(1 for v in obs if v["behaviour"] == lb)
              for lb in LABELS}
    assessable = {tid: tr for tid, tr in tracks.items()
                  if tr["n"] >= HeuristicAnnotator.MIN_OBSERVED_FRAMES}
    n_ass = len(assessable)
    trk_y = sum(1 for tr in assessable.values() if "yielded" in tr["labels"])
    trk_f = sum(1 for tr in assessable.values()
                if "failed_to_yield" in tr["labels"]
                and "yielded" not in tr["labels"])
    trk_b = sum(1 for tr in assessable.values()
                if "braked_abruptly" in tr["labels"])

    onsets = [tr["onset"] for tr in tracks.values() if tr["onset"] is not None]
    lat_y = [abs(v.get("lateral_speed_ms", 0.0)) for v in obs
             if v["behaviour"] == "yielded"]
    accels = [v.get("acceleration", 0.0) for v in obs]

    def share(x, n):
        return float(x) / n if n else float("nan")

    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else float("nan")

    return dict(
        n_frames=len(frames), n_obs=n_obs, n_tracks=len(tracks),
        n_tracks_assessable=n_ass,
        veh_per_frame=share(n_obs, len(frames)),
        obs_yielded=share(counts["yielded"], n_obs),
        obs_failed=share(counts["failed_to_yield"], n_obs),
        obs_braked=share(counts["braked_abruptly"], n_obs),
        obs_normal=share(counts["normal"], n_obs),
        trk_yielded=share(trk_y, n_ass),
        trk_failed=share(trk_f, n_ass),
        trk_braked=share(trk_b, n_ass),
        onset_dist_med=pct(onsets, 50), onset_dist_p90=pct(onsets, 90),
        lat_speed_med=pct(lat_y, 50), lat_speed_p90=pct(lat_y, 90),
        accel_p05=pct(accels, 5),
    )


#: (key, scale, weight) - scale normalizes |sim - real| to O(1);
#: obs-level shares carry the most weight (robust to GT track churn)
_DIST_SPEC = (
    ("obs_yielded", 0.25, 1.0), ("obs_failed", 0.25, 1.0),
    ("obs_braked", 0.25, 1.0), ("obs_normal", 0.25, 0.5),
    ("trk_yielded", 0.25, 0.5), ("trk_failed", 0.25, 0.5),
    ("onset_dist_med", 15.0, 1.0), ("lat_speed_med", 0.4, 1.0),
    ("lat_speed_p90", 0.6, 0.5), ("accel_p05", 2.0, 0.5),
)


def fingerprint_distance(fp_sim: dict, fp_gt: dict) -> dict:
    """Weighted normalized distance between two fingerprints.

    Returns per-metric normalized deviations plus 'total' (weighted mean;
    0 = identical behaviour statistics, ~1 = off by one full scale unit
    on average).  NaN metrics (e.g. no yields at all) are penalized at 2
    scale units so degenerate behaviour cannot score well.
    """
    parts, wsum, acc = {}, 0.0, 0.0
    for key, scale, w in _DIST_SPEC:
        a, b = fp_sim.get(key), fp_gt.get(key)
        if a is None or b is None or np.isnan(b):
            continue
        d = 2.0 if np.isnan(a) else abs(a - b) / scale
        parts[key] = round(float(d), 3)
        acc += w * d
        wsum += w
    parts["total"] = round(acc / wsum, 4) if wsum else float("nan")
    return parts
