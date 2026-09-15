"""Extract empirical calibration targets from the real highD trajectories.

Two passes, because the headline number needs no trajectory file at all:

  meta  All 120 metadata CSVs (~2 s). Gives the lane-change *rate* - every
        released track's `numLaneChanges` over `recordingMeta`'s total driven
        distance - plus geometry, truck share, per-recording flow and speed.
  traj  The 60 `tracks.csv`, one at a time, streamed from the zip (~5 min).
        Gives manoeuvre kinematics: two-lane-occupancy lane-change durations,
        peak lateral speed and acceleration, lane-keeping offset, headway and
        TTC distributions - all under the key names and definitions of
        `emv.metrics.behaviour_stats`, and with lane changes timed by the very
        function the model uses (`metrics.lane_change_from_track`).

Regimes are binned per carriageway per 60 s window, not per recording: one
carriageway can be jammed while the other flows freely (recording 25 is the
standing example), so a recording-level mean hides both regimes.

    python experiments/run_highd_extract.py                 # full,  ~5 min
    python experiments/run_highd_extract.py --only meta      # rate,  ~2 s
    python experiments/run_highd_extract.py --quick          # 3 rec, ~25 s
    python experiments/run_highd_extract.py --only 25 26     # named recordings

Writes out/highd_recordings.json (per-recording blocks), out/highd_targets.json
(the pooled targets - the single source of truth every figure, table and paper
reads via emv/empirical.py), out/highd_pools.npz (value pools, so percentiles and
bootstraps never re-read the CSVs) and out/highd/rec_NN.json (per-recording
cache; makes re-running one recording a 4 s operation).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv import highd as hd
from emv.runlog import RunLogger

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
CACHE = os.path.join(OUT, "highd")

#: Regime bins over (carriageway, 60 s window). `dense` deliberately brackets
#: the 1700 veh/h/lane of emv.scenarios.make_sumo_like so the model's SUMO
#: mirror has a real-world density twin.
REGIMES = {
    "freeflow":  dict(med_speed=(28.0, 99.0), flow=(0.0, 1200.0)),
    "dense":     dict(med_speed=(18.0, 30.0), flow=(1300.0, 2200.0)),
    "congested": dict(med_speed=(0.0, 15.0),  flow=(0.0, 3000.0)),
}

#: Leave-locations-out split. Locations differ in geometry, speed limit and lane
#: count, so holding whole locations back is a stricter test than holding back
#: recordings: 37 of 60 recordings share location 1 and are not independent.
TRAIN_LOCATIONS = (1, 2, 5)
HOLDOUT_LOCATIONS = (3, 4, 6)

#: Observables carried into out/highd_targets.json. `model_key` is the matching
#: emv.metrics.behaviour_stats key (None = no model counterpart yet).
TARGET_KEYS = (
    "lc_dur_med", "lc_dur_mean", "lc_dur_sd", "lc_dur_p10", "lc_dur_p90",
    "lc_dur_complete_med", "lc_peak_vy_med", "lc_peak_vy_p90",
    "lc_dur_full_med", "lc_dur_full_mean", "lc_dur_full_sd",
    "lc_dur_full_p10", "lc_dur_full_p90", "lc_c2c_med", "lc_c2c_p90",
    "lc_dy_full_med", "lc_peak_vy_full_med", "lc_peak_vy_full_p90",
    "lc_truncated_share", "flow_veh_h_lane", "density_veh_km_lane",
    "lat_speed_med", "lat_speed_p90",
    "peak_alat_med", "peak_alat_p90", "peak_alat_p99", "alat_p99",
    "alat_dataset_p99",
    "peak_decel_med", "peak_decel_p90", "peak_decel_p99",
    "lane_offset_mean", "lane_offset_p90",
    "bg_med_speed", "bg_p15_speed", "accel_p99", "decel_p01",
    "thw_med", "thw_p15", "thw_p85", "ttc_p01", "ttc_p05",
    "lc_per_veh_km", "lc_per_veh_km_complete",
    "truck_share", "obs_dur_med", "lane_width_med",
)

#: Pools kept for distributional distances and bootstraps (per-track scalars and
#: per-frame value pools only - never a per-frame matrix).
POOL_KEYS = ("trk_peak_decel", "trk_peak_alat", "trk_peak_vy", "trk_mean_vx",
             "trk_offset_abs", "trk_n_frames", "trk_class",
             "vy", "alat", "alat_dataset", "offset", "accel", "thw", "ttc",
             "lc_dur", "lc_peak_vy", "lc_dur_full", "lc_c2c", "lc_peak_vy_full")

#: Cap on samples kept per pool, taken on a uniform quantile grid so the
#: distribution is preserved while the .npz stays a few MB.
POOL_CAP = 200_000


def _subsample(a, cap=POOL_CAP, seed=11):
    a = np.asarray(a)
    if a.size <= cap:
        return a
    q = np.linspace(0, 1, cap)
    return np.quantile(np.sort(a.astype(np.float64)), q).astype(a.dtype)


def regime_of(w) -> str | None:
    for name, b in REGIMES.items():
        lo, hi = b["med_speed"]
        flo, fhi = b["flow"]
        if lo <= w["med_speed"] < hi and flo <= w["flow_veh_h_lane"] < fhi:
            return name
    return None


def summarise(values, key):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                   dtype=float)
    if a.size == 0:
        return None
    mean, lo, hi = hd.bootstrap_ci(a)
    return dict(value=round(float(np.median(a)), 4), mean=round(mean, 4),
                ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                sd=round(float(a.std()), 4), n=int(a.size),
                spread_lo=round(float(np.percentile(a, 10)), 4),
                spread_hi=round(float(np.percentile(a, 90)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=hd.HIGHD_ZIP)
    ap.add_argument("--only", nargs="*", default=None,
                    help="'meta' for the metadata pass only, or recording ids")
    ap.add_argument("--quick", action="store_true", help="3 recordings")
    ap.add_argument("--dt", type=float, default=hd.REC_DT)
    ap.add_argument("--win", type=float, default=60.0)
    ap.add_argument("--persist", type=float, default=0.5)
    ap.add_argument("--settle", type=float, default=0.6)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--n-lanes", type=int, default=3,
                    help="lane count both carriageways must have for the "
                         "geometry-matched target subset (0 = no filter)")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    ap.add_argument("--verify-zip", action="store_true",
                    help="also compute the full zip sha256 (~8 s)")
    args = ap.parse_args()

    if not os.path.exists(args.zip):
        raise SystemExit(f"highD zip not found: {args.zip}\n"
                         "pass --zip PATH or set $EMV_HIGHD_ZIP")
    os.makedirs(CACHE, exist_ok=True)
    fp = hd.zip_fingerprint(args.zip, full_sha=args.verify_zip)
    config = dict(dt=args.dt, win=args.win, persist=args.persist,
                  settle=args.settle, min_frames=args.min_frames,
                  n_lanes_filter=args.n_lanes, pool_cap=POOL_CAP,
                  regimes={k: {kk: list(vv) for kk, vv in v.items()}
                           for k, v in REGIMES.items()})

    meta_only = bool(args.only) and args.only[0] == "meta"
    rids = hd.recording_ids(args.zip)
    if args.quick:
        rids = rids[:3]
    elif args.only and not meta_only:
        rids = [int(r) for r in args.only]

    with RunLogger("highd_extract",
                   note=f"pass={'meta' if meta_only else 'meta+traj'} "
                        f"n_rec={len(rids)} dt={args.dt}",
                   params=config) as rl:
        t0 = time.time()
        print(f"[meta] {len(rids)} recordings ...", flush=True)
        ms = hd.meta_stats(args.zip, rids)
        recs = hd.recordings(args.zip, rids)
        by_rid = {r.rid: r for r in recs}
        print(f"[meta] {ms['n_vehicles']} vehicles, {ms['veh_km']} veh-km, "
              f"{ms['n_lane_changes']} lane changes -> "
              f"{ms['lc_per_veh_km']} LC/veh-km  ({time.time()-t0:.1f} s)",
              flush=True)

        provenance = dict(
            extractor_version=hd.EXTRACTOR_VERSION,
            generated=time.strftime("%Y-%m-%dT%H:%M:%S"), **fp,
            recordings=[int(r) for r in rids],
            numpy=np.__version__, python=sys.version.split()[0],
            platform=sys.platform)

        if meta_only:
            path = os.path.join(OUT, "highd_meta.json")
            json.dump(dict(provenance=provenance, config=config, meta=ms),
                      open(path, "w"), indent=1)
            print(f"[meta] wrote {path}")
            rl.finish(metrics=dict(lc_per_veh_km=ms["lc_per_veh_km"],
                                   n_recordings=ms["n_recordings"],
                                   veh_km=ms["veh_km"],
                                   n_lane_changes=ms["n_lane_changes"]),
                      outputs=["out/highd_meta.json"])
            return

        # ---------------- trajectory pass -------------------------------
        blocks, all_windows, all_events, pool_list = {}, [], [], []
        for i, rid in enumerate(rids, 1):
            cpath = os.path.join(CACHE, f"rec_{rid:02d}.json")
            cached = None
            if os.path.exists(cpath) and not args.force:
                try:
                    cached = json.load(open(cpath))
                except Exception:
                    cached = None
                if cached and not hd.cache_is_valid(cached, config, fp):
                    cached = None
            t1 = time.time()
            if cached:
                st = cached["stats"]
                pools = {k: np.load(os.path.join(CACHE, f"rec_{rid:02d}.npz"))[k]
                         for k in POOL_KEYS}
                pool_list.append(pools)
                print(f"[{i:2d}/{len(rids)}] rec {rid:02d} cached", flush=True)
            else:
                rs = hd.recording_stats(args.zip, rid, dt=args.dt,
                                        persist=args.persist,
                                        settle=args.settle, win=args.win,
                                        min_frames=args.min_frames)
                pooled = hd.pool_pools(list(rs["pools"].values()))
                pool_list.append({k: pooled[k] for k in POOL_KEYS if k in pooled})
                np.savez_compressed(
                    os.path.join(CACHE, f"rec_{rid:02d}.npz"),
                    **{k: pooled[k] for k in POOL_KEYS if k in pooled})
                st = dict(meta=rs["meta"], observables=rs["observables"],
                          lane_profile=rs["lane_profile"], windows=rs["windows"],
                          n_events=len(rs["events"]),
                          n_events_complete=int(sum(1 for e in rs["events"]
                                                    if e["complete"])))
                json.dump(dict(provenance=provenance, config=config, stats=st),
                          open(cpath, "w"), indent=1)
                all_events.extend(
                    [dict(rid=e["rid"], duration=e["duration"],
                          dur_c2c=e["dur_c2c"], peak_vy=e["peak_vy"],
                          dy=e["dy"], v_mean=e["v_mean"], cls=e["cls"],
                          complete=e["complete"], from_lane=e["from_lane"],
                          to_lane=e["to_lane"]) for e in rs["events"]])
                print(f"[{i:2d}/{len(rids)}] rec {rid:02d} "
                      f"loc{st['meta']['location_id']} "
                      f"{sum(o['n_tracks'] for o in st['observables'].values())} tracks, "
                      f"{st['n_events']} LC  ({time.time()-t1:.1f} s)", flush=True)
            blocks[f"{rid:02d}"] = st
            all_windows.extend(st["windows"])

        # ---------------- regime binning + targets ----------------------
        for w in all_windows:
            w["regime"] = regime_of(w)
        n_lanes_ok = set(hd.select_recordings(recs, n_lanes=args.n_lanes or None))
        split = {}
        for rid in rids:
            loc = by_rid[rid].location_id
            split[f"{rid:02d}"] = ("train" if loc in TRAIN_LOCATIONS
                                   else "holdout" if loc in HOLDOUT_LOCATIONS
                                   else "unused")

        # Pool over *carriageways*, labelled by their own regime. The bootstrap
        # unit stays the recording (both carriageways of one recording share the
        # drone flight, weather and time of day, so they are not independent).
        targets = {}
        for scope in ("all", "train", "holdout"):
            for regime in (None, "freeflow", "dense", "congested"):
                cells, recs_used = [], set()
                for rid in rids:
                    key = f"{rid:02d}"
                    if scope != "all" and split[key] != scope:
                        continue
                    if args.n_lanes and rid not in n_lanes_ok:
                        continue
                    for d, o in blocks[key]["observables"].items():
                        if regime is not None and o.get("regime") != regime:
                            continue
                        cells.append(o)
                        recs_used.add(key)
                if not cells:
                    continue
                name = scope if regime is None else f"{scope}:{regime}"
                block = {}
                for k in TARGET_KEYS:
                    s = summarise([o[k] for o in cells if k in o], k)
                    if s is not None:
                        block[k] = s
                block["_recordings"] = sorted(recs_used)
                block["_n_recordings"] = len(recs_used)
                block["_n_carriageways"] = len(cells)
                targets[name] = block

        pooled = hd.pool_pools(pool_list)
        npz = {k: _subsample(pooled[k]) for k in POOL_KEYS if k in pooled}
        npz_path = os.path.join(OUT, "highd_pools.npz")
        np.savez_compressed(npz_path, **npz)

        rec_path = os.path.join(OUT, "highd_recordings.json")
        json.dump(dict(provenance=provenance, config=config, meta=ms,
                       split=split, recordings=blocks, windows=all_windows),
                  open(rec_path, "w"), indent=1)

        tgt_path = os.path.join(OUT, "highd_targets.json")
        json.dump(dict(provenance=dict(provenance,
                                       subset=f"n_lanes=={args.n_lanes}"
                                              if args.n_lanes else "all",
                                       train_locations=list(TRAIN_LOCATIONS),
                                       holdout_locations=list(HOLDOUT_LOCATIONS),
                                       wall_s=round(time.time() - t0, 1)),
                       config=config, dataset=ms, split=split,
                       targets=targets),
                  open(tgt_path, "w"), indent=1)

        ncw = {}
        for key in blocks:
            for o in blocks[key]["observables"].values():
                ncw[o.get("regime", "other")] = ncw.get(o.get("regime", "other"), 0) + 1
        print(f"\ncarriageways per regime: {ncw}")
        hdr = ("regime", "nrec", "ncw", "lc_dur_full", "c2c", "pk_vy",
               "alat_p99", "v_bg", "dens", "lc/vkm")
        print("{:<16s}{:>5s}{:>5s}{:>12s}{:>7s}{:>7s}{:>9s}{:>7s}{:>7s}{:>8s}".format(*hdr))
        for name in ("all", "all:freeflow", "all:dense", "all:congested",
                     "train:freeflow", "holdout:freeflow"):
            if name not in targets:
                continue
            t = targets[name]
            g = lambda k: t.get(k, {}).get("value", float("nan"))
            print("{:<16s}{:>5d}{:>5d}{:>12.2f}{:>7.2f}{:>7.2f}{:>9.3f}"
                  "{:>7.1f}{:>7.1f}{:>8.3f}".format(
                      name, t["_n_recordings"], t["_n_carriageways"],
                      g("lc_dur_full_med"), g("lc_c2c_med"),
                      g("lc_peak_vy_full_med"), g("peak_alat_p99"),
                      g("bg_med_speed"), g("density_veh_km_lane"),
                      g("lc_per_veh_km")))
        print(f"\nwrote {rec_path}\n      {tgt_path}\n      {npz_path} "
              f"({os.path.getsize(npz_path)/1e6:.1f} MB)")
        rl.finish(metrics=dict(
            lc_per_veh_km=ms["lc_per_veh_km"],
            lc_dur_med=targets.get("all", {}).get("lc_dur_med", {}).get("value"),
            n_recordings=len(rids), veh_km=ms["veh_km"],
            n_lane_changes=ms["n_lane_changes"],
            lane_width_med=ms["lane_width_med"]),
            outputs=["out/highd_recordings.json", "out/highd_targets.json",
                     "out/highd_pools.npz"])


if __name__ == "__main__":
    main()
