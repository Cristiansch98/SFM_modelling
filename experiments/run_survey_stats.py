"""Multi-seed survey benchmark with seed-level statistical inference.

Extends the 3-seed survey benchmark (run_survey_compare.py) to the same
10-seed protocol as the companion paper and adds the statistics a reviewer
will ask for. Design:

  * unit of replication = the random seed (network/demand/EV identical, so
    conditions are PAIRED within seed);
  * primary endpoint  = POOLED behaviour-fingerprint distance to the
    rules-only relabelled ground truth (single-seed fingerprints are noisy,
    see PROJECT_LOG section 5) - pooling is over seeds, inference is at the
    seed (cluster) level;
  * uncertainty       = seed-cluster bootstrap (resample seeds with
    replacement, SAME resample for every condition -> paired CIs on levels
    and on differences);
  * hypothesis tests  = exact paired permutation tests (swap the two
    conditions' streams within a seed; 2^n_seeds enumerations, no
    distributional assumptions, no scipy needed), Holm-corrected within the
    primary family (force vs each alternative);
  * secondary endpoint = EV segment speed per seed: exact sign-flip
    permutation test on paired differences + Cohen's dz + bootstrap CI;
  * safety            = collision counts per seed (descriptive).

Pooled fingerprints are recomputed thousands of times (bootstrap /
permutation), so the stream of every (mode, seed) run is reduced once to
sufficient statistics (label counts, track outcomes, onset / |lateral
speed| / acceleration value pools); pooling = summing counts and
concatenating pools. fp_from_pools() is verified against gt.fingerprint()
on the actual pooled frames at runtime (hard assert), so the fast path
cannot silently diverge from the metric of record.

Each SUMO run is cached in out/survey_stats/runs/<mode>_<seed>.json, so
adding seeds later (or re-running the statistics) does not re-simulate.

    python experiments/run_survey_stats.py                 # 10 seeds 42-51
    python experiments/run_survey_stats.py --seeds 42 ...  # custom seeds
    python experiments/run_survey_stats.py --stats-only    # reuse cache

Writes out/survey_stats.json + out/fig_survey_stats.png.
Self-logs kind='survey_stats'.
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv.params import tuned_params
from emv.sumo.bridge import SumoBridge
from emv.sumo import make_scenario as sc
from emv import groundtruth as gt
from run_survey_compare import pooled_gt, ENTITY_LABEL

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
RUN_DIR = os.path.join(OUT, "survey_stats", "runs")

MODES = ("force", "bluelight", "rule", "none")
PRIMARY_PAIRS = (("force", "bluelight"), ("force", "rule"), ("force", "none"))
OTHER_PAIRS = (("bluelight", "rule"), ("bluelight", "none"), ("rule", "none"))


# ---------------------------------------------------------------------------
# simulation with per-run cache
# ---------------------------------------------------------------------------
def run_or_load(mode, seed, p, run_dir=RUN_DIR, sumo_dir=None,
                ev_fixed_lane=False):
    path = os.path.join(run_dir, f"{mode}_{seed}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    t0 = time.time()
    m = SumoBridge(mode, p=p, seed=seed,
                   out_dir=sumo_dir or os.path.join(OUT, "sumo"),
                   record_traj=False, ev_fixed_lane=ev_fixed_lane).run()
    labelled = gt.annotate_stream(m["ego_frames"], n_lanes=sc.N_LANES,
                                  lane_width=sc.LANE_W)
    for fr in labelled:                     # unique track ids across seeds
        for v in fr["vehicles"]:
            v["id"] = f"{seed}:{v['id']}"
    rec = dict(mode=mode, seed=seed,
               seg_speed=m.get("seg_speed"), seg_time=m.get("seg_time"),
               seg_complete=bool(m.get("seg_complete", False)),
               clearance_mean=m.get("clearance_mean"),
               collisions=int(m["collisions"]),
               wall_s=round(time.time() - t0, 1), frames=labelled)
    os.makedirs(run_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh)
    return rec


# ---------------------------------------------------------------------------
# sufficient statistics for fast pooled fingerprints
# ---------------------------------------------------------------------------
def stream_pools(frames):
    """Reduce one labelled stream to the sufficient statistics of
    gt.fingerprint(). Pooling streams = summing the counts and
    concatenating the value pools (track ids are seed-prefixed, so tracks
    never merge across seeds)."""
    R = gt.HeuristicAnnotator.PROXIMITY_THRESHOLD
    n_min = gt.HeuristicAnnotator.MIN_OBSERVED_FRAMES
    counts = dict.fromkeys(gt.LABELS, 0)
    tracks = {}
    lat_y, accels = [], []
    n_obs = 0
    for fr in frames:
        for v in fr["vehicles"]:
            if v.get("distance_to_ego", 999.0) > R:
                continue
            n_obs += 1
            counts[v["behaviour"]] += 1
            accels.append(v.get("acceleration", 0.0))
            if v["behaviour"] == "yielded":
                lat_y.append(abs(v.get("lateral_speed_ms", 0.0)))
            tr = tracks.setdefault(v["id"], dict(n=0, labels=set(),
                                                 onset=None))
            tr["n"] += 1
            tr["labels"].add(v["behaviour"])
            if v["behaviour"] == "yielded" and tr["onset"] is None:
                tr["onset"] = v["distance_to_ego"]
    ass = [tr for tr in tracks.values() if tr["n"] >= n_min]
    return dict(
        n_frames=len(frames), n_obs=n_obs, n_tracks=len(tracks),
        n_ass=len(ass),
        c_yield=counts["yielded"], c_fail=counts["failed_to_yield"],
        c_brake=counts["braked_abruptly"], c_normal=counts["normal"],
        trk_y=sum(1 for tr in ass if "yielded" in tr["labels"]),
        trk_f=sum(1 for tr in ass if "failed_to_yield" in tr["labels"]
                  and "yielded" not in tr["labels"]),
        trk_b=sum(1 for tr in ass if "braked_abruptly" in tr["labels"]),
        onsets=np.array([tr["onset"] for tr in tracks.values()
                         if tr["onset"] is not None], float),
        lat_y=np.array(lat_y, float), accels=np.array(accels, float))


def fp_from_pools(pools):
    """Pooled fingerprint (the gt.fingerprint() keys used by the distance)
    from a list of per-stream sufficient statistics."""
    n_frames = sum(p["n_frames"] for p in pools)
    n_obs = sum(p["n_obs"] for p in pools)
    n_ass = sum(p["n_ass"] for p in pools)
    onsets = np.concatenate([p["onsets"] for p in pools]) if pools else []
    lat_y = np.concatenate([p["lat_y"] for p in pools]) if pools else []
    accels = np.concatenate([p["accels"] for p in pools]) if pools else []

    def share(x, n):
        return float(x) / n if n else float("nan")

    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else float("nan")

    return dict(
        n_frames=n_frames, n_obs=n_obs,
        n_tracks=sum(p["n_tracks"] for p in pools),
        n_tracks_assessable=n_ass,
        veh_per_frame=share(n_obs, n_frames),
        obs_yielded=share(sum(p["c_yield"] for p in pools), n_obs),
        obs_failed=share(sum(p["c_fail"] for p in pools), n_obs),
        obs_braked=share(sum(p["c_brake"] for p in pools), n_obs),
        obs_normal=share(sum(p["c_normal"] for p in pools), n_obs),
        trk_yielded=share(sum(p["trk_y"] for p in pools), n_ass),
        trk_failed=share(sum(p["trk_f"] for p in pools), n_ass),
        trk_braked=share(sum(p["trk_b"] for p in pools), n_ass),
        onset_dist_med=pct(onsets, 50), onset_dist_p90=pct(onsets, 90),
        lat_speed_med=pct(lat_y, 50), lat_speed_p90=pct(lat_y, 90),
        accel_p05=pct(accels, 5),
    )


def self_check(pools_by_seed, frames_by_seed, seeds):
    """fp_from_pools must reproduce gt.fingerprint on the pooled frames."""
    pooled_frames = [fr for s in seeds for fr in frames_by_seed[s]]
    ref = gt.fingerprint(pooled_frames)
    fast = fp_from_pools([pools_by_seed[s] for s in seeds])
    for k, _, _ in gt._DIST_SPEC:
        a, b = fast[k], ref[k]
        if np.isnan(a) and np.isnan(b):
            continue
        assert abs(a - b) < 1e-9, f"self-check failed on {k}: {a} vs {b}"
    print("self-check: fp_from_pools == gt.fingerprint on pooled frames",
          flush=True)


# ---------------------------------------------------------------------------
# statistics (no scipy on this machine -> permutation + bootstrap)
# ---------------------------------------------------------------------------
def fid(pools, fp_ref):
    return gt.fingerprint_distance(fp_from_pools(pools), fp_ref)["total"]


def paired_permutation_fidelity(pools_a, pools_b, seeds, fp_refs,
                                max_exact=13, n_mc=20000, rng=None):
    """Exact paired permutation test on the pooled fidelity difference.

    H0: conditions exchangeable within seed. For every subset of seeds,
    swap the two conditions' streams and recompute the pooled-fidelity
    difference. Two-sided p per reference in fp_refs (dict name -> fp).
    """
    n = len(seeds)
    if n <= max_exact:
        masks = list(itertools.product((0, 1), repeat=n))
    else:
        masks = [tuple(rng.integers(0, 2, n)) for _ in range(n_mc)]
        masks[0] = (0,) * n                     # include the identity
    obs, null = {}, {name: [] for name in fp_refs}
    for mask in masks:
        pa = [pools_b[s] if m else pools_a[s] for s, m in zip(seeds, mask)]
        pb = [pools_a[s] if m else pools_b[s] for s, m in zip(seeds, mask)]
        fa, fb = fp_from_pools(pa), fp_from_pools(pb)
        for name, ref in fp_refs.items():
            d = (gt.fingerprint_distance(fa, ref)["total"]
                 - gt.fingerprint_distance(fb, ref)["total"])
            if mask == (0,) * n:
                obs[name] = d
            null[name].append(d)
    out = {}
    for name in fp_refs:
        arr = np.abs(np.array(null[name]))
        out[name] = dict(
            diff=round(float(obs[name]), 4),
            p=float(np.mean(arr >= abs(obs[name]) - 1e-12)),
            n_perm=len(masks), exact=n <= max_exact)
    return out


def sign_flip_test(d, max_exact=13, n_mc=20000, rng=None):
    """Exact paired sign-flip permutation test on mean(d)."""
    d = np.asarray(d, float)
    n = len(d)
    if n <= max_exact:
        signs = np.array(list(itertools.product((-1, 1), repeat=n)))
    else:
        signs = rng.choice((-1, 1), size=(n_mc, n))
        signs[0] = 1
    null = np.abs((signs * d).mean(axis=1))
    return float(np.mean(null >= abs(d.mean()) - 1e-12)), len(signs)


def holm(pvals):
    """Holm-Bonferroni step-down adjustment. pvals: dict name -> p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[name] = min(1.0, running)
    return adj


def boot_ci(a, lo=2.5, hi=97.5):
    a = np.asarray(a, float)
    return [round(float(np.percentile(a, lo)), 4),
            round(float(np.percentile(a, hi)), 4)]


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def plot_stats(res, seeds, path):
    import matplotlib
    matplotlib.use("Agg")
    from emv.viz import _fig, _style_ax, _save, LIGHT, CAT_LIGHT

    pal = LIGHT
    colors = dict(force=CAT_LIGHT["blue"], bluelight=CAT_LIGHT["yellow"],
                  rule=CAT_LIGHT["aqua"], none=pal["axis"])
    fig = _fig(11.5, 4.6, pal)
    ax1, ax2 = fig.subplots(1, 2, width_ratios=[1.15, 1])
    for a in (ax1, ax2):
        _style_ax(a, pal)

    ys = np.arange(len(MODES))[::-1]
    for y, m in zip(ys, MODES):
        r = res["fidelity"][m]
        ax1.plot(r["ci95"], [y, y], color=colors[m], lw=2.6,
                 solid_capstyle="round")
        ax1.plot([r["pooled"]], [y], "o", ms=9, color=colors[m],
                 mec=pal["surface"], mew=1.2, zorder=5)
        ax1.plot(r["per_seed"], np.full(len(r["per_seed"]), y + 0.22), "o",
                 ms=4, color=colors[m], alpha=0.45, mec="none")
        ax1.annotate(f"{r['pooled']:.3f}", (r["pooled"], y - 0.30),
                     ha="center", fontsize=8.5, color=pal["ink2"])
    ax1.set_yticks(ys, [ENTITY_LABEL[m] for m in MODES], fontsize=9)
    ax1.set_xlabel("fingerprint distance to real behaviour "
                   "(rules-only reference, lower = better)")
    ax1.set_title(f"Behavioural fidelity, {len(seeds)} seeds - pooled dot, "
                  "95 % cluster-bootstrap CI, faint dots = single seeds",
                  fontsize=10, loc="left")
    ax1.set_ylim(-0.7, len(MODES) - 0.25)

    for y, m in zip(ys, MODES):
        v = np.array(res["speed"][m]["per_seed_kmh"], float)
        v = v[np.isfinite(v)]
        ci = res["speed"][m]["ci95"]
        ax2.plot(ci, [y, y], color=colors[m], lw=2.6, solid_capstyle="round")
        ax2.plot([v.mean()], [y], "o", ms=9, color=colors[m],
                 mec=pal["surface"], mew=1.2, zorder=5)
        ax2.plot(v, np.full(len(v), y + 0.22), "o", ms=4, color=colors[m],
                 alpha=0.45, mec="none")
        ax2.annotate(f"{v.mean():.1f}", (v.mean(), y - 0.30), ha="center",
                     fontsize=8.5, color=pal["ink2"])
    ax2.set_yticks(ys, ["" for _ in MODES])
    ax2.set_xlabel("EV segment speed (km/h)")
    ax2.set_title("EV progress, same runs - mean, 95 % bootstrap CI",
                  fontsize=10, loc="left")
    ax2.set_ylim(-0.7, len(MODES) - 0.25)

    lines = ["paired permutation (Holm-adj.):"]
    for a, b in PRIMARY_PAIRS:
        p = res["fidelity_tests"][f"{a}_vs_{b}"]["rules"]["p_holm"]
        lines.append(f"  {a} vs {b}: p = {p:.3f}")
    ax1.annotate("\n".join(lines), (0.02, 0.03), xycoords="axes fraction",
                 fontsize=8, color=pal["ink2"], va="bottom",
                 bbox=dict(boxstyle="round,pad=0.35", fc=pal["surface"],
                           ec=pal["grid"]))
    fig.suptitle("Survey benchmark with seed-level statistics",
                 fontsize=12, color=pal["ink"], x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(42, 52)))
    ap.add_argument("--gt-dir", default=gt.PIPELINE_OUT)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--stats-only", action="store_true",
                    help="fail instead of simulating when a run is not cached")
    ap.add_argument("--ev-fixed-lane", action="store_true",
                    help="ablation: pin the EV to its departure lane; all "
                         "corridor-making comes from surrounding traffic")
    ap.add_argument("--tag", default=None,
                    help="suffix for cache dir and outputs (default: "
                         "'evfix' when --ev-fixed-lane, else none)")
    args = ap.parse_args()
    seeds = args.seeds
    tag = args.tag if args.tag is not None else (
        "evfix" if args.ev_fixed_lane else "")
    sfx = f"_{tag}" if tag else ""
    run_dir = os.path.join(OUT, "survey_stats", f"runs{sfx}")
    sumo_dir = os.path.join(OUT, f"sumo{sfx}")

    from emv.runlog import RunLogger
    with RunLogger("survey_stats",
                   note=f"{len(seeds)}-seed survey benchmark + cluster "
                        "bootstrap CIs + exact paired permutation tests"
                        + (f" [{tag}: EV pinned to its lane]" if
                           args.ev_fixed_lane else ""),
                   params=dict(seeds=",".join(map(str, seeds)),
                               n_boot=args.n_boot, tag=tag,
                               ev_fixed_lane=args.ev_fixed_lane)) as rl:
        # ---- ground truth ------------------------------------------------
        gt_frames, gt_names = pooled_gt(args.gt_dir)
        if not gt_frames:
            raise SystemExit(f"no ground truth in {args.gt_dir}")
        fp_refs = dict(rules=gt.fingerprint(gt.reannotate(gt_frames)),
                       reviewed=gt.fingerprint(gt_frames))
        print(f"ground truth: {len(gt_names)} video(s), "
              f"{fp_refs['reviewed']['n_obs']} obs, "
              f"{fp_refs['reviewed']['n_tracks']} tracks", flush=True)

        # ---- simulations (cached) ----------------------------------------
        p = tuned_params()
        runs = {}                                # (mode, seed) -> record
        for mode in MODES:
            for seed in seeds:
                path = os.path.join(run_dir, f"{mode}_{seed}.json")
                if args.stats_only and not os.path.exists(path):
                    raise SystemExit(f"--stats-only but {path} missing")
                r = run_or_load(mode, seed, p, run_dir=run_dir,
                                sumo_dir=sumo_dir,
                                ev_fixed_lane=args.ev_fixed_lane)
                runs[(mode, seed)] = r
                print(f"  {mode:10s} seed {seed}: "
                      f"{(r['seg_speed'] or 0) * 3.6:6.1f} km/h  "
                      f"coll {r['collisions']}  "
                      f"[{r.get('wall_s', 0):.0f}s]", flush=True)

        # ---- sufficient statistics + self-check --------------------------
        pools = {m: {s: stream_pools(runs[(m, s)]["frames"]) for s in seeds}
                 for m in MODES}
        self_check(pools["force"],
                   {s: runs[("force", s)]["frames"] for s in seeds}, seeds)

        res = dict(seeds=seeds, n_boot=args.n_boot, tag=tag,
                   ev_fixed_lane=args.ev_fixed_lane,
                   gt=dict(videos=gt_names,
                           n_obs=fp_refs["reviewed"]["n_obs"],
                           n_tracks=fp_refs["reviewed"]["n_tracks"]))

        # ---- fidelity: pooled, per seed, bootstrap CI ---------------------
        rng = np.random.default_rng(7)
        n = len(seeds)
        draws = rng.integers(0, n, size=(args.n_boot, n))
        boot = {m: {"rules": [], "reviewed": []} for m in MODES}
        t0 = time.time()
        for row in draws:
            sel = [seeds[i] for i in row]
            for m in MODES:
                fp = fp_from_pools([pools[m][s] for s in sel])
                for name, ref in fp_refs.items():
                    boot[m][name].append(
                        gt.fingerprint_distance(fp, ref)["total"])
        print(f"bootstrap: {args.n_boot} draws x {len(MODES)} conditions "
              f"[{time.time() - t0:.0f}s]", flush=True)

        res["fidelity"] = {}
        for m in MODES:
            fp_m = fp_from_pools([pools[m][s] for s in seeds])
            res["fidelity"][m] = dict(
                pooled=round(gt.fingerprint_distance(
                    fp_m, fp_refs["rules"])["total"], 4),
                pooled_reviewed=round(gt.fingerprint_distance(
                    fp_m, fp_refs["reviewed"])["total"], 4),
                ci95=boot_ci(boot[m]["rules"]),
                ci95_reviewed=boot_ci(boot[m]["reviewed"]),
                per_seed=[round(fid([pools[m][s]], fp_refs["rules"]), 4)
                          for s in seeds],
                fingerprint=fp_m)
        b_force = np.array(boot["force"]["rules"])
        best = np.logical_and.reduce(
            [b_force <= np.array(boot[m]["rules"]) + 1e-12
             for m in MODES if m != "force"])
        res["p_force_best"] = round(float(best.mean()), 4)
        res["diff_ci"] = {}
        for a, b in PRIMARY_PAIRS + OTHER_PAIRS:
            d = np.array(boot[a]["rules"]) - np.array(boot[b]["rules"])
            res["diff_ci"][f"{a}_minus_{b}"] = boot_ci(d)

        # ---- fidelity: exact paired permutation tests ---------------------
        res["fidelity_tests"] = {}
        for a, b in PRIMARY_PAIRS + OTHER_PAIRS:
            t0 = time.time()
            res["fidelity_tests"][f"{a}_vs_{b}"] = paired_permutation_fidelity(
                pools[a], pools[b], seeds, fp_refs, rng=rng)
            print(f"permutation {a} vs {b} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        for name in ("rules", "reviewed"):
            fam = {f"{a}_vs_{b}": res["fidelity_tests"][f"{a}_vs_{b}"]
                   [name]["p"] for a, b in PRIMARY_PAIRS}
            adj = holm(fam)
            for k, v in adj.items():
                res["fidelity_tests"][k][name]["p_holm"] = round(v, 4)

        # ---- EV speed: paired tests ---------------------------------------
        res["speed"] = {}
        v = {m: np.array([(runs[(m, s)]["seg_speed"] or np.nan) * 3.6
                          for s in seeds]) for m in MODES}
        for m in MODES:
            vv = v[m][np.isfinite(v[m])]
            bs = rng.choice(vv, size=(args.n_boot, len(vv))).mean(axis=1)
            res["speed"][m] = dict(
                per_seed_kmh=[None if not np.isfinite(x) else round(x, 2)
                              for x in v[m]],
                mean=round(float(vv.mean()), 2),
                sd=round(float(vv.std(ddof=1)), 2),
                ci95=boot_ci(bs),
                n_seg_complete=sum(runs[(m, s)]["seg_complete"]
                                   for s in seeds))
        res["speed_tests"] = {}
        for a, b in PRIMARY_PAIRS + OTHER_PAIRS:
            keep = np.isfinite(v[a]) & np.isfinite(v[b])
            d = (v[a] - v[b])[keep]
            p_perm, n_perm = sign_flip_test(d, rng=rng)
            dz = float(d.mean() / d.std(ddof=1)) if len(d) > 1 else np.nan
            bs = rng.choice(d, size=(args.n_boot, len(d))).mean(axis=1)
            res["speed_tests"][f"{a}_vs_{b}"] = dict(
                n=int(keep.sum()), mean_diff_kmh=round(float(d.mean()), 2),
                ci95=boot_ci(bs), cohen_dz=round(dz, 2),
                p=p_perm, n_perm=n_perm)
        fam = {f"{a}_vs_{b}": res["speed_tests"][f"{a}_vs_{b}"]["p"]
               for a, b in PRIMARY_PAIRS}
        for k, val in holm(fam).items():
            res["speed_tests"][k]["p_holm"] = round(val, 4)

        # ---- collisions ----------------------------------------------------
        res["collisions"] = {
            m: dict(total=int(sum(runs[(m, s)]["collisions"] for s in seeds)),
                    per_seed={str(s): int(runs[(m, s)]["collisions"])
                              for s in seeds},
                    seeds_with_any=int(sum(runs[(m, s)]["collisions"] > 0
                                           for s in seeds)))
            for m in MODES}

        # ---- rank stability -------------------------------------------------
        per_seed_rank = []
        for i, s in enumerate(seeds):
            ds = {m: res["fidelity"][m]["per_seed"][i] for m in MODES}
            per_seed_rank.append(min(ds, key=ds.get))
        res["per_seed_best"] = {m: per_seed_rank.count(m) for m in MODES}

        # ---- write + print ---------------------------------------------------
        with open(os.path.join(OUT, f"survey_stats{sfx}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        plot_stats(res, seeds,
                   os.path.join(OUT, f"fig_survey_stats{sfx}.png"))

        print("\n=== fidelity vs rules-only GT (pooled, "
              f"{n} seeds; 95% cluster-bootstrap CI) ===", flush=True)
        for m in MODES:
            r = res["fidelity"][m]
            print(f"  {m:10s} {r['pooled']:.3f}  CI [{r['ci95'][0]:.3f}, "
                  f"{r['ci95'][1]:.3f}]  (reviewed: {r['pooled_reviewed']:.3f})",
                  flush=True)
        print("=== paired permutation tests (rules ref; Holm within "
              "primary family) ===", flush=True)
        for a, b in PRIMARY_PAIRS + OTHER_PAIRS:
            t = res["fidelity_tests"][f"{a}_vs_{b}"]["rules"]
            ph = t.get("p_holm")
            print(f"  {a:10s} vs {b:10s} diff {t['diff']:+.3f}  "
                  f"p {t['p']:.4f}" + (f"  p_holm {ph:.4f}" if ph else ""),
                  flush=True)
        print("=== EV speed (km/h) ===", flush=True)
        for m in MODES:
            r = res["speed"][m]
            print(f"  {m:10s} {r['mean']:6.1f} +- {r['sd']:.1f}  "
                  f"CI [{r['ci95'][0]:.1f}, {r['ci95'][1]:.1f}]", flush=True)
        for a, b in PRIMARY_PAIRS:
            t = res["speed_tests"][f"{a}_vs_{b}"]
            print(f"  {a} vs {b}: diff {t['mean_diff_kmh']:+.1f} km/h  "
                  f"dz {t['cohen_dz']:+.2f}  p {t['p']:.4f}  "
                  f"p_holm {t['p_holm']:.4f}", flush=True)

        rl.finish(
            metrics=dict(
                seeds=len(seeds),
                **{f"{m}_fid": res["fidelity"][m]["pooled"] for m in MODES},
                **{f"{m}_kmh": res["speed"][m]["mean"] for m in MODES},
                p_fb=res["fidelity_tests"]["force_vs_bluelight"]["rules"]["p"],
                p_fr=res["fidelity_tests"]["force_vs_rule"]["rules"]["p"],
                p_fn=res["fidelity_tests"]["force_vs_none"]["rules"]["p"]),
            outputs=[f"out/survey_stats{sfx}.json",
                     f"out/fig_survey_stats{sfx}.png"])
