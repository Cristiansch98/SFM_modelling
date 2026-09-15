"""Per-term influence analysis for the two-stage framework (three instruments).

    python experiments/run_term_influence.py --only sensitivity   # ~40 min
    python experiments/run_term_influence.py --only budget ablation
    python experiments/run_term_influence.py --quick              # ~3 min

Instruments (see emv/terms.py for why all three are needed):

  sensitivity  Latin-hypercube over the whole 19-parameter normal vector, with
               COMMON RANDOM NUMBERS (the same seeds for every sample, so
               differences are parameter effects and not seed noise), analysed
               with Spearman rho (monotone, robust to saturation) and
               standardized regression coefficients with the linear surrogate's
               R2 reported alongside - SRC is only interpretable where R2 is
               high, and half of these responses are threshold-shaped.
               Also the identifiability check: each parameter should own a
               distinct primary response, or the block-wise fit is ill-posed.
  budget       Per-term share of the accelerations actually integrated, in the
               normal scenario and the emergency-vehicle one, split by driver
               awareness. Uses the accounting identity asserted in
               tests/test_terms.py, so the shares are exact, not estimated.
  ablation     Switch one term off, re-score the whole objective, report the
               change and which observables moved. The instrument that says what
               a term is *for*; the other two explain its answers.

Reuses `run_param_stats.spearman/src` rather than reimplementing them, so the
statistics are the same ones the 2026-07-23 study reported.

Writes out/term_influence.json (analysis) + out/term_influence.npz (raw design
matrices, so figures can be rebuilt without re-running).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv import highd_fit as hf
from emv import terms
from emv.params import Params, highd_params, normal_params
from emv.runlog import RunLogger
from emv.scenarios import highd_regime, make_highd_like

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_param_stats import spearman, src            # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")

#: Observables the sensitivity pass tracks. Every weighted key of the normal
#: objective, plus the objective itself and its two halves.
RESPONSES = tuple(k for k, _, w in hf.OBS_SPEC_NORMAL if w > 0) + (
    "loss", "moment", "shape")


# ---------------------------------------------------------------------------
#: Hand-set starting point for the discretionary term, used before the stage-1
#: fit exists. NOT a calibration result. It has to be on: with A_pass below the
#: pinning threshold no lane change ever happens, every lc_* observable is NaN,
#: and the analysis would be run at a degenerate point where the whole lateral
#: half of the objective is a flat penalty rather than a response surface.
LC_START = dict(A_pass=4.0, A_keep_right=2.0)


def base_params(preset: str) -> Params:
    """Starting point. `highd` = the 2026-07-29 fit plus LC_START; `normal` = the
    stage-1 fit once it has been run and pasted into params.TUNED_NORMAL."""
    if preset == "normal":
        try:
            return normal_params().copy(dt=0.06)
        except RuntimeError as exc:
            print(f"  (falling back to 'highd': {exc})")
    return highd_params().copy(dt=0.06, **LC_START)


def _merge(pool_list):
    m = {}
    for k in pool_list[0]:
        v = [q[k] for q in pool_list]
        m[k] = float(np.sum(v)) if np.isscalar(v[0]) else np.concatenate(v)
    return m


# ---------------------------------------------------------- 1. sensitivity
def run_sensitivity(ctx, base, seeds, n_lhs, regimes):
    names = [s[0] for s in hf.SPEC_NORMAL]
    lo = np.array([s[1] for s in hf.SPEC_NORMAL], float)
    hi = np.array([s[2] for s in hf.SPEC_NORMAL], float)
    rng = np.random.default_rng(2026)
    dims = lo.size
    u = (rng.permuted(np.tile(np.arange(n_lhs), (dims, 1)), axis=1).T
         + rng.random((n_lhs, dims))) / n_lhs
    X = lo + u * (hi - lo)

    Y = np.full((n_lhs, len(RESPONSES)), np.nan)
    t0 = time.time()
    for i, th in enumerate(X):
        L, parts, per = hf.loss_normal(th, ctx, seeds=seeds, base=base,
                                       regimes=regimes, return_detail=True)
        # responses are averaged over regimes, as the objective is
        obs = {}
        for k in RESPONSES:
            if k in ("loss", "moment", "shape"):
                continue
            vals = [per[r]["observables"].get(k, np.nan) for r in regimes]
            obs[k] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
        obs["loss"] = L
        obs["moment"] = float(np.mean([per[r]["moment"] for r in regimes]))
        obs["shape"] = float(np.mean([per[r]["shape"] for r in regimes]))
        Y[i] = [obs[k] for k in RESPONSES]
        if (i + 1) % 8 == 0 or i == n_lhs - 1:
            print(f"  LHS {i+1:3d}/{n_lhs}  best loss {np.nanmin(Y[:i+1, -3]):.4f}"
                  f"  ({time.time()-t0:.0f} s)", flush=True)

    rho = np.full((dims, len(RESPONSES)), np.nan)
    beta = np.full((dims, len(RESPONSES)), np.nan)
    r2 = np.full(len(RESPONSES), np.nan)
    for c in range(len(RESPONSES)):
        for j in range(dims):
            rho[j, c] = spearman(X[:, j], Y[:, c])
        beta[:, c], r2[c] = src(X, Y[:, c])

    #: |rho| above which a rank correlation is significant at ~5 % for this n
    crit = 1.96 / np.sqrt(max(n_lhs - 3, 1))
    primary = {}
    for j, n in enumerate(names):
        col = np.abs(rho[j, :len(RESPONSES) - 3])
        if np.all(np.isnan(col)):
            primary[n] = None
            continue
        c = int(np.nanargmax(col))
        primary[n] = dict(response=RESPONSES[c], rho=round(float(rho[j, c]), 3),
                          significant=bool(abs(rho[j, c]) > crit))
    return dict(names=names, responses=list(RESPONSES), X=X, Y=Y, rho=rho,
                beta=beta, r2=r2, crit=float(crit), n_lhs=n_lhs,
                primary=primary, seeds=list(seeds), regimes=list(regimes))


# --------------------------------------------------------------- 2. budget
def run_budget(base, seeds, regimes):
    """Per-term force budget, EV absent vs EV present."""
    out = {}
    for regime in regimes:
        spec = highd_regime(regime)
        for scen, kw in (("normal", dict(ev_inert=True)),
                         ("bluelight", dict(ev_inert=False))):
            reps = []
            for s in seeds:
                sim = make_highd_like(seed=s, p=base, regime=regime, spec=spec,
                                      road_len=3000.0, **kw)
                _, rep = terms.budget_run(sim, 110.0, rec_dt=0.12)
                reps.append(rep)
            merged = {}
            for key in reps[0]:
                if isinstance(reps[0][key], dict):
                    merged[key] = {t: round(float(np.mean([r[key][t] for r in reps])), 5)
                                   for t in reps[0][key]}
                else:
                    merged[key] = float(np.mean([r[key] for r in reps]))
            out[f"{regime}:{scen}"] = merged
            top = sorted(merged["share_y"].items(), key=lambda kv: -kv[1])[:3]
            print(f"  {regime:9s} {scen:9s} lateral budget: "
                  + ", ".join(f"{t} {v*100:.0f}%" for t, v in top), flush=True)
    return out


# ------------------------------------------------------------- 3. ablation
def run_ablation(ctx, base, seeds, regimes, which):
    scales = {k: s for k, s, _ in hf.OBS_SPEC_NORMAL}
    keys = [k for k, _, w in hf.OBS_SPEC_NORMAL if w > 0]
    names = [s[0] for s in hf.SPEC_NORMAL]
    th = np.array([getattr(base, n) for n in names], float)

    # clip=False here too: the baseline must be evaluated at exactly the same
    # point the ablations are perturbed from. With clipping on, a base carrying
    # A_pass=0 was silently raised to its search lower bound of 0.5, which is
    # *worse* than 0 (a barely-super-threshold incentive produces 20-40 s
    # manoeuvres), so every ablation appeared to improve the loss by ~7.
    L0, _, per0 = hf.loss_normal(th, ctx, seeds=seeds, base=base,
                                 regimes=regimes, return_detail=True, clip=False)
    obs0 = {r: per0[r]["observables"] for r in regimes}
    print(f"  baseline loss {L0:.4f}", flush=True)

    res = {"_baseline": dict(loss=round(float(L0), 4),
                             per_regime={r: per0[r]["total"] for r in regimes},
                             theta=dict(zip(names, np.round(th, 4).tolist())))}
    for name in which:
        over, term, stage, why = terms.ABLATIONS[name]
        p_ab = base.copy(**over)
        th_ab = np.array([getattr(p_ab, n) for n in names], float)
        t0 = time.time()
        # clip=False: an ablation sets an amplitude to exactly 0, and several
        # search bounds start above 0 (see make_params_spec)
        L, _, per = hf.loss_normal(th_ab, ctx, seeds=seeds, base=p_ab,
                                   regimes=regimes, return_detail=True,
                                   clip=False)
        att = {r: terms.attribution(obs0[r], per[r]["observables"], keys, scales)
               for r in regimes}
        worst = sorted(
            ((k, v) for k, v in att[regimes[0]].items() if v is not None),
            key=lambda kv: -abs(kv[1]))[:3]
        res[name] = dict(term=term, stage=stage, why=why, overrides=over,
                         loss=round(float(L), 4),
                         d_loss=round(float(L - L0), 4),
                         per_regime={r: per[r]["total"] for r in regimes},
                         collisions={r: per[r]["collisions"] for r in regimes},
                         attribution=att)
        print(f"  {name:20s} loss {L:7.3f}  d {L-L0:+7.3f}   "
              f"moves: {', '.join(f'{k} {v:+.1f}' for k, v in worst)}"
              f"   ({time.time()-t0:.0f} s)", flush=True)
    return res


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*",
                    default=["sensitivity", "budget", "ablation"],
                    choices=["sensitivity", "budget", "ablation"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[11, 12, 13])
    ap.add_argument("--n-lhs", type=int, default=96)
    ap.add_argument("--regimes", nargs="*", default=list(hf.FIT_REGIMES))
    ap.add_argument("--preset", default="highd", choices=["highd", "normal"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="out/term_influence.json")
    args = ap.parse_args()
    if args.quick:
        args.n_lhs, args.seeds = 8, [11]

    ctx = hf.normal_context(tuple(args.regimes))
    base = base_params(args.preset)
    regimes = tuple(args.regimes)
    print(f"preset {args.preset}  seeds {args.seeds}  regimes {list(regimes)}")
    for r in regimes:
        print(f"  {r:9s} {ctx[r]['n_cw']} carriageways, "
              f"density {ctx[r]['spec']['density']} veh/km/lane, "
              f"window {ctx[r]['window']} s")

    res, npz = dict(design=dict(seeds=list(args.seeds), regimes=list(regimes),
                                preset=args.preset, n_lhs=args.n_lhs)), {}
    t0 = time.time()
    with RunLogger("term_influence",
                   note=f"instruments={','.join(args.only)} preset={args.preset}",
                   params=dict(seeds=",".join(map(str, args.seeds)),
                               regimes=",".join(regimes),
                               n_lhs=args.n_lhs)) as rl:
        if "sensitivity" in args.only:
            print("\n1. sensitivity / identifiability")
            s = run_sensitivity(ctx, base, tuple(args.seeds), args.n_lhs, regimes)
            for k in ("X", "Y", "rho", "beta", "r2"):
                npz[f"sens_{k}"] = s.pop(k)
            res["sensitivity"] = s
            print("\n  primary response per parameter:")
            for n, d in s["primary"].items():
                if d:
                    print(f"    {n:14s} -> {d['response']:24s} rho {d['rho']:+.3f}"
                          f"{'' if d['significant'] else '   (not significant)'}")

        if "budget" in args.only:
            print("\n2. force budget")
            res["budget"] = run_budget(base, tuple(args.seeds), regimes)

        if "ablation" in args.only:
            print("\n3. ablation")
            # Only the N-stage ablations: an E-stage term provably changes
            # nothing in a scenario with no emergency vehicle in it, so scoring
            # it here would fill the table with zeros that look like findings.
            # The E ablations are run by run_ev_calibration.py --ablate against
            # the dashcam-ground-truth objective, where they are observable.
            print(f"  N-stage terms only ({len(terms.N_ABLATIONS)}); the "
                  f"{len(terms.E_ABLATIONS)} E-stage ones need an EV present "
                  f"and are scored by run_ev_calibration.py --ablate")
            res["ablation"] = run_ablation(ctx, base, tuple(args.seeds),
                                          regimes, list(terms.N_ABLATIONS))

        res["wall_s"] = round(time.time() - t0, 1)
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(ROOT, args.out)
        json.dump(res, open(path, "w"), indent=1, default=float)
        if npz:
            np.savez_compressed(path.replace(".json", ".npz"), **npz)
        print(f"\nwrote {path}" + (" (+ .npz)" if npz else ""))
        m = dict(wall_s=res["wall_s"])
        if "ablation" in res:
            m["baseline_loss"] = res["ablation"]["_baseline"]["loss"]
        if "sensitivity" in res:
            m["n_lhs"] = args.n_lhs
        rl.finish(metrics=m, outputs=[args.out])


if __name__ == "__main__":
    main()
