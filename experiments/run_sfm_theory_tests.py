"""Quantitative tests of the SFM EV-yielding formulation, beyond calibration.

Four studies. All use `emv.scenarios` + `emv.metrics`; none need SUMO.

  DEPIN     the closed-form predictions, measured. Sweep (A_c, B_c, a_pin) with
            the density stiffening switched off (k_rho=0) so a_pin_eff is
            unambiguous, run the overtake scenario, and from each run extract
              * the cleared half-width d_meas  = median |y - y_corr| of the
                yielding vehicles that left the EV's path, and
              * the depinning order parameter phi = fraction of the vehicles
                that had to move that did.
            Compare d_meas with the analytic
              d* = w_need + B_c ln(A_c ubar / a_pin_eff),
              a_pin_eff = a_pin (1 - eps ubar),
            and plot phi against the reduced control x = A_c ubar / a_pin_eff
            (theory: escape iff x > 1). A jam grid is added for the collapse.

  PREDPATH  is it the *predicted* path that matters, or the EV's instantaneous
            position? Lower the corridor-length floor to 10 m and sweep the
            look-ahead T_pred from 1 s (corridor hugs the EV) to 16 s
            (calibrated regime). Overtake + jam.

  ENVELOPE  operating envelope. Sweep traffic density (overtake) and jam
            spacing; record EV progress, clearance, min TTC, collisions,
            disruption. Locates the density at which the free-flow calibration
            stops being collision-free.

  CONVERGE  numerical soundness. Sweep the integrator step dt and show the
            headline metrics are converged at the production dt = 0.05 s.

    .venv/bin/python experiments/run_sfm_theory_tests.py            # ~15 min
    .venv/bin/python experiments/run_sfm_theory_tests.py --quick    # ~1 min
    .venv/bin/python experiments/run_sfm_theory_tests.py --only depin predpath

Writes out/sfm_theory_tests.json; self-logs kind='theory_tests'.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv.metrics import behaviour_stats, evaluate
from emv.params import bluelight_params
from emv.runlog import RunLogger
from emv.scenarios import make_jam, make_overtake
from emv.state import YIELDING

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")


# --------------------------------------------------------------- measurement
def corridor_stats(h, s_lo=4.0, s_hi=45.0):
    """From a finished run: depinning order parameter phi, measured cleared
    half-width d_meas, and the mean urgency ubar of the vehicles that had to
    move. 'Had to move' = started within w_need of the corridor line. Pooled
    over ALL frames in a tight window just ahead of the EV, where the corridor
    is fully formed (the ps_corridor.png convention)."""
    ev = h.ev
    n = h.x.shape[1]
    others = np.arange(n) != ev
    wn = 0.5 * (h.W[ev] + h.W) + h.params.margin_c          # per-vehicle w_need
    d0 = np.abs(h.y[0] - h.y_corr)
    need = others & (d0 < wn)                               # in the EV's path at t0
    tt = h.t
    ind, dcl, ucl, uall = [], [], [], []
    # per-vehicle settled state: median |d_perp| over the vehicle's last 1.5 s
    # in the band, and whether it ever began to move (|d_perp| > 0.5 w_need)
    per_left, per_onset = [], []
    for i in np.flatnonzero(need):
        s_i = h.x[:, i] - h.x[:, ev]
        inb = (s_i > s_lo) & (s_i < s_hi)
        kk = np.flatnonzero(inb)
        if kk.size == 0:
            continue
        dperp_i = np.abs(h.y[kk, i] - h.y_corr)
        uall.extend(h.u[kk, i].tolist())
        ind.extend((dperp_i > wn[i]).tolist())
        per_onset.append(float(np.max(dperp_i) > 0.5 * wn[i]))
        tail = tt[kk] >= tt[kk][-1] - 1.5
        d_settled = float(np.median(dperp_i[tail]))
        per_left.append(float(d_settled > wn[i]))
        yld = h.aware[kk, i] == YIELDING
        m = (dperp_i > wn[i]) & yld
        if m.any():
            dcl.extend(dperp_i[m].tolist())
            ucl.extend(h.u[kk, i][m].tolist())
    phi_pooled = float(np.mean(ind)) if ind else 0.0
    phi = float(np.mean(per_left)) if per_left else 0.0        # settled fraction
    phi_onset = float(np.mean(per_onset)) if per_onset else 0.0
    ubar = float(np.mean(uall)) if uall else np.nan
    if len(dcl) >= 10:
        d_meas = float(np.median(dcl))
        d_p25, d_p75 = (float(np.percentile(dcl, 25)),
                        float(np.percentile(dcl, 75)))
        ubar_cl = float(np.mean(ucl))
    else:
        d_meas = d_p25 = d_p75 = ubar_cl = np.nan
    return dict(phi=phi, phi_pooled=phi_pooled, phi_onset=phi_onset,
                ubar=ubar, ubar_cleared=ubar_cl, n_cleared=len(dcl),
                d_meas=d_meas, d_p25=d_p25, d_p75=d_p75,
                w_need=float(np.median(wn[need])))


def d_analytic(p, A_c, B_c, a_pin, ubar, w_need):
    """d* = w_need + B_c ln(A_c ubar / a_pin_eff),  a_pin_eff = a_pin(1-eps u).
    Returns (d_star, x_control). k_rho is assumed 0 (rho_fac = 1)."""
    eps = p.urgency_pin_relief
    a_pin_eff = a_pin * (1.0 - eps * ubar)
    x = A_c * ubar / max(a_pin_eff, 1e-9)
    return w_need + B_c * np.log(max(x, 1e-9)), float(x)


# --------------------------------------------------------------- study: depin
def study_depin(seeds, log, quick=False):
    base = bluelight_params().copy(dt=0.06, k_rho=0.0)          # clean a_pin_eff
    A_cs = [1.0, 1.4, 2.0, 2.8, 3.8, 5.0]
    a_pins = [0.7, 1.1, 1.6, 2.2]
    B_cs = [0.8, 1.4, 2.2]
    if quick:
        A_cs, a_pins, B_cs = [1.4, 3.8], [0.7, 2.2], [1.4]
    rows = []
    for Ac in A_cs:
        for Bc in B_cs:
            for ap in a_pins:
                p = base.copy(A_c=float(Ac), B_c=float(Bc), a_pin=float(ap))
                acc = []
                for s in seeds:
                    h = make_overtake(seed=s, p=p, density=20.0,
                                      road_len=1500.0).run(
                        60.0, rec_dt=0.2, stop_when_ev_x=1450.0)
                    st = corridor_stats(h)
                    st["collisions"] = evaluate(h)["collisions"]
                    acc.append(st)
                phi = float(np.mean([a["phi"] for a in acc]))
                phi_on = float(np.mean([a["phi_onset"] for a in acc]))
                phi_pl = float(np.mean([a["phi_pooled"] for a in acc]))
                ub = float(np.nanmean([a["ubar"] for a in acc]))
                dms = [a["d_meas"] for a in acc if np.isfinite(a["d_meas"])]
                ucls = [a["ubar_cleared"] for a in acc
                        if np.isfinite(a["ubar_cleared"])]
                dm = float(np.median(dms)) if dms else np.nan
                wn = float(np.nanmedian([a["w_need"] for a in acc]))
                ub_cl = float(np.mean(ucls)) if ucls else np.nan
                u_use = ub_cl if np.isfinite(ub_cl) else ub
                dstar, xctl = d_analytic(p, Ac, Bc, ap, u_use, wn)
                rows.append(dict(scen="overtake", A_c=Ac, B_c=Bc, a_pin=ap,
                                 phi=phi, phi_onset=phi_on, phi_pooled=phi_pl,
                                 ubar=ub, ubar_cleared=ub_cl, u_used=u_use,
                                 d_meas=dm, d_star=dstar, x_ctrl=xctl, w_need=wn,
                                 n_cleared=float(np.mean([a["n_cleared"]
                                                          for a in acc]))))
                log(f"  depin OT A_c={Ac} B_c={Bc} a_pin={ap}  "
                    f"phi={phi:.2f} on={phi_on:.2f} d_meas={dm:.2f} "
                    f"d*={dstar:.2f} x={xctl:.2f}")
    # jam grid, for the phi(x) collapse only
    jg = ([(1.5, 1.4)] if quick else
          [(Ac, ap) for Ac in (0.8, 1.6, 2.6, 4.0) for ap in (0.8, 1.4, 2.2)])
    for Ac, ap in jg:
        p = base.copy(A_c=float(Ac), B_c=2.0, a_pin=float(ap))
        acc = []
        for s in seeds:
            h = make_jam(seed=s + 100, p=p, road_len=560.0).run(
                55.0, rec_dt=0.2, stop_when_ev_x=520.0)
            acc.append(corridor_stats(h))
        phi = float(np.mean([a["phi"] for a in acc]))
        phi_on = float(np.mean([a["phi_onset"] for a in acc]))
        phi_pl = float(np.mean([a["phi_pooled"] for a in acc]))
        ub = float(np.nanmean([a["ubar"] for a in acc]))
        _, xctl = d_analytic(p, Ac, 2.0, ap, ub, 2.4)
        rows.append(dict(scen="jam", A_c=Ac, B_c=2.0, a_pin=ap, phi=phi,
                         phi_onset=phi_on, phi_pooled=phi_pl,
                         ubar=ub, ubar_cleared=np.nan, u_used=ub, d_meas=np.nan,
                         d_star=np.nan, x_ctrl=xctl, w_need=2.4, n_cleared=0.0))
        log(f"  depin JAM A_c={Ac} a_pin={ap}  phi={phi:.2f} x={xctl:.2f}")
    return rows


# ------------------------------------------------------------ study: predpath
def study_predpath(seeds, log, quick=False):
    """Does the *predicted* path matter? Lower the corridor-length floor so
    T_pred actually controls how far ahead the corridor reaches, and sweep it
    from 1 s (corridor hugs the EV -> near-instantaneous) to 16 s (calibrated
    regime). Two free-flow densities; the jam is collision-saturated at these
    parameters and would only add noise, so it is left out."""
    tps = [1.0, 2.0, 4.0, 8.0, 16.0] if not quick else [1.0, 16.0]
    cfgs = {"overtake_d22": 22.0, "overtake_d30": 30.0}
    if quick:
        cfgs = {"overtake_d22": 22.0}
    out = {}
    for name, dens in cfgs.items():
        rows = []
        for tp in tps:
            p = bluelight_params().copy(dt=0.06, L_pred_min=10.0,
                                        T_pred=float(tp))
            ms = []
            for s in seeds:
                h = make_overtake(seed=s, p=p, density=dens,
                                  road_len=1500.0).run(
                    60.0, rec_dt=0.12, stop_when_ev_x=1450.0)
                m = {k: v for k, v in evaluate(h).items()
                     if isinstance(v, (int, float))}
                m.update(behaviour_stats(h))
                ms.append(m)
            agg = {k: float(np.nanmean([m.get(k, np.nan) for m in ms]))
                   for k in ("ev_speed_ratio", "clearance_mean", "min_ttc",
                             "collisions", "react_dist_mean", "disruption")}
            agg["T_pred"] = tp
            agg["clearance_sd"] = float(np.nanstd(
                [m["clearance_mean"] for m in ms]))
            agg["speed_sd"] = float(np.nanstd(
                [m["ev_speed_ratio"] for m in ms]))
            rows.append(agg)
            log(f"  predpath {name} T_pred={tp}  "
                f"spd={agg['ev_speed_ratio']:.3f} "
                f"clr={agg['clearance_mean']:.1f} coll={agg['collisions']:.1f}")
        out[name] = rows
    return out


# ------------------------------------------------------------ study: envelope
def study_envelope(seeds, log, quick=False):
    dens = [6, 9, 12, 15, 18, 22, 26, 30, 35] if not quick else [10, 26]
    spac = [28, 20, 15, 12, 9, 7] if not quick else [20, 9]
    keys = ("ev_speed_ratio", "clearance_mean", "min_ttc", "collisions",
            "disruption")
    ot = []
    for d in dens:
        ms = []
        for s in seeds:
            h = make_overtake(seed=s, p=bluelight_params().copy(dt=0.06),
                              density=float(d), road_len=1500.0).run(
                60.0, rec_dt=0.12, stop_when_ev_x=1450.0)
            ms.append({k: v for k, v in evaluate(h).items()
                       if isinstance(v, (int, float))})
        row = {k: float(np.nanmean([m.get(k, np.nan) for m in ms]))
               for k in keys}
        row["density"] = d
        row["coll_sd"] = float(np.nanstd([m["collisions"] for m in ms]))
        ot.append(row)
        log(f"  envelope OT rho={d}  spd={row['ev_speed_ratio']:.3f} "
            f"coll={row['collisions']:.1f}")
    jm = []
    for sp in spac:
        ms = []
        for s in seeds:
            h = make_jam(seed=s + 100, p=bluelight_params().copy(dt=0.06),
                         spacing=float(sp), road_len=620.0).run(
                55.0, rec_dt=0.12, stop_when_ev_x=580.0)
            ms.append({k: v for k, v in evaluate(h).items()
                       if isinstance(v, (int, float))})
        row = {k: float(np.nanmean([m.get(k, np.nan) for m in ms]))
               for k in keys}
        row["spacing"] = sp
        row["density_eq"] = 1000.0 / sp
        jm.append(row)
        log(f"  envelope JAM spc={sp}  spd={row['ev_speed_ratio']:.3f} "
            f"coll={row['collisions']:.1f}")
    return dict(overtake=ot, jam=jm)


# ------------------------------------------------------------ study: converge
def study_converge(seeds, log, quick=False):
    dts = [0.10, 0.05, 0.025, 0.0125] if not quick else [0.1, 0.05]
    keys = ("ev_speed_ratio", "clearance_mean", "min_ttc", "disruption",
            "react_dist_mean")
    out = {}
    for sc in ("overtake", "jam"):
        rows = []
        for dt in dts:
            ms = []
            for s in seeds:
                p = bluelight_params().copy(dt=float(dt))
                if sc == "overtake":
                    h = make_overtake(seed=s, p=p, density=22.0,
                                      road_len=1500.0).run(
                        60.0, rec_dt=0.1, stop_when_ev_x=1450.0)
                else:
                    h = make_jam(seed=s + 100, p=p, road_len=560.0).run(
                        55.0, rec_dt=0.1, stop_when_ev_x=520.0)
                ms.append({k: v for k, v in evaluate(h).items()
                           if isinstance(v, (int, float))})
            rows.append(dict(dt=dt, **{k: float(np.nanmean(
                [m.get(k, np.nan) for m in ms])) for k in keys}))
            log(f"  converge {sc} dt={dt}")
        out[sc] = rows
    return out


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="+",
                    default=["depin", "predpath", "envelope", "converge"],
                    choices=["depin", "predpath", "envelope", "converge"])
    a = ap.parse_args()

    s_dep, s_pp, s_env, s_cv = (11, 12, 13), (11, 12, 13, 14, 15), \
        tuple(range(11, 17)), tuple(range(11, 17))
    if a.quick:
        s_dep = s_pp = s_env = s_cv = (11, 12)

    t0 = time.time()
    def log(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    jpath = os.path.join(OUT, "sfm_theory_tests.json")
    # merge with an existing file so `--only` can refresh one study in place
    summary = {}
    if os.path.exists(jpath) and set(a.only) != {"depin", "predpath",
                                                 "envelope", "converge"}:
        try:
            summary = json.load(open(jpath))
        except Exception:
            summary = {}
    summary["design"] = dict(
        quick=a.quick, only=a.only,
        seeds=dict(depin=list(s_dep), predpath=list(s_pp),
                   envelope=list(s_env), converge=list(s_cv)),
        base="bluelight_params",
        note="depin sweep sets k_rho=0 for a clean a_pin_eff; "
             "predpath sets L_pred_min=10")

    with RunLogger("theory_tests",
                   note=f"{'quick' if a.quick else 'full'}; "
                        f"{','.join(a.only)}") as rl:
        if "depin" in a.only:
            log("study DEPIN: cleared half-width vs closed form + phi(x)")
            summary["depin"] = study_depin(s_dep, log, a.quick)
        if "predpath" in a.only:
            log("study PREDPATH: look-ahead sweep")
            summary["predpath"] = study_predpath(s_pp, log, a.quick)
        if "envelope" in a.only:
            log("study ENVELOPE: density / spacing sweep")
            summary["envelope"] = study_envelope(s_env, log, a.quick)
        if "converge" in a.only:
            log("study CONVERGE: dt sweep")
            summary["converge"] = study_converge(s_cv, log, a.quick)

        # regression + rank-correlation summary for DEPIN
        if "depin" in summary:
            reliable = [r for r in summary["depin"] if r["scen"] == "overtake"
                        and np.isfinite(r["d_meas"]) and np.isfinite(r["d_star"])
                        and r["phi_pooled"] > 0.55 and r["x_ctrl"] > 1.3]
            if len(reliable) >= 6:
                xa = np.array([r["d_star"] for r in reliable])
                ya = np.array([r["d_meas"] for r in reliable])
                # Spearman over the whole reliable set (monotone tracking)
                def _rank(a):
                    o = np.argsort(a, kind="mergesort")
                    rr_ = np.empty(len(a)); rr_[o] = np.arange(len(a))
                    return rr_
                rx, ry = _rank(xa) - _rank(xa).mean(), _rank(ya) - _rank(ya).mean()
                rho = float((rx * ry).sum()
                            / np.sqrt((rx**2).sum() * (ry**2).sum()))
                # OLS on the sub-saturation band only
                sub = [r for r in reliable if r["d_star"] < 4.2]
                xs = np.array([r["d_star"] for r in sub])
                ys = np.array([r["d_meas"] for r in sub])
                (slope, icpt), *_ = np.linalg.lstsq(
                    np.c_[xs, np.ones_like(xs)], ys, rcond=None)
                pred = slope * xs + icpt
                r2 = 1 - np.sum((ys - pred)**2) / np.sum((ys - ys.mean())**2)
                rmse = float(np.sqrt(np.mean((ys - xs) ** 2)))
                summary["depin_fit"] = dict(
                    n_reliable=len(reliable), spearman_rho=rho,
                    n_sub=len(sub), slope=float(slope), intercept=float(icpt),
                    r2_sub=float(r2), rmse_vs_identity_sub=rmse,
                    d_meas_range=[float(ya.min()), float(ya.max())],
                    d_star_range=[float(xa.min()), float(xa.max())],
                    d_meas_saturation=float(np.median(
                        [r["d_meas"] for r in reliable if r["d_star"] > 5.5])))
                log(f"  DEPIN: rho={rho:.2f} (n={len(reliable)}); "
                    f"sub-band OLS slope={slope:.2f} R2={r2:.2f} "
                    f"RMSE={rmse:.2f} m (n={len(sub)}); "
                    f"d_meas saturates ~{summary['depin_fit']['d_meas_saturation']:.1f} m")
            # logistic midpoint of the settled order parameter, overtake only
            ot = [(r["x_ctrl"], r["phi"]) for r in summary["depin"]
                  if r["scen"] == "overtake" and np.isfinite(r["x_ctrl"])]
            if len(ot) >= 8:
                xs = np.array([p for p, _ in ot]); ys = np.array([q for _, q in ot])
                lx0, kk = 0.0, 2.0
                for _ in range(300):
                    z = kk * (np.log(xs) - lx0); sg = 1 / (1 + np.exp(-z))
                    r = sg - ys; ds = sg * (1 - sg)
                    g0 = np.sum(r * ds * (-kk)); gk = np.sum(r * ds * (np.log(xs) - lx0))
                    lx0 -= g0 / (np.sum((ds * kk) ** 2) + 1e-9)
                    kk -= gk / (np.sum((ds * (np.log(xs) - lx0)) ** 2) + 1e-9)
                summary["depin_transition"] = dict(x_mid=float(np.exp(lx0)),
                                                   slope_k=float(kk),
                                                   phi_below_1=float(np.mean(
                                                       [q for p, q in ot if p < 1.0])),
                                                   phi_onset_below_1=float(np.mean(
                                                       [r["phi_onset"] for r in summary["depin"]
                                                        if r["scen"] == "overtake" and r["x_ctrl"] < 1.0])))
                log(f"  DEPIN transition: x_mid={np.exp(lx0):.2f} "
                    f"phi(x<1)={summary['depin_transition']['phi_below_1']:.3f} "
                    f"onset(x<1)={summary['depin_transition']['phi_onset_below_1']:.3f}")

        with open(os.path.join(OUT, "sfm_theory_tests.json"), "w") as fh:
            json.dump(summary, fh, indent=1, default=float)
        log("wrote out/sfm_theory_tests.json")
        rl.finish(metrics=dict(studies=",".join(a.only), quick=a.quick),
                  outputs=["out/sfm_theory_tests.json"])


if __name__ == "__main__":
    main()
