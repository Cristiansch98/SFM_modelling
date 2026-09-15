"""Figures and tables for the highD calibration/validation study.

    python experiments/make_highd_figs.py [--no-corr]

Reads out/highd_recordings.json, out/highd_targets.json, out/highd_pools.npz,
out/highd_calibration.json and (when present) out/highd_validation.{json,npz}.
Writes out/fig_highd_*.png and out/highd_tables.md.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emv import empirical, viz
from emv import highd_fit as hf
from emv.runlog import log_run
from emv.scenarios import HIGHD_REGIMES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
PAL, CAT = viz.LIGHT, viz.CAT_LIGHT
REG_COLOR = dict(freeflow=CAT["blue"], dense=CAT["yellow"],
                 congested=CAT["red"], other=PAL["muted"])


def _load(name, required=True):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        if required:
            raise SystemExit(f"{p} missing - run the extract/calibration first")
        return None
    return np.load(p) if name.endswith(".npz") else json.load(open(p))


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=160, facecolor=PAL["page"], bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)
    return p


def _ax(ax, title="", xlabel="", ylabel=""):
    viz._style_ax(ax, PAL)
    if title:
        ax.set_title(title, color=PAL["ink"], fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=PAL["ink2"], fontsize=9)
    ax.set_ylabel(ylabel, color=PAL["ink2"], fontsize=9)
    return ax


# ------------------------------------------------------------------ figures
def fig_regimes(recs):
    """Where the 88 real carriageways sit, and where the model's scenarios do."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=PAL["page"])
    _ax(ax, "highD carriageways and the model's scenarios",
        "density (veh/km/lane)", "median speed (m/s)")
    seen = set()
    for blk in recs["recordings"].values():
        for o in blk["observables"].values():
            r = o.get("regime", "other")
            ax.scatter(o.get("density_veh_km_lane"), o.get("bg_med_speed"),
                       s=26, alpha=.75, lw=0, color=REG_COLOR.get(r, PAL["muted"]),
                       label=r if r not in seen else None)
            seen.add(r)
    for name, sp in HIGHD_REGIMES.items():
        ax.scatter(sp["density"], np.mean(sp["lane_speeds"]), marker="*", s=260,
                   color=REG_COLOR.get(name), edgecolor=PAL["ink"], lw=.9, zorder=5)
    # the model's own scenarios of record
    ax.scatter(22.0, 26.1, marker="X", s=110, color=PAL["ink"], zorder=6)
    ax.annotate("make_overtake\n(22 veh/km/lane)", (22.0, 26.1),
                textcoords="offset points", xytext=(10, 6),
                fontsize=8, color=PAL["ink"])
    ax.scatter(77.0, 1.9, marker="X", s=110, color=CAT["violet"], zorder=6)
    ax.annotate("make_jam - outside\nhighD's coverage", (77.0, 1.9),
                textcoords="offset points", xytext=(-6, 16),
                fontsize=8, color=CAT["violet"])
    ax.axhspan(0, 12, color=CAT["red"], alpha=.05)
    ax.set_xlim(0, 85)
    ax.set_ylim(0, 40)
    ax.legend(frameon=False, fontsize=8, loc="upper right",
              title="carriageway regime", title_fontsize=8)
    ax.text(.99, .02, "stars = the scenario built from each regime's measurement",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=PAL["muted"])
    return _save(fig, "fig_highd_regimes.png")


def fig_window(cal):
    """The measurement-window artefact: half-manoeuvre vs full manoeuvre."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), facecolor=PAL["page"])
    obs = cal["observables"]
    tgt = cal["targets"]
    ax = _ax(axes[0], "Lane-change duration: which window is measured",
             "", "seconds")
    labels = ["shipped window\n(arrival half)", "full manoeuvre",
              "centre-to-centre"]
    model = [obs.get("lc_dur_med"), obs.get("lc_dur_full_med"),
             obs.get("lc_c2c_med")]
    real = [empirical.measured_value("lc_dur_med"),
            empirical.measured_value("lc_dur_full_med"),
            empirical.measured_value("lc_c2c_med")]
    x = np.arange(3)
    ax.bar(x - .19, model, .36, color=CAT["blue"], label="model (fitted)")
    ax.bar(x + .19, real, .36, color=CAT["aqua"], label="highD (measured)")
    lit = empirical.band("lc_dur_med")
    ax.axhline(lit[2], color=CAT["red"], ls="--", lw=1.2)
    ax.axhspan(lit[0], lit[1], color=CAT["red"], alpha=.08)
    ax.text(2.42, lit[2] + .1, "NGSIM 4.01 $\\pm$ 2.31 s\n(full manoeuvre)",
            fontsize=7.5, color=CAT["red"], ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = _ax(axes[1], "Lane-change rate: three definitions", "", "per veh-km")
    rec = empirical.lc_rate_reconciliation()
    keys = ["published", "complete_like_for_like", "all_events"]
    names = ["highD paper\n(as printed)", "like-for-like\n(this repo)",
             "all recorded\nevents"]
    vals = [rec[k]["value"] for k in keys]
    ax.bar(np.arange(3), vals, .55,
           color=[CAT["red"], CAT["aqua"], CAT["blue"]])
    for i, v in enumerate(vals):
        ax.text(i, v + .006, f"{v:.3f}", ha="center", fontsize=8.5,
                color=PAL["ink"])
    ax.axhline(obs.get("lc_per_veh_km", 0), color=PAL["ink"], ls=":", lw=1.3)
    ax.text(2.45, obs.get("lc_per_veh_km", 0) + .006,
            "model (EV-induced only)", ha="right", fontsize=7.5, color=PAL["ink"])
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(names, fontsize=8)
    fig.tight_layout()
    return _save(fig, "fig_highd_lcrate.png")


def fig_observables(cal, val):
    """Model against the measured band, before and after calibration."""
    keys = [k for k, _, w in hf.OBS_SPEC if w > 0]
    tgt, sd = cal["targets"], cal["target_sd"]
    after = cal["observables"]
    before = None
    if val and "holdout" in val:
        b = val["holdout"].get("train:freeflow", {})
        before = b.get("tuned", {}).get("observables")
    n = len(keys)
    LIM = 30.0
    fig, ax = plt.subplots(figsize=(8.2, 0.44 * n + 2.1), facecolor=PAL["page"])
    _ax(ax, "Model vs highD, in units of between-carriageway s.d.",
        "(model $-$ highD) / s.d. of the real spread", "")
    y = np.arange(n)[::-1]

    def put(val, yi, color, size, label):
        """Points beyond the axis are drawn at the edge as a triangle with their
        true value annotated, so a far-out observable is visible rather than
        silently missing. The two series are offset in y so they never hide each
        other when both are off-scale."""
        v = float(np.clip(val, -LIM, LIM))
        off = abs(val) > LIM
        ax.scatter(v, yi, s=size, color=color, zorder=5, label=label,
                   marker=(">" if val > 0 else "<") if off else "o")
        if off:
            ax.annotate(f"{val:+.0f}", (v, yi), textcoords="offset points",
                        xytext=(-17 if val > 0 else 17, -2), fontsize=7,
                        color=color, ha="center", va="center")

    for i, k in enumerate(keys):
        s = sd.get(k) or float("nan")
        ax.plot([-2, 2], [y[i], y[i]], color=PAL["grid"], lw=9,
                solid_capstyle="butt", zorder=1)
        if before and k in before:
            put((before[k] - tgt[k]) / s, y[i] + 0.17, PAL["muted"], 46,
                "previous parameters" if i == 0 else None)
        put((after[k] - tgt[k]) / s, y[i] - 0.17, CAT["blue"], 64,
            "highD-calibrated" if i == 0 else None)
    ax.axvline(0, color=PAL["ink"], lw=1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(keys, fontsize=8.5)
    ax.set_xlim(-LIM * 1.6, LIM * 1.6)
    ax.set_ylim(-1.1, n - 0.3)
    ax.set_xscale("symlog", linthresh=2)
    ax.text(0.0, -0.95, "grey band = within 2 s.d. of the natural variation "
            "between real carriageways", ha="center", fontsize=7.5,
            color=PAL["muted"])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, "fig_highd_observables.png")


def _model_pools(cal, seeds=(11, 12, 13, 14, 15)):
    """Pooled model samples under the previous and the calibrated parameters,
    from the same scenario and window the calibration used."""
    from emv import metrics
    from emv.params import highd_params, tuned_params
    from emv.scenarios import make_highd_like
    regime = cal.get("regime", "freeflow")
    spec = cal.get("regime_spec") or None
    win = cal.get("obs_window")
    out = {}
    for label, p in (("before", tuned_params().copy(dt=0.06)),
                     ("after", highd_params().copy(dt=0.06))):
        acc = []
        for s in seeds:
            h = make_highd_like(seed=s, p=p, regime=regime, spec=spec,
                                road_len=3000.0).run(110.0, rec_dt=0.12,
                                                     stop_when_ev_x=2950.0)
            acc.append(metrics.behaviour_pools(h, track_window=win))
        out[label] = {k: (np.concatenate([q[k] for q in acc])
                          if not np.isscalar(acc[0][k])
                          else float(np.sum([q[k] for q in acc])))
                      for k in acc[0]}
    return out


def fig_cdf(pools, cal, model=None):
    """Distribution overlays: the shape behind the percentiles.

    Percentile targets can be hit by distributions of quite different shape, so
    the whole empirical CDF is shown for both sides.
    """
    panels = [("lc_dur_full", "lane-change duration (s)", (0, 8)),
              ("lc_peak_vy_full", "peak |v$_{lat}$| per manoeuvre (m/s)", (0, 2.5)),
              ("trk_offset_abs", "lane-keeping |offset| (m)", (0, 1.2)),
              ("trk_peak_alat", "per-track peak |a$_{lat}$| (m/s$^2$)", (0, 3))]
    #: highD pools carry the shipped-window lane-change keys under their own
    #: names; the model side uses the full-manoeuvre ones.
    HD = {"lc_dur_full": "lc_dur_full", "lc_peak_vy_full": "lc_peak_vy_full"}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), facecolor=PAL["page"])
    for ax, (key, lbl, xlim) in zip(axes.ravel(), panels):
        _ax(ax, "", lbl, "cumulative fraction")
        src = HD.get(key, key)
        real = np.asarray(pools[src], float) if src in pools.files else np.zeros(0)
        real = real[np.isfinite(real)]
        if real.size:
            ax.plot(np.sort(real), np.linspace(0, 1, real.size),
                    color=CAT["aqua"], lw=2.4, label=f"highD (n={real.size})",
                    zorder=4)
        if model:
            for label, color, ls in (("before", PAL["muted"], "--"),
                                     ("after", CAT["blue"], "-")):
                a = np.asarray(model[label].get(key, []), float)
                a = a[np.isfinite(a)]
                if a.size:
                    ax.plot(np.sort(a), np.linspace(0, 1, a.size), color=color,
                            lw=1.8, ls=ls, zorder=3,
                            label=("model, previous params" if label == "before"
                                   else "model, highD-calibrated"))
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("Distributions, not just the percentiles that were fitted",
                 color=PAL["ink"], fontsize=11, x=.02, ha="left")
    fig.tight_layout()
    return _save(fig, "fig_highd_cdf.png")


def fig_correlation(val, npz):
    c = val["correlation"]
    names, resp = c["names"], c["responses"]
    R = np.array([[c["rho"][n][r] for r in resp] for n in names], float)
    fig, ax = plt.subplots(figsize=(0.62 * len(resp) + 3.2,
                                    0.5 * len(names) + 2.2), facecolor=PAL["page"])
    _ax(ax, "Which parameter drives which mismatch (Spearman $\\rho$)", "", "")
    im = ax.imshow(R, cmap=viz.CMAP_DIV, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(resp)))
    ax.set_xticklabels(resp, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    thr = c["significant"]
    for i in range(len(names)):
        for j in range(len(resp)):
            if abs(R[i, j]) > thr:
                ax.text(j, i, f"{R[i, j]:+.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if abs(R[i, j]) > .55 else PAL["ink"])
    fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    ax.text(0, -.9, f"N={c['n']}, |rho| > {thr} significant",
            fontsize=7.5, color=PAL["muted"])
    return _save(fig, "fig_highd_correlation.png")


def write_tables(recs, cal, val):
    L = ["# highD calibration and validation - tables", "",
         "Generated by `experiments/make_highd_figs.py`. Method: "
         "`docs/HIGHD_VALIDATION.md`.", ""]
    ds = recs["meta"]
    L += ["## 1. Dataset", "",
          f"- {ds['n_recordings']} recordings, {ds['n_vehicles']} vehicles, "
          f"{ds['veh_km']} veh-km, {ds['driven_h']} driven hours",
          f"- lane width median {ds['lane_width_med']} m "
          f"({ds['n_lane_widths']} lanes, {ds['lane_width_min']}-{ds['lane_width_max']})",
          f"- {ds['n_lane_changes']} lane changes -> {ds['lc_per_veh_km']} per veh-km",
          ""]
    rec = empirical.lc_rate_reconciliation()
    L += ["## 2. Lane-change rate: three definitions", "",
          "| definition | per veh-km | note |", "|---|---|---|"]
    for k, v in rec.items():
        L.append(f"| {v['label']} | {v['value']} | {v['note']} |")
    L += ["", "## 3. Calibrated parameters", "",
          "| parameter | default | highD-fitted |", "|---|---|---|"]
    from emv.params import Params
    for k, v in cal["theta"].items():
        L.append(f"| `{k}` | {getattr(Params(), k)} | **{v}** |")
    L += ["", f"Objective {cal['loss']} against {cal['loss_tuned']} for the "
          f"previous parameters ({cal['n_evals']} evaluations, "
          f"targets `{cal['targets_block']}`).", ""]
    L += ["## 4. Observables", "",
          "| observable | highD | s.d. | model | dev/s.d. | fitted |",
          "|---|---|---|---|---|---|"]
    for k, scale, w in hf.OBS_SPEC:
        if k not in cal["targets"]:
            continue
        t = cal["targets"][k]
        s = cal["target_sd"].get(k) or float("nan")
        m = cal["observables"].get(k, float("nan"))
        L.append(f"| `{k}` | {t:.3f} | {s:.3f} | {m:.3f} | "
                 f"{abs(m-t)/s:.2f} | {'yes' if w else 'no'} |")
    if val and "selfdist" in val:
        L += ["", "## 5. Acceptance bar (highD's own self-distance)", "",
              "| pool | carriageways | mean | median | p90 |", "|---|---|---|---|---|"]
        for r, v in val["selfdist"].items():
            L.append(f"| {r} | {v['n_cells']} | {v['mean']} | {v['med']} | {v['p90']} |")
    if val and "holdout" in val:
        L += ["", "## 6. Holdout", "",
              "| target block | previous | highD-calibrated | observables within 2 s.d. |",
              "|---|---|---|---|"]
        for name, row in val["holdout"].items():
            L.append(f"| {name} | {row['tuned']['distance']['total']} | "
                     f"{row['highd']['distance']['total']} | "
                     f"{row['n_pass']}/{row['n_obs']} |")
    if val and "tradeoff" in val:
        a, b = val["tradeoff"]["tuned"], val["tradeoff"]["highd"]
        L += ["", "## 7. What the realism costs the EV", "",
              "| metric | previous | highD-calibrated |", "|---|---|---|"]
        for k in ("ev_speed_ratio", "ev_mean_speed", "min_ttc", "collisions",
                  "clearance_mean", "react_dist_mean", "d_star", "a_pin_eff"):
            if k in a:
                L.append(f"| `{k}` | {a[k]} | {b[k]} |")
    p = os.path.join(OUT, "highd_tables.md")
    open(p, "w").write("\n".join(L) + "\n")
    print("wrote", p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-corr", action="store_true")
    ap.add_argument("--no-model-cdf", action="store_true",
                    help="skip the model runs behind the CDF overlay (~30 s)")
    args = ap.parse_args()
    recs = _load("highd_recordings.json")
    cal = _load("highd_calibration.json")
    pools = _load("highd_pools.npz")
    val = _load("highd_validation.json", required=False)
    npz = _load("highd_validation.npz", required=False)
    model = None if args.no_model_cdf else _model_pools(cal)
    outs = [fig_regimes(recs), fig_window(cal), fig_cdf(pools, cal, model),
            fig_observables(cal, val)]
    if val and "correlation" in val and not args.no_corr:
        outs.append(fig_correlation(val, npz))
    outs.append(write_tables(recs, cal, val))
    log_run("highd_figs", note="highD study figures + tables",
            outputs=[os.path.relpath(o, ROOT) for o in outs])


if __name__ == "__main__":
    main()
