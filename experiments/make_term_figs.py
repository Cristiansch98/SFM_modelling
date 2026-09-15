"""Figures and tables for the two-stage framework extension.

    python experiments/make_term_figs.py            # all figures + tables
    python experiments/make_term_figs.py --only budget ablation

Reads out/term_influence*.json(.npz), out/normal_calibration_*.json and
out/ev_calibration.json; writes out/fig_term_*.png and out/term_tables.md.
Nothing is recomputed here - every number comes from a stored record, so a
figure can never disagree with the table it is next to.

Figure conventions follow experiments/make_highd_figs.py (same palette, same
`_ax`/`_save` helpers via emv.viz) so these sit beside the existing study
figures without looking like a different project.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from emv import terms, viz
from emv import highd_fit as hf
from emv.runlog import log_run

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
PAL, CAT = viz.LIGHT, viz.CAT_LIGHT
TERM_COLOR = dict(drive=PAL["muted"], sfm=CAT["blue"], lane=CAT["aqua"],
                  lc=CAT["violet"], ev=CAT["red"], corr=CAT["yellow"])
STAGE_COLOR = dict(N=CAT["blue"], E=CAT["red"])


def _load(name, required=True):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        if required:
            raise SystemExit(f"{p} missing - run the experiment that writes it")
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
def fig_budget(ti):
    """Which term does the lateral and longitudinal work, EV absent vs present."""
    bud = ti["budget"]
    scen = [k for k in bud if k.endswith(":normal")] + \
           [k for k in bud if k.endswith(":bluelight")]
    order = ["drive", "sfm", "lane", "lc", "ev", "corr"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), facecolor=PAL["page"])
    # Lateral shares, then the longitudinal *interaction* shares. The raw
    # longitudinal budget is ~99 % `drive` at motorway speeds, which says
    # nothing; the informative quantity is what the interaction terms do relative
    # to each other, plus how often the IDM bound overrides them (annotated).
    for ax, key, lbl in zip(
            axes, ("share_y", "share_x_interaction"),
            ("across the road (lateral, all terms)",
             "along the road (interaction terms only; drive excluded)")):
        left = np.zeros(len(scen))
        for t in order:
            v = np.array([bud[s].get(key, {}).get(t, 0.0) for s in scen])
            if not v.any():
                continue
            ax.barh(np.arange(len(scen)), v, left=left, height=.58,
                    color=TERM_COLOR[t], label=t, edgecolor=PAL["page"], lw=.6)
            for i, (a, b) in enumerate(zip(left, v)):
                if b > .07:
                    ax.text(a + b / 2, i, f"{b*100:.0f}", ha="center",
                            va="center", fontsize=7.5, color="white")
            left += v
        _ax(ax, lbl, "share of the summed |acceleration|")
        ax.set_yticks(np.arange(len(scen)))
        ax.set_yticklabels([s.replace(":", "\n") for s in scen], fontsize=8)
        ax.set_xlim(0, 1)
    for i, s in enumerate(scen):
        d = bud[s].get("idm", {})
        if d:
            axes[1].text(1.02, i, f"IDM binds {d['binding_share']*100:.0f} %, "
                         f"adds {d['added_braking_when_binding']:.2f} m/s$^2$",
                         fontsize=7, va="center", color=PAL["ink2"])
    axes[0].legend(fontsize=7.5, ncol=6, loc="upper left",
                   bbox_to_anchor=(0.0, -0.30), frameon=False)
    fig.suptitle("Force budget per term: effort, not importance",
                 color=PAL["ink"], fontsize=11, x=.02, ha="left")
    fig.tight_layout(rect=(0, .10, .93, .93))
    return _save(fig, "fig_term_budget.png")


def fig_ablation(ti, ev=None):
    """Delta objective when one term is switched off - the tornado plot."""
    rows = []
    for name, d in ti.get("ablation", {}).items():
        if name.startswith("_"):
            continue
        rows.append((name, d["d_loss"], "N"))
    for name, d in (ev or {}).get("ablation", {}).items():
        rows.append((name, d["d_loss"], "E"))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.6, .34 * len(rows) + 1.5),
                           facecolor=PAL["page"])
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], height=.66,
            color=[STAGE_COLOR[r[2]] for r in rows])
    for i, (n, d, s) in enumerate(rows):
        ax.text(d + (0.02 if d >= 0 else -0.02) * max(abs(d) for _, d, _ in rows),
                i, f"{d:+.2f}", va="center", fontsize=7.5,
                ha="left" if d >= 0 else "right", color=PAL["ink2"])
    ax.axvline(0, color=PAL["ink2"], lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0].replace("no_", "- ") for r in rows], fontsize=8.5)
    _ax(ax, "Leave-one-term-out: change in the objective",
        "worse to the right  |  better to the left        (objective units)")
    ax.set_xscale("symlog", linthresh=1.0)
    h = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLOR[k]) for k in ("N", "E")]
    ax.legend(h, ["stage 1: ordinary traffic vs highD (23 observables)",
                  "stage 2: emergency vehicle vs dashcam (fingerprint)"],
              fontsize=7.5, frameon=False, loc="lower right")
    # The two stages are scored by DIFFERENT objectives, so bar lengths are only
    # comparable within a colour. Rank within each stage, never across.
    ax.text(0.0, -1.35, "The two stages use different objectives: compare "
            "within a colour, never across.", fontsize=7.5, color=CAT["red"],
            transform=ax.get_yaxis_transform(), va="top")
    fig.tight_layout()
    return _save(fig, "fig_term_ablation.png")


def fig_attribution(ti):
    """Term x observable: which observable does each term actually move."""
    abl = {k: v for k, v in ti.get("ablation", {}).items()
           if not k.startswith("_")}
    if not abl:
        return None
    regime = list(next(iter(abl.values()))["attribution"])[0]
    keys = [k for k, _, w in hf.OBS_SPEC_NORMAL if w > 0]
    M = np.full((len(abl), len(keys)), np.nan)
    for i, (n, d) in enumerate(abl.items()):
        for j, k in enumerate(keys):
            v = d["attribution"][regime].get(k)
            if v is not None:
                M[i, j] = v
    lim = np.nanpercentile(np.abs(M), 95) or 1.0
    fig, ax = plt.subplots(figsize=(1.0 + .42 * len(keys), 1.1 + .34 * len(abl)),
                           facecolor=PAL["page"])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90, fontsize=7.5, color=PAL["ink2"])
    ax.set_yticks(range(len(abl)))
    ax.set_yticklabels([n.replace("no_", "- ") for n in abl], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]) and abs(M[i, j]) > lim * .45:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(M[i, j]) > lim * .75 else PAL["ink"])
    ax.set_title(f"Observable movement per ablation, {regime} "
                 f"(objective scale units)", color=PAL["ink"], fontsize=10,
                 loc="left")
    fig.colorbar(im, ax=ax, shrink=.7, label="deviation change")
    fig.tight_layout()
    return _save(fig, "fig_term_attribution.png")


def fig_sensitivity(ti, npz):
    """Spearman rho per (parameter, response) with the significance threshold."""
    s = ti["sensitivity"]
    rho, names, resp = npz["sens_rho"], s["names"], s["responses"]
    keep = [i for i, r in enumerate(resp) if r not in ("loss", "moment", "shape")]
    M = rho[:, keep]
    lim = 1.0
    fig, ax = plt.subplots(figsize=(1.2 + .42 * len(keep), 1.2 + .34 * len(names)),
                           facecolor=PAL["page"])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels([resp[i] for i in keep], rotation=90, fontsize=7.5,
                       color=PAL["ink2"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    crit = s["crit"]
    for i in range(M.shape[0]):
        j = int(np.nanargmax(np.abs(M[i]))) if np.any(np.isfinite(M[i])) else None
        if j is not None and abs(M[i, j]) > crit:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       edgecolor=PAL["ink"], lw=1.6))
    ax.set_title(f"Spearman rho, LHS n={s['n_lhs']} with common random numbers "
                 f"(|rho| > {crit:.2f} significant; box = primary response)",
                 color=PAL["ink"], fontsize=9.5, loc="left")
    fig.colorbar(im, ax=ax, shrink=.7, label="rho")
    fig.tight_layout()
    return _save(fig, "fig_term_sensitivity.png")


def fig_stage1(val, cals):
    """Model vs highD on every scored observable, before and after stage 1.

    Prefers the validation record (`out/normal_validation.json`), because that is
    measured on **fresh seeds** and carries the previous fit and the uncalibrated
    default alongside the new one; the calibration record only has the fitted
    point, at its own fitting seeds.
    """
    keys = [k for k, _, w in hf.OBS_SPEC_NORMAL if w > 0]
    if val:
        names = [b for b in ("train:freeflow", "train:dense",
                             "holdout:freeflow", "all:congested")
                 if b in val["blocks"]]
        source, tgt_of = "validation (fresh seeds %s)" % val["seeds"], None
    else:
        cal = cals[-1]
        names = cal["regimes"]
        source = "fitting seeds %s" % cal["seeds"]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names) + 1.2, 5.4),
                             facecolor=PAL["page"], sharey=True)
    axes = np.atleast_1d(axes)
    SERIES = (("fitted", CAT["blue"], "o"), ("previous", PAL["muted"], "s"),
              ("default", CAT["yellow"], "^"))
    for ax, name in zip(axes, names):
        if val:
            row = val["blocks"][name]
            blk = hf.target_block(name)
            tgt = hf.targets_of_spec(blk, hf.OBS_SPEC_NORMAL)
            sd = hf.spread_of_spec(blk, hf.OBS_SPEC_NORMAL)
            series = [(lbl, c, m, row[lbl]["observables"])
                      for lbl, c, m in SERIES if lbl in row]
            title = (f"{name}\nmoment {row['fitted']['moment']:.3f} "
                     f"(prev {row['previous']['moment']:.3f}), "
                     f"bar {row.get('bar') or float('nan'):.3f}")
        else:
            tgt, sd = cal["targets"][name], cal["target_sd"][name]
            series = [("fitted", CAT["blue"], "o", cal["observables"][name])]
            title = f"{name}  (loss {cal['per_regime'][name]['total']:.3f})"
        y = np.arange(len(keys))
        ax.axvspan(-2, 2, color=CAT["aqua"], alpha=.10)
        ax.axvline(0, color=PAL["ink2"], lw=1.0)
        for off, (lbl, c, mk, obs) in enumerate(series):
            dev = [((obs.get(k, np.nan) - tgt[k]) / (sd.get(k) or np.nan))
                   if k in tgt else np.nan for k in keys]
            ax.scatter(np.clip(dev, -12, 12), y - 0.22 * (off - 1), s=34,
                       color=c, marker=mk, zorder=4 - off, label=lbl)
        _ax(ax, title,
            "deviation from highD (multiples of its\nbetween-carriageway s.d.)")
        ax.set_yticks(y)
        ax.set_yticklabels(keys, fontsize=8)
        ax.set_xlim(-12.5, 12.5)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Stage 1: ordinary traffic against highD, no emergency vehicle "
                 f"in the scenario — {source}; shaded band = within 2 s.d.",
                 color=PAL["ink"], fontsize=10.5, x=.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .93))
    return _save(fig, "fig_term_stage1.png")


def fig_stage2(evc):
    """Stage 2: the four repulsion parameters against the dashcam ground truth."""
    from emv import groundtruth as gt
    fp_keys = [k for k, _, w in gt._DIST_SPEC if w > 0]
    gtfp = evc["gt"]["rules"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), facecolor=PAL["page"])
    ax = axes[0]
    x = np.arange(len(fp_keys))
    for off, (lbl, src, c) in enumerate((
            ("inherited", evc["baselines"]["inherited"]["fingerprint"], PAL["muted"]),
            ("fitted", evc["fingerprint"], CAT["blue"]),
            ("dashcam GT", gtfp, CAT["aqua"]))):
        ax.bar(x + (off - 1) * .27, [src.get(k, np.nan) for k in fp_keys], .26,
               color=c, label=lbl)
    _ax(ax, "Behaviour fingerprint vs the dashcam ground truth")
    ax.set_xticks(x)
    ax.set_xticklabels(fp_keys, rotation=90, fontsize=7.5)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    names = ["A_ev", "B_ev", "A_c", "B_c"]
    inh = [evc["baselines"]["inherited"]["theta"][n] for n in names]
    fit = [evc["theta"][n] for n in names]
    y = np.arange(len(names))
    ax.barh(y - .19, np.array(fit) / np.array(inh), .34, color=CAT["blue"],
            label="fitted / inherited")
    ax.axvline(1.0, color=PAL["ink2"], lw=1.0)
    for i, (a, b) in enumerate(zip(inh, fit)):
        ax.text(b / a + .02, i - .19, f"{a:.2f} -> {b:.2f}", fontsize=7.5,
                va="center", color=PAL["ink2"])
    _ax(ax, "The four fitted parameters (everything else frozen)",
        "ratio to the inherited value")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig_term_stage2.png")


# ------------------------------------------------------------------- tables
def table_validation(val) -> list:
    """Out-of-sample: fitted vs previous vs uncalibrated, against the bar."""
    L = ["", "## 6b. Stage-1 out-of-sample validation", "",
         f"Fresh seeds {val['seeds']}, never used in any fit. The bar is highD's "
         "own leave-one-carriageway-out distance under this objective - a moment "
         "distance, so compare it with the **moment** column, not with a total "
         "that includes the shape term. `congested` was **not fitted**: it is the "
         "block the 2026-07-29 free-flow-only fit left worse than uncalibrated.",
         "", "| block | bar | fitted moment | previous | uncalibrated | fitted "
         "shape | within 2 s.d. | collisions |",
         "|---|---|---|---|---|---|---|---|"]
    for name, row in val["blocks"].items():
        bar = row.get("bar")
        L.append(
            f"| `{name}` | {bar:.4f} | **{row['fitted']['moment']:.4f}** | "
            f"{row['previous']['moment']:.4f} | {row['default']['moment']:.4f} | "
            f"{row['fitted']['shape']:.4f} | {row['fitted']['within_2sd']} | "
            f"{row['fitted']['collisions']} |"
            if bar is not None else
            f"| `{name}` | - | **{row['fitted']['moment']:.4f}** | "
            f"{row['previous']['moment']:.4f} | {row['default']['moment']:.4f} | "
            f"{row['fitted']['shape']:.4f} | {row['fitted']['within_2sd']} | "
            f"{row['fitted']['collisions']} |")
    L += ["", "| regime | carriageways | bar (mean) | bar (median) |",
          "|---|---|---|---|"]
    for r, b in val["bars"].items():
        L.append(f"| {r} | {b['n_cells']} | {b['mean']:.4f} | {b['med']:.4f} |")
    return L


def tables(ti, cals, evc, ev_abl, val=None):
    L = ["# Two-stage framework extension - tables",
         "", "Generated by `experiments/make_term_figs.py` from the stored "
         "records; do not edit by hand.", ""]

    L += ["## 1. Term registry", "",
          "| term | axis | stage | what it is |", "|---|---|---|---|"]
    for n, (key, axis, block, desc) in terms.TERMS.items():
        L.append(f"| `{n}` | {axis} | {block} | {desc} |")

    if ti and "budget" in ti:
        L += ["", "## 2. Force budget (share of the summed |acceleration|)", "",
              "Effort, not importance: a term can dominate the budget and change "
              "no outcome. Read next to table 3.", ""]
        scen = list(ti["budget"])
        order = [t for t in terms.TERMS if t != "idm"]
        L += ["| scenario | axis | " + " | ".join(f"`{t}`" for t in order)
              + " | total \\|a\\| | IDM binds | braking it adds |",
              "|---" * (len(order) + 5) + "|"]
        for s in scen:
            for axis, key, mag in (
                    ("lateral", "share_y", "mean_abs_y"),
                    ("longitudinal (interaction only)", "share_x_interaction",
                     None)):
                sh = ti["budget"][s].get(key, {})
                d = ti["budget"][s].get("idm", {})
                tot = (sum(ti["budget"][s].get(mag, {}).values()) if mag
                       else d.get("mean_abs_social", float("nan")))
                L.append(f"| {s} | {axis} | "
                         + " | ".join(f"{sh.get(t, 0)*100:.0f} %" for t in order)
                         + f" | {tot:.3f} m/s2"
                         + (f" | {d.get('binding_share', 0)*100:.0f} % | "
                            f"{d.get('added_braking_when_binding', 0):.2f} m/s2 |"
                            if axis.startswith("long") else " | - | - |"))
        L += ["", "The raw longitudinal budget is ~99 % `drive` at motorway "
              "speeds, so the interaction terms are shown relative to each other. "
              "`idm` has no share because it is a *bound*, not a summand. Its "
              "influence is the two right-hand columns: how often it is the "
              "binding constraint, and how much extra braking it imposes when it "
              "binds (`social - applied`, measured after the same `-b_emerg` clip "
              "the integrator applies - the raw IDM demand is unbounded and "
              "reporting it would overstate what was integrated).", ""]

    if ti and "ablation" in ti:
        L += ["", "## 3. Leave-one-term-out (stage 1, ordinary traffic vs highD)",
              "", f"Baseline objective **{ti['ablation']['_baseline']['loss']}**. "
              "Positive = the model is worse without the term.", "",
              "| ablation | term | d objective | what it tests |",
              "|---|---|---|---|"]
        rows = sorted(((k, v) for k, v in ti["ablation"].items()
                       if not k.startswith("_")),
                      key=lambda kv: -kv[1]["d_loss"])
        for k, v in rows:
            L.append(f"| `{k}` | `{v['term']}` | **{v['d_loss']:+.3f}** | "
                     f"{v['why']} |")

    if ev_abl:
        L += ["", "## 4. Leave-one-term-out (stage 2, EV present vs dashcam)", "",
              f"Baseline objective **{ev_abl['baseline']['loss']}** at "
              f"`{ev_abl.get('base_label', '?')}`. **This objective is not the "
              "stage-1 one** - it is the dashcam behaviour-fingerprint distance "
              "plus safety guards, so these numbers rank terms among themselves "
              "and must not be compared with table 3's magnitudes.", "",
              "| ablation | d objective | fingerprint | collisions | EV speed "
              "ratio | clearance (m) |", "|---|---|---|---|---|---|"]
        for k, v in sorted(ev_abl["ablation"].items(),
                           key=lambda kv: -kv[1]["d_loss"]):
            m = v["measures"]
            L.append(f"| `{k}` | **{v['d_loss']:+.3f}** | "
                     f"{v['parts']['fingerprint']:.3f} | {m['collisions']:.0f} | "
                     f"{m['ev_speed_ratio']:.3f} | {m['clearance_mean']:.1f} |")

    if ti and "sensitivity" in ti:
        s = ti["sensitivity"]
        L += ["", "## 5. Identifiability: primary response per parameter", "",
              f"LHS n={s['n_lhs']}, common random numbers, "
              f"|rho| > {s['crit']:.3f} significant.", "",
              "| parameter | primary response | rho | significant |",
              "|---|---|---|---|"]
        for n, d in s["primary"].items():
            if d:
                L.append(f"| `{n}` | {d['response']} | {d['rho']:+.3f} | "
                         f"{'yes' if d['significant'] else 'no'} |")

    for cal in cals:
        L += ["", f"## 6. Stage-1 fit, block `{cal['block']}`", "",
              f"Objective **{cal['loss']}** against "
              f"{cal['baselines']['start']['loss']} at the starting point and "
              f"{cal['baselines']['highd_2026_07_29']['loss']} for the "
              f"2026-07-29 fit, on {cal['regimes']}, seeds {cal['seeds']}, "
              f"EV absent.", "",
              "| parameter | fitted |", "|---|---|"]
        for k, v in cal["theta"].items():
            L.append(f"| `{k}` | {v} |")
        r0 = cal["regimes"][0]
        L += ["", "| observable | weight | "
              + " | ".join(f"{r} highD | {r} model | dev/sd" for r in cal["regimes"])
              + " |", "|---" * (2 + 3 * len(cal["regimes"])) + "|"]
        for key, scale, w in hf.OBS_SPEC_NORMAL:
            if key not in cal["targets"][r0]:
                continue
            cells = []
            for r in cal["regimes"]:
                t = cal["targets"][r][key]
                sd = cal["target_sd"][r].get(key) or float("nan")
                m = cal["observables"][r].get(key, float("nan"))
                cells += [f"{t:.3f}", f"{m:.3f}", f"{abs(m-t)/sd:.2f}"]
            L.append(f"| `{key}` | {w:.1f} | " + " | ".join(cells) + " |")

    if val:
        L += table_validation(val)

    if evc:
        L += ["", "## 7. Stage-2 fit (four parameters, everything else frozen)",
              "", f"Objective **{evc['loss']}** against "
              f"{evc['baselines']['inherited']['loss']} for the inherited "
              f"values; out-of-sample on fresh seeds {evc['val_seeds']}: "
              f"**{evc['validation']['loss']}**. Frozen normal block: "
              f"`{evc['base_label']}`.", "",
              "| parameter | inherited | fitted |", "|---|---|---|"]
        for n in ("A_ev", "B_ev", "A_c", "B_c"):
            L.append(f"| `{n}` | {evc['baselines']['inherited']['theta'][n]} | "
                     f"**{evc['theta'][n]}** |")

    path = os.path.join(OUT, "term_tables.md")
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--influence", default="term_influence.json",
                    help="budget/ablation record to read from out/")
    ap.add_argument("--sensitivity", default="term_influence_sens.json")
    args = ap.parse_args()
    want = lambda n: args.only is None or n in args.only

    ti = _load(args.influence, required=False)
    sens = _load(args.sensitivity, required=False)
    npz = _load(args.sensitivity.replace(".json", ".npz"), required=False)
    evc = _load("ev_calibration.json", required=False)
    ev_abl = _load("ev_ablation.json", required=False)
    val = _load("normal_validation.json", required=False)
    cals = [c for c in (_load(f"normal_calibration_{b}.json", required=False)
                        for b in ("lon", "lat", "lc", "joint")) if c]
    if ti and sens and "sensitivity" not in ti:
        ti["sensitivity"] = sens["sensitivity"]

    outs = []
    if ti and want("budget") and "budget" in ti:
        outs.append(fig_budget(ti))
    if ti and want("ablation") and "ablation" in ti:
        outs.append(fig_ablation(ti, ev_abl))
        outs.append(fig_attribution(ti))
    if sens and npz is not None and want("sensitivity"):
        outs.append(fig_sensitivity(sens, npz))
    if (cals or val) and want("stage1"):
        outs.append(fig_stage1(val, cals))
    if evc and want("stage2"):
        outs.append(fig_stage2(evc))
    outs.append(tables(ti, cals, evc, ev_abl, val))
    outs = [o for o in outs if o]
    log_run("term_figs", metrics=dict(n_outputs=len(outs)),
            outputs=[os.path.relpath(o, ROOT) for o in outs],
            note="term-influence figures + tables from stored records")
    print(f"\n{len(outs)} outputs")


if __name__ == "__main__":
    main()
