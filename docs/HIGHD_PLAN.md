# highD real-trajectory calibration & realism validation

Repo: `/home/cubos/physics_behaviour_EMV_final/physics_behaviour_EMV` — the **active** copy
(strict superset of `/home/cubos/physics_behaviour_EMV`, updated today). All paths below are
relative to it.

## Context

`paper/ieee_emv_yielding.tex:420` states the model's own biggest limitation: *"No empirical
trajectory validation. Calibration targets are…"* distilled from literature. `emv/calibrate.py`'s
docstring says the same: *"No public trajectory dataset exists for fitting…"*. So every realism
claim in `PROJECT_LOG.md` §5 rests on **cited** anchors (NGSIM 4.01±2.31 s, "highD 0.124
LC/veh-km", "naturalistic 99th pct 2.85 m/s²") — and §10 records that one of those had to be
dropped mid-review because it could not be verified in the source.

You have the real dataset: `~/highD-dataset-v1.0.zip`, 60 recordings of German Autobahn traffic
at 25 Hz, 110,516 vehicles, 44,476 veh-km, with per-frame lateral position, lane markings,
THW/TTC. This turns the anchors from cited into **measured**, fits the model's manoeuvre
kinematics to real trajectories, and validates them out-of-sample.

Decisions taken: **highD only** (not inD); **add `TUNED_HIGHD` alongside `TUNED`** so
`tuned_params()` stays byte-identical and all four papers stay reproducible; **draft paper 5**.

### What highD can and cannot validate — say this in every deliverable

highD contains **no emergency vehicles**. It validates:

- ✅ how real drivers *execute* a lateral manoeuvre — duration, lateral speed/accel envelope,
  lane-keeping precision: the kinematics every yielding manoeuvre is built from;
- ✅ the traffic the EV drives through — per-lane speed distributions, headways, TTC floor,
  truck share, geometry;
- ❌ the EV-response *trigger* — yield onset, compliance rate, corridor strength. That stays the
  dashcam GT's job. Hence a **two-stage** calibration, not one.

**Validation framing — the two datasets bracket the model.** highD gives *discretionary* lane-change
kinematics (peak |v_lat| p90 = 1.13 m/s); the dashcam GT gives *EV-induced* yielding (lat speed p90
0.50 reviewed / 1.01 rules-only). Real yielding is, if anything, gentler than a discretionary lane
change. A realistic model must sit inside that bracket — a far stronger claim than hitting one number.

## Evidence already measured this session (read-only prototype)

highD recording 01, two-lane-occupancy rule (the *same* definition as
`emv/metrics.py:lane_change_events`, line 176-184):

| observable | highD | model now (`out/param_stats.json`) | governing param |
|---|---|---|---|
| LC duration | 2.51 ± 1.55 s (med 2.28, p90 3.92) | free-flow **0.93** / jam 4.19 | `a_lat_max`, `v_lat_max`, `a_pin` |
| peak \|v_lat\| per LC | med 0.76, p90 1.13, max 1.79 m/s | free-flow **1.24** / jam 0.41 | `v_lat_max` = **2.2** |
| peak \|a_lat\| per LC | med 0.18, p90 0.35, **p99 0.55** m/s² | free-flow 1.14 / jam 0.79 | `a_lat_max` = **3.0** |
| lane-keeping \|offset\| | mean 0.26, p90 0.46 m | *no model metric yet* | `a_pin`, `zeta_lat` |
| per-track peak decel | p99 2.04, max 2.95 m/s² | free-flow 2.58 / jam 0.80 | `b_comf` = 3.0 |
| THW | med 1.62, p15 0.79, p85 4.03 s | `T_hw ~ U(1.1, 1.8)` | IDM `T_hw` |
| TTC | p1 4.60, p5 7.42 s | min TTC free-flow 4.6 | safety envelope |

**The mechanism is already identified.** `a_lat_max=3.0` and `v_lat_max=2.2`
(`emv/params.py:43-44`, clamped in `emv/dynamics.py:29,35`) are ~5× and ~2× the measured
envelope, so they **never bind** — which is *why* free-flow lane changes take 0.93 s. This is
exactly the "lateral bound analogous to the IDM one" that paper 4 argues for on theoretical
grounds; highD supplies the numbers. Predicted (before fitting): binding these two clamps at
highD levels lengthens `lc_dur_med` toward ~2.5 s and drops `peak_alat_*` by ~2-3×, at some cost
in EV progress — the safety/performance trade-off paper 4 identified. That prediction is
falsifiable and step 3 tests it.

Also measured:

- **Full-release LC rate = 0.314 LC/veh-km** (13,949 `numLaneChanges` / 44,476 veh-km) — not the
  0.124 (=5600/45000, arXiv:1810.05642) the repo hard-codes in five places. Report as a
  definitional discrepancy, do not silently swap.
- **Lane width**: measured median 3.89 m (per location 3.56-3.96); model `Road.lane_width` = 3.5.
  Lane-change duration scales with the width traversed, so geometry must be matched before
  comparing.
- **Per-lane speeds are nothing like the model's.** rec 08 (loc 4, free-flow), right→left:
  25.2 / 34.1 / 39.8 m/s, within-lane σ 3.8/4.2/4.9, truck share 64 % / 9 % / 1 %. Model
  `make_overtake` uses `lane_speeds=[24.5, 27.5, 30.0]` with σ=1.2 and **no trucks**
  (`emv/scenarios.py:70,28`). Real left-lane traffic (≈40 m/s) is *faster than the model's EV*
  (`v0=36`) — a substantive realism finding.
- **Regime coverage** (per carriageway, 60 s windows — one carriageway can be jammed while the
  other flows, so windowing per direction is required): free-flow at loc 4 (rec 07-10, ~34.5 m/s);
  density twins of the model's 1700 veh/h/lane SUMO mirror at rec 11 (1744) and rec 12 (1784);
  genuine congestion at rec 25 dir 1 (per-lane medians 7.3/9.8/10.7 m/s, 37 % of frames < 10 m/s,
  windows down to 6 m/s) and rec 26 dir 1 (early windows 11-16 m/s). 44 of 60 recordings have
  3 lanes/direction, matching `Road.n_lanes=3`.
- **The model's severe `jam` (2.5 m/s) is outside highD's coverage** — highD never fully stops.
  So a new highD-matched `congested` scenario (7-11 m/s) is the validated one; `make_jam` stays,
  labelled an extrapolation.
- Loading one `tracks.csv` from inside the zip with pandas = 0.7 s → all 60 ≈ 60 s. **Never
  extract** (4.78 GB uncompressed). 25 Hz decimated by 3 = **0.12 s exactly** = the model's
  `rec_dt`, so sampling is matched with no interpolation.
- The model imports and runs on this Linux box (py 3.14.4 + numpy 2.5) — verified.

## Approach

### Stage 0 — environment (this box differs from `PROJECT_LOG.md` §2)

System `python3` has **no numpy**. Create a dedicated venv and record it in `PROJECT_LOG.md` §2 as
a second environment block:

```
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib pillow pandas
```

`emv/` stays **numpy-only** so the Windows box keeps working; pandas is confined to the highD
reader's fast path with a `csv`-module fallback, and enters `requirements.txt` only as a commented
optional extra. Not available here and deferred: **SUMO** (no binary → bridge/survey benchmark
cannot be re-run with the new parameters) and **LaTeX** (no texlive, `sudo` needs an interactive
password → paper 5 ships as `.tex` + figures, compiled on the Windows box; you can install it here
yourself with `! sudo apt-get install -y texlive-latex-extra texlive-fonts-recommended` if you
prefer).

### 1. `emv/highd.py` — dataset reader + observables (new, mirrors `emv/groundtruth.py`'s role)

One flat module, matching repo idiom (`groundtruth.py` is loader + annotator + fingerprint in one
file). Public API:

- `HIGHD_ZIP` default `~/highD-dataset-v1.0.zip`, overridable by `$EMV_HIGHD_ZIP` / `--zip`.
- `read_recording(ds, rid)` → meta / tracksMeta / tracks. pandas fast path, `csv` fallback.
  Applies the verified conventions: lateral centre = `y + height/2`, `width` = length,
  `height` = lateral width.
- `carriageway(rec, direction)` → **highD put into the model's road frame**: x increasing in
  travel direction (mirror `drivingDirection==1`), lane 0 = rightmost, lateral centre measured
  from the measured markings, decimated 25 Hz → 0.12 s.
- `lane_change_events_highd(cw, persist=0.5, settle=0.6)` → the same event dicts as
  `metrics.lane_change_events`, same two-lane-occupancy `duration`, plus a completed-LC filter
  (lane index must actually change — recording 01 has 225 straddle episodes vs 139 completed
  changes; report both).
- `behaviour_stats_highd(cw)` → the **same keys** as `metrics.behaviour_stats` for every EV-free
  observable, so model and data are literally the same computation.
- `window_stats(rec, win=60.0)` → per-(direction, window) regime record + observables.
- `recording_index(ds)` → meta-only table (loc, lanes, lane widths, fps, duration, flow/lane,
  veh-km, speeds, truck share, LC count) — ~3 s for all 60.
- `TARGET_SPEC` — the `(key, scale, weight)` table for the loss, in the style of
  `groundtruth._DIST_SPEC:331`.

### 2. Small additive model changes (nothing existing changes behaviour)

| file | change | safety |
|---|---|---|
| `emv/metrics.py` | add `lane_offset_mean` / `lane_offset_p90` to `behaviour_stats` (non-EV, non-straddling frames) — there is no lane-keeping metric today | additive keys only |
| `emv/scenarios.py` | `_spawn_traffic` gains optional `v0_sd=1.2` and `classes=None` (defaults = today's behaviour exactly); new `make_highd_like(...)` factory built from the same `_spawn_traffic`/`_assemble` helpers | new code path |
| `emv/params.py` | append `TUNED_HIGHD` + `highd_params()` with a provenance comment block mirroring `TUNED:124-137`. **No `Params` default changes.** | `tuned_params()` byte-identical |
| `emv/empirical.py` | new tiny accessor: measured constants from `out/highd_targets.json`, falling back to a checked-in `emv/empirical_fallback.json` so figure scripts work without the dataset | replaces 5 duplicated hard-codings |

`make_highd_like` sets, per reference set, everything measured rather than fitted: `lane_width`
(measured), `n_lanes=3`, per-lane desired speeds and within-lane σ, per-lane truck fraction with
truck `L≈16 m, W≈2.5 m, v0≈22-25 m/s` (`VehState` is already fully heterogeneous — `L, W, v0,
tau, T_hw, s0, amax, bcomf` per vehicle, `emv/state.py:16-27`), flow → spacing, and the `T_hw`
distribution from measured THW.

### 3. `experiments/extract_highd_stats.py` → measured targets (~4 min)

Writes `out/highd_index.json` (60 rows), `out/highd_windows.npz` (per-window observables),
`out/highd_samples.npz` (pooled raw samples for distributional distances, subsampled < 10 MB),
and `out/highd_targets.json` — the regime-binned pooled targets with bootstrap CIs (reuse
`run_param_stats.py:boot_ci`), between-window spread, and the train/holdout split. Logged via
`RunLogger("highd_stats")` with headline metrics first (`logs/LATEST.md` shows only 4).

**Regime bins** (per carriageway-window): `freeflow` = median ≥ 28 m/s and ≤ 1100 veh/h/lane;
`dense` = 1400-1800 veh/h/lane (the SUMO-mirror twin); `congested` = median ≤ 15 m/s.
**Split**: leave-locations-out — train on loc 1, 2, 5 (50 recordings), hold out loc 3, 4, 6
(10 recordings). Congestion exists only at loc 1, so the congested set splits by recording
(train 25, holdout 26). Both reported.

### 4. Two-stage calibration

**What is SET vs FITTED** — the partition is the scientific core:

- **SET from highD, not fitted**: lane geometry, per-lane desired-speed distributions, truck
  share and dimensions, flow/spacing, `T_hw` distribution, `b_comf` ceiling from measured p99 decel.
- **FITTED, stage 1** (`experiments/run_highd_calibration.py`, LHS 96 + Nelder-Mead 60, 3 seeds):
  `a_lat_max` ∈ [0.4, 2.0], `v_lat_max` ∈ [0.6, 2.0], `a_pin` ∈ [0.3, 3.0], `zeta_lat` ∈ [0.5, 2.0].
  Reuses `calibrate.latin_hypercube` / `calibrate.nelder_mead` via the in-place `THETA_SPEC`
  monkeypatch precedent at `run_surrogate_sumo.py:105-108` — `calibrate.py` itself is untouched.
- **NOT identifiable from highD** (no EV present): `A_ev, B_ev, A_c, B_c, T_react, gamma_c,
  R_front, R_rear, p_noncomply, delay_*` → **stage 2**: re-fit the existing 6-param EV block on
  the highD-grounded base against the dashcam GT (available locally,
  `--gt-dir /home/cubos/emergency-vehicle-dataset-pipeline/output`, 33,147 frames) using the
  standalone fingerprint path already built for the surrogate fit — verified present:
  `run_surrogate_sumo.py:121 ego_stream(h)` builds the 1 Hz ego-frame stream from a plain
  `History` exactly like the bridge recorder, then `gt.annotate_stream` → `gt.fingerprint` →
  `gt.fingerprint_distance`, so **no SUMO is needed for stage 2**. Lift `ego_stream` into
  `emv/groundtruth.py` as a shared helper and import it back (the same behaviour-safe move session 5
  did with `make_sumo_like`). Stage 2 is required because stage 1 changes the lateral dynamics the
  EV response rides on.

`TUNED_HIGHD` = stage 1 ∪ stage 2.

**Stage-1 loss** — squared normalised deviations, regime-matched and seed-averaged, in
`calibrate.loss`'s style: `lc_dur_med` (scale 0.5 s, w 2.0), `lc_dur_p90` (1.0, 1.0),
`lc_peak_vy_med` (0.15 m/s, 2.0), `lat_speed_p90` (0.25, 1.0), `peak_alat_med` (0.10 m/s², 1.5),
`alat_p99` (0.20, 1.0), `peak_decel_p90` (0.4, 1.0), `lane_offset_mean` (0.10 m, 1.0),
`bg_med_speed` (1.0 m/s, 1.0), one-sided `min_ttc ≥` measured p1 (1.0, 1.0), collision penalty as
in `calibrate.loss:65`, plus a scipy-free 1-Wasserstein distributional term (mean |quantile
difference| on a common grid) on LC duration and peak |v_lat|, w 1.0 each.

**One target is deliberately excluded: `lc_per_veh_km`.** highD's 0.314 LC/veh-km is almost
entirely *discretionary*, and the model has **no discretionary lane-change mechanism** — the
`no_yield` control produces 0.00 LC/veh-km. It is structurally unmatchable, so it is reported as
a scope limit with a MOBIL-style discretionary LC named as future work, **not** fitted around.
Correspondingly, stage 1 fits *conditional* kinematics (given a lane change happens), which in the
model are EV-induced and in highD discretionary — an assumption stated explicitly and checked by
the bracket against the dashcam GT.

### 5. `experiments/run_highd_validation.py` — holdout + correlation (~30-45 min)

- Out-of-sample: `TUNED_HIGHD` evaluated against the **holdout** locations, 8 seeds, bootstrap CI
  both sides; acceptance bar = highD's own between-recording/between-window spread (the same logic
  that made SUMO's 0.213 self-distance the bar for the surrogate); paired permutation test
  (`run_survey_stats.py:197`) for TUNED vs TUNED_HIGHD; per-observable pass/fail.
- **The correlation study you asked for**: LHS N=128 × 2 common seeds over the four stage-1
  parameters, Spearman ρ + SRC with per-metric linear-surrogate R² (reuse
  `run_param_stats.py:spearman/src`, |ρ|>0.17 significance bar) against **each highD-matching
  error** → a parameter × observable identifiability matrix.
- Trade-off table: EV speed ratio / min TTC / collisions / disruption at `TUNED` vs `TUNED_HIGHD`,
  so the realism gain is reported next to its cost.
- Re-verify the analytic depinning prediction `d* = w_need + B_c·ln(A_c/a_pin_eff)` at the new
  `a_pin` (§3 item 5 of `PROJECT_LOG.md`).

### 6. `experiments/make_highd_figs.py` → 6 figures + `out/highd_tables.md`

`fig_highd_regimes` (60 recordings/windows in flow×speed space with the model's scenarios
overlaid — shows coverage *and* the jam extrapolation) · `fig_highd_observables` (model TUNED vs
TUNED_HIGHD per regime against highD bands with CIs, dashcam GT overlaid = the bracket figure) ·
`fig_highd_cdf` (empirical CDF overlays: LC duration, peak |v_lat|, |a_lat|, THW, before/after) ·
`fig_highd_correlation` (ρ heatmap) · `fig_highd_holdout` (train vs holdout dot plot) ·
`fig_highd_tradeoff` (realism error vs EV performance). Uses `emv/viz.py`'s palette/`_save` idiom.

### 7. Retire the duplicated empirical constants

Migrate all five sites to `emv/empirical.py`: the canonical `LIT` dict
(`make_param_stats_figs.py:47-66`), its independent copy (`make_param_paper_extra.py:175-180`,
which already disagrees — 3.4 vs 2.85 m/s² for decel), the HTML strings in
`artifact/make_param_memo.py:141-156` (+ prose at :218,:315,:367,:385), paper 4's Table I
(`ieee_emv_param_sensitivity.tex:547-548`) and `PROJECT_LOG.md`/`README.md:86`. NGSIM stays as a
**secondary cited** reference (different dataset: congested US freeway) rather than being deleted,
and the 0.124-vs-0.314 discrepancy is presented as a definitional note, not a correction.
`groundtruth._DIST_SPEC` is left alone (two figure scripts import it directly).

### 8. Documentation

`docs/HIGHD_VALIDATION.md` (dataset, frame transform, observable definitions side-by-side with
`metrics.py`, regime bins, split, loss, results, limitations) · `PROJECT_LOG.md` §1 three new rows,
§2 a Linux-environment block (it currently describes only Windows and is wrong for this box), §4
`TUNED_HIGHD`, §5 results + the anchor correction, §6 file map, §7 commands, §12 backlog
(discretionary LC, truck class, SUMO re-run), §13 dated session entry · **`info/` folder created
with `info/2026-07-28_highd_calibration_validation.txt`** per your standing convention (this repo
has none yet) · `README.md`, `requirements.txt` · `docs/FRAMEWORK.md` at its existing anchors:
§2.8 "Constraints and integration" (the lateral envelope becomes empirically bounded), §4
"Calibration (gap 3)" (add the two-stage framing + a stage-1 bounds/fitted table beside the
existing one at line 253), §6 "Limitations and extensions" (retire the no-empirical-validation
limitation, add the honest ones: no discretionary LC, jam extrapolated, one country).

### 9. Paper 5 — `paper/ieee_emv_highd_validation.tex`

"Calibrating and Validating a Physics-Inspired Emergency-Vehicle Yielding Model on Real Motorway
Trajectories". IEEEtran conference, the 7-section structure you asked for on papers 3/4
(Abstract / Intro / Related Work / Methodology / Implementation / Results and Analysis /
Conclusions), themed functional safety / smart mobility / autonomy, target 7 pages. Column-width
figures via `make_highd_paper_figs.py` → `paper/figs/hd_*.png`. Tables: observable commensurability
· SET vs FITTED with bounds · measured highD targets per regime with CIs · model vs highD
train/holdout · TUNED vs TUNED_HIGHD trade-off. Reuses `thiemann08`, `krajewski18`, IDM/SFM refs;
every new reference web-verified per §10's reference-hygiene rule. Limitations paragraph: no EV in
highD (hence the bracket), no discretionary LC, jam regime extrapolated, one country/road type.
**Not compiled here** — page-fitting is a Windows-box step.

## Execution order

| # | step | compute | checkpoint |
|---|---|---|---|
| 0 | venv bootstrap | 3 min | `emv` imports, smoke passes |
| 1 | `emv/highd.py` + `extract_highd_stats.py`, run all 60 | ~4 min | **measured targets table for review** |
| 2 | additive model changes (§2) + `make_highd_like` | — | `tests/smoke.py` + `test_ue_bridge.py` unchanged |
| 3 | stage-1 calibration | 25-40 min | fitted lateral block; prediction confirmed or not |
| 4 | stage-2 EV-block re-fit vs dashcam GT | 30-45 min | **`TUNED_HIGHD` + trade-off numbers for review** |
| 5 | validation + correlation study | 30-45 min | holdout pass/fail, ρ matrix |
| 6 | figures + tables | 2 min | 6 figures |
| 7 | empirical-constant migration | — | old figures regenerate identically except the cited→measured column |
| 8 | docs + `info/` + PROJECT_LOG | — | — |
| 9 | paper 5 `.tex` + paper figures | — | ships uncompiled |

Nothing destructive: additive files, additive `Params` fields, new scenario factory. `TUNED`,
`Params()` defaults and every `out/*.json` of record stay untouched. No git operations.

## Verification

1. **Reproducibility of the existing record** — new `tests/test_highd.py` asserts `TUNED` dict
   equality, `Params().a_lat_max == 3.0` / `v_lat_max == 2.2`, that `behaviour_stats` still returns
   every pre-existing key, and a **bit-identical** seed-11 overtake `evaluate()` comparison against
   the stored reference (the same technique session 4 used to clear the `perception.py` change).
   `tests/smoke.py` and `tests/test_ue_bridge.py` run before and after every step. Note
   `tests/smoke.py:28-30` asserts the EV gains >1 m/s from yielding and takes 0 collisions **with
   `Params()` defaults** — since defaults are untouched, it stays insensitive to `TUNED_HIGHD`;
   the equivalent assertions for the new preset go in `tests/test_highd.py`.
2. **Definitional equivalence** — a self-check asserting `behaviour_stats_highd` and
   `metrics.behaviour_stats` agree to ~1e-9 when fed the *same* synthetic trajectory set through
   both paths (the pattern `run_survey_stats.py:175 self_check()` already uses).
3. **Out-of-sample realism** — holdout-location observables inside highD's between-recording
   spread, per-observable pass/fail, paired permutation test vs `TUNED`.
4. **Falsifiable prediction** — `lc_dur_med` free-flow rises from 0.93 s toward the measured
   ~2.5 s and `peak_alat_*` falls ~2-3× once the clamps bind. If it does not, the structural fix is
   the rate-limited "sublane edging" mode already on the backlog — reported as such, **not** hidden
   by widening bounds.
5. **Honest negative reporting** — `lc_per_veh_km` (no discretionary LC) and the severe-jam
   extrapolation are stated as scope limits; any EV-performance loss is reported beside the realism
   gain; the SUMO-side re-validation is explicitly listed as deferred (no SUMO on this box).

## Open items to confirm during execution

- If stage 1 cannot reach highD kinematics without collapsing EV progress, that trade-off **is**
  the finding — report it, don't tune around it.
- Trucks are 48-65 % of highD's rightmost lane; the model is homogeneous cars. Included in
  `make_highd_like` only; flagged if it destabilises the force balance.
- `out/highd_samples.npz` size to be held under ~10 MB by quantile subsampling.
