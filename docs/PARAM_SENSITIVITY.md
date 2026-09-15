# Parameter sensitivity & performance of the two-stage SFM EV-yielding model

Study run 2026-09-07. Code `experiments/run_sfm_param_sensitivity.py`
(figures `experiments/make_sfm_sensitivity_figs.py`). Records
`out/sfm_param_sensitivity.{json,npz}`, run ledger `kind='param_sensitivity'`
(`wall_s≈4089`). Figures `paper/figs/sens_{tornado,curves,gsa,reference,surface}.png`.

**Base model:** `emv.params.bluelight_params()` (`TUNED_NORMAL` stage-1 host block
frozen + stage-2 EV block `A_ev,B_ev,A_c,B_c` refitted) — the calibration the
ITEC full paper describes.

**Evaluation kernel:** one `make_overtake` (density 22, 1500 m) + one `make_jam`
(520 m) run per parameter vector and seed — the pair `emv.calibrate` uses, so
numbers line up with the calibration record. Metrics from `emv.metrics.evaluate`
+ `behaviour_stats`: EV speed / desired, EV median speed, corridor clearance,
min TTC, yield-onset distance, traffic disruption, lane-changes/veh-km,
collisions.

**29 tunable parameters**, each over its calibration search bound
(`emv/highd_fit.py` `SPEC_LON/LAT/LC`, `emv/calibrate.py` `THETA_SPEC`) plus four
normally hand-set perception knobs (`p_noncomply`, `R_front`, `delay_med`,
`urgency_pin_relief`).

## Designs

| design | what | size |
|---|---|---|
| OAT   | one-at-a-time response curve per parameter, seeds kept separate for a bootstrap CI | 29 params × ~8 levels × 4 seeds |
| GSA   | Latin-hypercube over all 29 at once; Spearman ρ + standardized regression coeff.; linear-surrogate R² per metric; common random numbers | 180 samples × 2 seeds |
| SURFACE | `A_c × a_pin` grid vs the analytic escape boundary `A_c u > a_pin(1−ε u)` | 9×9 × 2 seeds |
| REF   | calibrated (blue-light) vs dataclass-default vs no-yielding control | 10 seeds |

## Findings

### 1. Free-flow yielding is robust; the jam is where parameters bite

OAT swing of EV speed / desired over each parameter's full range:

| regime | largest swings (Δ EV speed/desired) |
|---|---|
| free-flow overtake | `T_react` 0.11, `p_nc` 0.11, `a_pin` 0.08, `A_c` 0.07, `ε` 0.05, `k_ρ` 0.04, `ζ` 0.04 — **everything ≤ 0.11** |
| stop-and-go jam | `k_ρ` **0.72**, `a_pin` **0.59**, `ε` **0.55**, `p_nc` **0.50**, `B_x` 0.35, `A_c` 0.33, `A_v` 0.17, `τ` 0.15 |

In free flow the corridor opens with margin to spare (`sens_surface.png`, left:
the calibrated point sits deep in the ≥0.95 region), so no single parameter
moves the outcome much. In the jam the calibrated point sits on the 0.85
iso-progress contour, just above the analytic escape boundary, and the
lane-keeping / density group (`a_pin`, `k_ρ`, `ε`) plus non-compliance `p_nc`
govern whether the corridor forms at all.

### 2. The operative parameters are the depinning group, not the EV field

GSA Spearman ρ (linear-surrogate R² in brackets), |ρ| ≥ 0.30:

- **EV progress, jam** [R² 0.72]: `k_ρ` −0.53, `p_nc` −0.49, `a_pin` −0.32.
- **EV progress, overtake** [R² 0.63]: `T_react` +0.43, `a_pin` −0.38, `A_c` +0.30, `R_front` +0.28.
- **corridor clearance, overtake** [R² 0.71]: `T_react` +0.47, `a_pin` −0.40, `A_c` +0.33, `R_front` +0.28.
- **yield-onset distance, overtake** [R² 0.85]: `R_front` **+0.70**, `T_react` +0.38, `a_pin` −0.33.
- **traffic disruption, overtake** [R² 0.91]: `τ` +0.73, `a_max` +0.62 (soft/slow car-following disrupts the background more).
- **min TTC, overtake** [R² 0.37 — unreliable]: `τ` −0.28, `a_max` −0.26.
- **lane-changes/veh-km, overtake** [R² 0.27 — unreliable]: `a_pin` −0.48, `p_nc` −0.38.

`A_ev` and `B_ev` (the EV point field) clear |ρ| ≥ 0.30 on **no** headline
metric — consistent with the paper's per-term result that the corridor term, not
the field attached to the EV, carries the yielding response. `A_c`, `B_c` matter
but saturate quickly (`sens_curves.png`, `A_c` panel: flat above ≈4 m/s²).

### 3. Response shapes (`sens_curves.png`)

- **threshold / collapse:** `k_ρ`, `a_pin`, `B_x` — jam EV progress falls off a cliff past a critical value (depinning fails).
- **saturating:** `A_c`, `ε`, `A_ev` — monotone but flat beyond the calibrated value; extra push buys nothing.
- **monotone:** `p_nc` (−), `A_v` (−).
- **non-monotone:** `τ` — a floor near the calibrated 0.6 s, worse both below (jitter) and far above.
- **~inert on EV progress:** `T_hw_lo/hi`, `b`, `v_lat^max`, `σ_off`, `het_lat`, `A_pass`, `A_keep`, `T_frust`, `s_veto`, `γ_c` (11 of 29). The four discretionary-lane-change parameters do essentially nothing here because discretionary changes barely fire in a scenario dominated by EV-forced merges — as expected, and the reason stage 1 needed EV-free highD data to identify them.

### 4. Depinning theory holds (`sens_surface.png`)

Iso-progress contours run parallel to and above the analytic escape line
`A_c = a_pin(1−ε)`; progress collapses below it. Same picture as the earlier
`tuned_params()` study (`out/param_stats`), now on the two-stage base.

### 5. PERFORMANCE — and a regression in the harsh jam (`sens_reference.png`)

10-seed means (95 % bootstrap CI):

| preset | scenario | EV speed/desired | clearance (m) | min TTC (s) | collisions/run |
|---|---|---|---|---|---|
| no-yield control | overtake | 0.80 [0.76, 0.83] | 42 | 3.71 | **0.0** |
| no-yield control | jam | 0.22 [0.20, 0.24] | 5 | 1.04 | **0.0** |
| dataclass default | overtake | 0.99 [0.98, 1.00] | 90 | 4.75 | **0.0** |
| dataclass default | jam | 0.98 [0.97, 0.99] | 60 | 1.50 | **0.0** |
| **blue-light (two-stage)** | overtake | 0.95 [0.94, 0.96] | 64 | 3.62 | **0.4** |
| **blue-light (two-stage)** | jam | 0.71 [0.62, 0.77] | 9 | 1.04 | **9.3** |

- Yielding works: blue-light lifts EV progress from 0.80 → 0.95 (overtake) and
  0.22 → 0.71 (jam) over the no-yield control.
- **But the two-stage calibrated set is not collision-free in `make_jam`**:
  ≈10 contacts per run (3-seed spot-check `[5, 16, 9]`), vs **zero** for both the
  no-yield control and the uncalibrated dataclass defaults on the identical
  kernel and seeds. So this is parameter-driven, not a kernel artefact.
- It is the **stage-1 host block as a whole** (`TUNED_NORMAL`: soft
  `a_max ≈ 1.16` m/s², longer headways, tight lateral caps `a_lat^max ≈ 0.42`,
  `v_lat^max ≈ 1.05`, driver dispersion `het_lat ≈ 0.34`), **not any single
  knob** and not the EV block. Verified: restoring `a_max` to 2.6 makes it
  *worse* (16.7); restoring the lateral caps, `zeta_lat`, `het_lat` or
  `sigma_off` individually to their defaults each leaves 6–10 collisions. In a
  jam this dense, an EV forcing merges into traffic that both brakes weakly and
  moves sideways slowly produces a pile-up that no one parameter unwinds. OAT:
  best single moves are `ζ=3.0` → 1.2, `k_ρ=1.7` → 0, `A_ev=1.5` → 1.2; `τ`
  above 1.0 s makes it far worse (up to 106).
- This is the operational face of two results already on record: the two-stage
  fit's **worst block is the congested regime** (moment distance 2.48 vs bar
  0.49, `PROJECT_LOG` §5), and session 8 flagged **non-monotone collision counts**
  near the useful `a_commit` setting. The stage-2 EV validation that was
  "collision-free, min TTC 3.05 s" was on the recorded-passage scenarios, which
  are less dense than `make_jam`.

## Recommendation

`make_jam` is denser than anything stage 2 was fitted or validated on, so the
blue-light preset should be read as a **free-flow / moderate-density** model
until the host block is hardened for congestion. Concrete next steps, in order:

1. Re-fit `TUNED_NORMAL` with the congested highD regime **in** the objective
   (currently withheld), or add a jam scenario to the stage-1 loss, so the host
   block is not free to be this soft where it is never scored.
2. Ship `params.a_commit` (session-8 backlog) to decouple the depinning
   threshold from the restoring force, then re-check jam collisions across ≥10
   seeds.
3. Weight `collisions` (currently ~0) in the stage-2 objective, or add a hard
   feasibility gate: reject any parameter vector with >0 collisions on the jam
   pair.

Until then the paper's operational claims ("0 collisions / 40 runs", multi-seed
EV-speed gains) hold for the overtake / moderate-jam scenarios they were
measured on and should be scoped that way.
