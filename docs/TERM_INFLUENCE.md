# Two-stage calibration and per-term influence

**What this document is.** The method, results and limits of the framework
extension of 2026-08-06: calibrating ordinary motorway traffic against highD with
**no emergency vehicle in the scenario at all**, then extending to the blue-light
case by changing **only** the parameters that describe how other cars are
repelled by the emergency vehicle — and measuring what every force term
contributes along the way.

Reproduction commands are in §8. Numbers of record live in
`out/normal_calibration_*.json`, `out/ev_calibration.json`,
`out/term_influence*.json` and `out/term_tables.md`; this file explains them.

---

## 1. Why the previous calibration had to be redone

The 2026-07-29 highD fit (`PROJECT_LOG.md` §5, fit (c)) is described as a
*host-traffic* calibration. Two properties of how it was scored mean it was not
quite that.

**It was scored with a live emergency vehicle in the scenario.**
`highd_fit.model_observables` ran `make_highd_like` with the EV active, and the
model had no discretionary lane-change mechanism, so *every lane change it
measured was EV-induced*. Those events were then compared against highD's, which
are almost entirely discretionary. Measured this session, with the EV removed:

| scenario | lane changes | veh-km | rate (per veh-km) |
|---|---|---|---|
| model, EV active, free-flow | 7 | 134 | 0.052 |
| model, EV inert, free-flow | **0** | 206 | **0.000** |
| highD, complete manoeuvres, free-flow (44 carriageways) | — | — | **0.2758** |
| highD, complete manoeuvres, dense (27 carriageways) | — | — | 0.2026 |

(The highD rates are `lc_per_veh_km_complete` from `out/highd_targets.json`,
computed per carriageway and then aggregated, so they are *not* recoverable by
dividing the 8 975-event pool by the 44 476 veh-km total — that division gives
0.202 and would mix regimes. The objective scores each regime against its own
rate. All recorded events, rather than complete manoeuvres only, run 0.247–0.324.)

With no EV present every `lc_*` observable is undefined. So the lateral block was
identifiable *only because* an emergency vehicle was in the scenario, which makes
"we calibrated ordinary driving behaviour" circular.

**It was fitted on one density.** Fitted on free-flow alone, it drove `a_pin`
down 84 % (1.4 → 0.2236) and left the congested block **worse than no
calibration at all** (6.366 against a 3.304 uncalibrated baseline). Weak lane
discipline fits an isolated free-flow manoeuvre; in a jam, lane discipline is
what holds the structure together.

**Nothing longitudinal was scored.** `out/highd_targets.json` has carried
`thw_med/p15/p85`, `ttc_p01/p05`, `accel_p99`, `decel_p01` and `bg_p15_speed`
since the extraction, and `out/highd_pools.npz` has carried 200 000 real samples
of the headway and TTC distributions. None were used, because the model side had
no counterpart — so the car-following block had never been compared with real
spacing at all.

---

## 2. The two mechanisms this required

Both default to their neutral value, so every result in papers 1–6 reproduces
bit-identically (asserted in `tests/test_terms.py`).

### 2a. Discretionary lane changing (`emv/lanechange.py`, `forces.lane_incentive`)

A driver commits to a lane change when the incentive can lift them out of their
lane's potential well:

```
A_pass       * frust      * (1 - frust_left)   >  a_pin_eff     (overtake)
A_keep_right * rear       * (1 - frust_right)  >  a_pin_eff     (return right)

frust_i = clip((v0_i - v_lead_i)/v0_i, 0, 1) * clip((T_frust - thw_i)/T_frust, 0, 1)
rear_i  = the same quantity computed for the vehicle BEHIND i
a_pin_eff = a_pin * (rho/rho_ref)^k_rho * (1 - urgency_pin_relief * u)
```

That inequality is the **same depinning condition** the emergency-vehicle
corridor force must cross (`A_c·u > a_pin_eff`, `FRAMEWORK.md` §4). The framework
therefore carries one escape condition with two things that can drive it, and the
lane-change rate becomes predictable from the parameters rather than merely
observed.

Three design decisions were forced by measurement, and each is worth more than
the reasoning that preceded it:

1. **The decision is latched, not gated.** The first version applied the
   incentive only near the lane centre, expecting the washboard slope to carry
   the car the rest of the way. With the calibrated lateral block (`a_pin` 0.224,
   `zeta_lat` 2.42 — weak pinning, heavily overdamped) that residual slope moves
   a car sideways at ~0.17 m/s, so a manoeuvre took **24–38 s against a real
   2.28 s**. A committed driver steers continuously; the target lane is latched
   and the force sustained until the marking is crossed.
2. **Keep-right is pressure from a faster follower, not "move right when it is
   free".** The blunt rule made every unimpeded car in the middle and left lanes
   commit right on each cooldown expiry, arrive, find itself slower than it
   wanted and pull out again: ~660 manoeuvres per 3 runs, lane-keeping offset
   0.42 m against a real 0.27, median manoeuvre 7–11 s. Keying it to the
   *follower's* frustration reproduces the actual rule drivers follow, and it
   makes the framework's central claim structural rather than stipulated — an
   ordinary faster follower and an emergency vehicle push the car in front
   sideways through the same mechanism, differing only in the strength and range
   of the push.
3. **Gap acceptance is a closing time that covers the crossing, not a
   distance.** This took two rounds of measurement. A static ±20 m veto
   **produced collisions**: highD's measured per-lane speeds differ by up to
   17 m/s (24 / 34 / 41 m/s right-to-left), so a follower 21 m back is 1.2 s
   away. Adding a closing-time test at the driver's own headway (1.1–1.8 s)
   **still produced collisions** — 3 contact frame-pairs in free flow, all
   between 17–19 m trucks, against **0 with the incentive switched off**, which
   is what identified the new term rather than the pre-existing geometry as the
   cause. The reason is that a lane change *takes* ~2.3 s, so a follower judged
   "1.85 s away" arrives mid-manoeuvre, and at 13 m/s closing a following truck
   cannot brake out of the conflict once it has begun. The threshold is
   therefore `T_hw_i + w / v_lat_max_i` — the driver's own headway plus their own
   time to cross the lane. Both are existing per-driver quantities, so gap
   acceptance stays tied to the limits the manoeuvre itself obeys and no new
   parameter was introduced.

### 2b. Density-dependent lane discipline

```
a_pin_eff,i = a_pin_i * (rho_i / rho_ref)^k_rho * (1 - urgency_pin_relief * u_i)
rho_i   = own-lane vehicles within +/- R_rho (100 m), per km per lane
rho_ref = 10.61 veh/km/lane   <- MEASURED (highD train:freeflow), not fitted
```

`k_rho = 0` is exactly the previous behaviour. `k_rho > 0` means discipline
stiffens as traffic densifies, which is the physical reading of the congested
regression, and it suppresses discretionary lane changing in dense traffic
without a second mechanism. Since `k_rho` is a density exponent it cannot be
identified at one density, so the stage-1 objective spans **free-flow (44
carriageways) and dense (27)**; **congested (2 carriageways) is held out** — it
is both the weakest evidence in the dataset and the block the previous fit
damaged.

Consequence for the theory: the analytic cleared half-width
`d* = w_need + B_c·ln(A_c/a_pin_eff)` now carries the density factor inside
`a_pin_eff`, so the depinning prediction is density-dependent too.

---

## 3. Parameter partition

| block | parameters | fitted against | stage |
|---|---|---|---|
| N-lon | `tau`, `T_hw_lo/hi`, `s0`, `A_v`, `Bx_v`, `a_max`, `b_comf` | highD `thw_*`, `ttc_*`, `accel_p99`, `decel_p01`, `peak_decel_*` | 1 |
| N-lat | `a_pin`, `zeta_lat`, `v_lat_max`, `a_lat_max`, `sigma_off`, `het_lat`, `k_rho` | highD `lc_dur_full_*`, `lc_c2c_*`, `lc_peak_vy_full_*`, `peak_alat_*`, `lane_offset_*` | 1 |
| N-lc | `A_pass`, `A_keep_right`, `T_frust`, `s_veto` | highD `lc_per_veh_km_complete` (0.2758 free-flow, 0.2026 dense) | 1 |
| **E** | **`A_ev`, `B_ev`, `A_c`, `B_c` — these four only** | dashcam ground truth (fingerprint) | 2 |
| frozen | everything else | — | — |

### The objective rewarded switching the new term off — twice

Worth stating before any result, because it is the failure that would have made
the whole extension look unnecessary. In **both** halves of the objective,
"the model cannot produce this observable at all" was cheaper than "the model
produces it badly":

| where | old behaviour | consequence | fix |
|---|---|---|---|
| moment term (`distance`) | a NaN cost **2** scale units; a real value could cost up to **99** (a 37 s manoeuvre against a 2.28 s target on a 0.35 s scale) | switching the incentive off was the cheapest move available | cap every deviation at `D_CAP = 8` and price a NaN at `D_CAP` too |
| shape term (`shape_distance`) | an **empty** model pool was *skipped*, so the mean was taken over the remaining keys | a model with no manoeuvres scored a better distribution distance than one with imperfect manoeuvres | price a missing distribution at `SHAPE_MISS = 2.0`, the value `loss()` already used when the shape term was unavailable |
| moment term, again | a median was taken over **however many** events existed, with no minimum sample size | a fit could suppress manoeuvres almost entirely and be credited for the few well-shaped survivors | blank the kinematic keys below `LC_MIN_EVENTS = 20` complete manoeuvres |

The shape hole was large enough to invert the baseline ordering: the
no-lane-change preset scored **4.23** against **4.72** for the same model with
the term switched on, i.e. the objective preferred the defect.

**The third one produced a result that had to be retracted, which is why it is
worth this much space.** With the first two fixed, the `lat` block was fitted on
its own and reported an objective of **1.86** against 4.59 for the previous
parameters, with `a_pin` rising from 0.2236 to 1.53 — a tidy story about the
2026-07-29 fit having been distorted by EV-induced manoeuvres. Then the
observable table showed it: that fit produced **5 complete manoeuvres over
612 veh-km**, a rate of 0.008 against a measured 0.276, and was nevertheless
credited with a lane-change duration 1.25 s.d. from target. Missing the rate cost
2.5 scale units; matching a median estimated from 5 events earned more than that.
Re-scored with the minimum-event rule, the same parameters give **4.63** — the
whole apparent gain was credit for kinematics the model had not demonstrated, and
`a_pin = 1.53` is not a finding.

Two consequences beyond the guard itself:

* **`a_pin` and `A_pass` must be fitted together** (`SPEC_LATLC`). They are
  coupled through the depinning threshold `A_pass·frust > a_pin_eff`, so in
  separate blocks raising the threshold in one silently deletes what the other
  produced — which is exactly the mechanism above.
* Within the `lc` block `A_pass` is bounded below at 0.5, where manoeuvres are so
  slow that every kinematic key sits at the cap, so the fit cannot reach "off" by
  drifting to a bound either.

All three fixes are opt-in — `distance(nan=2.0, cap=None)`,
`shape_distance(miss=None)` and the minimum-event rule living in
`normal_observables` rather than in `model_observables` — so the published
2026-07-29 objective is bit-identical, which `tests/test_highd.py` still asserts.
`tests/test_terms.py` §4b asserts all three properties directly.

The general lesson, stated once because it cost three rounds to learn: **when a
mechanism is added to a model and scored against data, check that the objective
prices its absence at least as harshly as its worst possible presence — and that
every statistic it is scored on is actually supported by enough samples to
mean anything.** Otherwise the fit quietly deletes the mechanism and reports an
improvement.

### The objective's scale convention, corrected

`OBS_SPEC`'s docstring says its `scale` is "roughly the between-carriageway
spread". Measured against `out/highd_targets.json`, the ratios run 1.7–5.4 with a
**median of 2.44**, i.e. one scale unit is ~2.4 s.d., not one. The new
longitudinal scales are therefore set at 2.5× the measured s.d.; had they been
set at 1× s.d. as the docstring implies, every longitudinal residual would have
counted ~2.5× a lateral one of the same statistical size and the objective would
have been silently reweighted toward car-following.

---

## 4. Stage separation is a tested property, and it has six channels not four

`tests/test_terms.py` §2 asserts what makes a two-stage fit meaningful: **the
stage-2 parameters are provably inert in the stage-1 scenario.** Perturbing
`A_ev`, `B_ev`, `A_c`, `B_c` far outside their search bounds leaves an EV-free
run bit-identical, so stage 1 cannot be contaminated by stage-2 values.

The converse — "zero the EV force amplitudes and you are back to the normal
model" — is **false**, and the test asserts that too, because it changes how the
extension must be described. Perceiving a siren also engages:

* `aware_amax_boost` (1.35): yielding drivers accept stronger acceleration;
* `urgency_pin_relief` (0.5): urgency relaxes lane discipline, lowering the
  depinning threshold;
* and, since this extension, the rule in `lanechange.py` that a driver already
  yielding does not *also* start a discretionary manoeuvre.

So the blue-light extension has **six channels**: three force amplitudes, two
host-traffic parameters that only act on a driver responding to a siren, and one
suppression rule. Closing all six does reproduce the EV-free model bit for bit
(also asserted). This is why `emv/terms.py:ABLATIONS` carries an explicit `stage`
field per entry rather than inferring it from the term: `urgency_pin_relief` and
`aware_amax_boost` belong to the lane-keeping and car-following blocks but are
invisible to the EV-free objective, and inference classified them wrongly.

---

## 5. The three instruments

Kept in one registry (`emv/terms.py`) so they cannot drift apart, and so a force
term added to `forces.total` without being analysed is a test failure.

| instrument | question | character |
|---|---|---|
| **force budget** | what does the work? | descriptive — effort, not importance |
| **ablation** | what is the term *for*? | prescriptive — the one that answers it |
| **sensitivity** | which parameter moves which observable? | continuous, sees interactions |

All three are needed because they disagree in an informative way: a term can
dominate the acceleration budget and change no outcome (the lane potential spends
most of its effort holding cars where they already are — which is why the budget
reports the signed *impulse* next to the mean |a|), or be a small share of the
budget and decide everything (the corridor term).

The budget is exact rather than estimated: `tests/test_terms.py` asserts the
accounting identity, including the IDM `min`, to 1e-12:

```
ay = ay_drive + ay_lane + ay_incentive + ay_sfm + ay_ev + ay_corr
ax = ax_drive + min(ax_sfm + ax_ev + ax_corr, a_IDM)
```

Ablations switch a term off **through a parameter**, never by editing the force
composition, so an ablation is a point in the same space the calibration searches
and "the term is off" means exactly "its amplitude is zero". This required a
`clip=False` path (`highd_fit.make_params_spec`): several amplitudes have search
bounds starting above zero (`a_pin` at 0.15, `A_pass` at 0.5, because values below
the pinning threshold are all equivalent to "off"), and clipping silently turned
"no lane discipline" into "a little lane discipline" — which showed up as every
ablation apparently *improving* the objective by ~7 units.

---

## 5b. What "good" means here

The acceptance bar is highD's own **leave-one-carriageway-out** distance: how far
one real carriageway sits from the pooled reference of the others, scored with the
same objective and the same `D_CAP` convention. A model no further away than that
is inside the data's own variability. Recomputed for the normal objective
(`experiments/run_normal_validation.py`), it is higher than the published bar
because more observables are scored:

| regime | carriageways | bar (mean) | median | published bar (old objective) |
|---|---|---|---|---|
| free-flow | 88 | **0.4913** | 0.4688 | 0.408 |
| dense | 29 | **0.3869** | 0.3618 | 0.324 |
| congested | **2** | 0.5840 | — | — |

**Compare the bar with the *moment* component, not the total.** The bar is a
moment distance; `loss_normal`'s total is `moment + w_shape · shape` averaged over
regimes, so quoting the total against the bar would flatter or damn the fit by an
arbitrary amount. The validation record reports the two separately for exactly
this reason.

Stage 2 gets the analogous bar from its own ground truth: the mean distance of one
dashcam video's fingerprint to the pooled reference of all of them
(`run_ev_calibration.py:load_gt`). It is needed there too, because the corpus has
grown to 5 videos / ~145 k observations since the survey benchmark ran on one
video / 2528 observations — so **stage-2 fingerprint distances are not comparable
with the 0.646–0.812 in `PROJECT_LOG.md` §5** — and because the pooled corpus
carries visible tracking noise (5th-percentile acceleration −16 m/s², which no
vehicle produces).

## 6. Results

Full tables: `out/term_tables.md`, regenerated from the stored records by
`experiments/make_term_figs.py` — no number in them is typed by hand.

### 6a. Identifiability (LHS n=96, common random numbers, seeds 11–13)

`out/term_influence_sens.json`, `out/fig_term_sensitivity.png`. Significance
threshold |ρ| > 0.203. **18 of the 19 parameters clear it**, and their primary
responses spread across distinct observables — which is what makes the block-wise
fit well posed rather than a convenience:

| block | parameter | primary response | ρ |
|---|---|---|---|
| N-lat | `sigma_off` | `lane_offset_p90` | **+0.970** |
| N-lat | `a_pin` | `lat_speed_p90` | −0.762 |
| N-lat | `v_lat_max` | `lc_peak_vy_full_med` | +0.756 |
| N-lat | `a_lat_max` | `peak_alat_p99` | +0.740 |
| N-lat | `zeta_lat` | `lc_c2c_med` | +0.512 |
| N-lat | `het_lat` | `thw_p85` | +0.353 |
| N-lat | `k_rho` | `ttc_p01` | +0.274 |
| N-lon | `T_hw_hi` | `decel_p01` | +0.660 |
| N-lon | `tau` | `ttc_p05` | +0.572 |
| N-lon | `a_max` | `decel_p01` | +0.538 |
| N-lon | `A_v` | `lat_speed_p90` | +0.444 |
| N-lon | `T_hw_lo` | `ttc_p05` | +0.342 |
| N-lon | `b_comf` | `peak_decel_p99` | −0.309 |
| N-lon | `s0` | `thw_p85` | −0.230 |
| N-lon | `Bx_v` | `peak_alat_p99` | −0.219 |
| N-lc | `T_frust` | **`lc_per_veh_km_complete`** | **+0.553** |
| N-lc | `A_pass` | `peak_alat_p90` | +0.427 |
| N-lc | `A_keep_right` | `thw_med` | +0.339 |
| N-lc | `s_veto` | `peak_alat_p99` | −0.196 **(not significant)** |

Three readings worth keeping:

1. **The rate and the shape of a discretionary manoeuvre are controlled by
   different parameters, as the design intended.** `T_frust` — how close a leader
   has to be before it frustrates — owns the lane-change *rate*, while `A_pass`
   owns the lateral *acceleration*. Had one parameter owned both, the "fit the
   rate and the kinematics against different observables" split in §3 would have
   been wishful thinking rather than a property of the model.
2. **`s_veto` is not identifiable** and is the only parameter that is not.
   Reported rather than dropped: gap acceptance is dominated by the closing-time
   criterion (§2a item 3), which is built from `T_hw` and `v_lat_max`, so the
   static distance window has little left to do. A later fit could freeze it.
3. **`k_rho` is weakly identified** (+0.274, its strongest response, and not a
   lateral one). Expected: free-flow and dense differ by only ~2× in density
   (8.4 against 16.0 veh/km/lane), and the regime that would pin the exponent
   down — congested, at 37.8 — is the 2-carriageway holdout. The density
   mechanism is therefore *available* and *tested* but not strongly constrained
   by this dataset, which is a statement about highD's coverage, not about the
   mechanism.

The linear surrogate explains only R² = 0.36–0.52 of the objective and its two
halves, so the rank correlations above are the trustworthy summary and
standardized regression coefficients (stored alongside) should not be read on
their own — the same caveat the 2026-07-23 study recorded.

### 6b. Force budget

See §5's table 2 in `out/term_tables.md` and `out/fig_term_budget.png`. At the
pre-fit reference point: laterally `lc` 25–31 %, `drive` 28–29 %, `lane` 21–23 %,
`sfm` 14–20 %, and with an emergency vehicle present `corr` 7–9 % plus `ev` ~1 %.
Longitudinally the social terms are tiny in absolute terms (0.013–0.048 m/s²)
while the **IDM bound binds 36 % of samples in free flow and 74 % in dense,
adding ~4.5–5.0 m/s² of braking when it binds**.

So the emergency-vehicle terms account for about a tenth of the lateral effort and
decide the outcome, while longitudinally the model is very nearly its safety layer
alone. That is the clearest possible statement of why the budget is not an
importance measure — and why the ablation instrument exists next to it.

### 6c. The rate/shape tension: `a_pin` is doing two jobs

The two stage-1 blocks bracket a structural limitation of the model, and finding
it is arguably the most useful thing the per-term analysis did. Both were fitted
on the same EV-free objective over the same two regimes; they differ mainly in
`a_pin` (1.139 in `latlc`, left at 0.2236 in `lon`, which does not touch it):

| free-flow observable | `latlc`, a_pin 1.14 | `lon`, a_pin 0.22 | highD |
|---|---|---|---|
| lane-change duration, med | **2.280** (dev 0.00) | 4.320 (5.83) | 2.280 |
| centre-to-centre, med | 3.600 (0.27) | 6.840 (7.47) | 3.480 |
| peak lateral accel p90 | 0.299 (0.34) | 0.520 (1.87) | 0.333 |
| peak lateral accel p99 | 0.567 (0.50) | 1.171 (2.52) | 0.667 |
| **rate per veh-km** | 0.091 (1.74) | **0.245** (0.29) | 0.276 |
| complete manoeuvres | 57 | 159 | — |

Strong pinning reproduces highD's manoeuvre *shape* almost exactly and makes only
a third of the manoeuvres; weak pinning makes nearly the right *number* and each
takes twice as long as a real one. Neither is a bad optimum — they are two ends of
one trade-off, because `a_pin` sets both

* the **depinning threshold** `A_pass·frust > a_pin_eff`, hence how *often* a
  manoeuvre is started, and
* the **restoring force** of the well, hence how *long* it takes and how hard the
  car accelerates sideways.

One parameter cannot set a rate and a shape independently. That is the same
lesson the 2026-07-29 session learned about medians and spreads — which was fixed
by adding a mechanism (`het_lat`), not by optimising harder — and the same remedy
applies: `params.a_commit` separates the decision threshold from the potential
amplitude, defaulting to `None` = `a_pin` so the pure depinning reading and every
existing result are unchanged.

**Measured, at the fitted `latlc`+`lon` point** (3 seeds, free-flow;
`out/log_probe_commit.txt`):

| `a_commit` | manoeuvres | rate | duration | centre-to-centre | peak alat p90 | collisions | loss |
|---|---|---|---|---|---|---|---|
| highD | — | **0.2758** | 2.280 | 3.480 | 0.333 | — | — |
| `None` (= `a_pin` 1.14) | 65 | 0.1026 | 2.280 | 3.480 | 0.350 | 0 | 1.747 |
| **0.8** | 145 | **0.2263** | 2.400 | **3.480** | 0.395 | **0** | 1.765 |
| 0.5 | 239 | 0.3693 | 2.400 | 3.720 | 0.416 | **9** | 24.565 |
| 0.3 | 361 | 0.5557 | 2.400 | 3.600 | 0.454 | 4 | 12.478 |
| 0.15 | 485 | 0.7482 | 2.400 | 3.600 | 0.458 | many | 45.392 |

Three conclusions, and the second is the one that matters most:

1. **The decoupling works as predicted.** At `a_commit = 0.8` the rate more than
   doubles (0.103 → 0.226, i.e. 82 % of the measured 0.276) while the manoeuvre
   shape is preserved — centre-to-centre stays at exactly the measured 3.480 s and
   the duration moves only 2.280 → 2.400 s. So `a_pin` was genuinely overloaded,
   and the rate is now *nearly* matchable where before the extension it was
   structurally 0.03, a factor of ten out.
2. **There is a collision-free rate ceiling of ~0.23 per veh-km**, against a real
   0.276. Below `a_commit = 0.5` free-flow merges start colliding (penalty 45 and
   20 in the table; **dense stays collision-free throughout**), because free-flow
   carries the largest per-lane speed differences — 24 / 31 / 34 m/s — so a merge
   there is a high-closing-speed event. Real traffic achieves 0.276 without
   colliding, so the model's gap acceptance is not conservative enough once
   manoeuvres are frequent. That is a specific, mechanised limitation with an
   obvious next step, not a vague residual.
3. **The objective is nearly indifferent between the two** (1.765 against 1.747),
   because more manoeuvres also mean more lateral acceleration and highD's peak is
   low (p90 0.333). Stated rather than silently re-weighted: `a_commit = 0.8` buys
   a 2.2× better lane-change rate for a 0.018 penalty in the objective, and the
   objective's weighting is what makes those equivalent. Promoting it is a
   judgement call — the one observable this whole extension exists to make
   matchable — and it is recorded here as such rather than presented as an
   optimiser output.

Two notes for whoever reads the fitted values:

* **`lon`'s lane-change numbers are not a car-following result.** That block only
  fits the longitudinal parameters; its rate of 0.245 is inherited from
  `TUNED_HIGHD`'s weak `a_pin`, and its 4.32 s manoeuvres are the price. It is
  quoted here purely as the other end of the trade-off.
* Both blocks are **collision-free in both regimes**, and the longitudinal
  residuals they share — `thw_p85` at the cap, `peak_decel_p99` and `accel_p99`
  far above the measured values — say the model's car-following is still more
  aggressive and more widely spread than real German motorway traffic. That is
  consistent with the force budget: longitudinally this model is almost entirely
  its IDM safety layer, and the layer is tuned for safety rather than for
  reproducing a headway distribution.

### 6d. Stage 1: the fit and its out-of-sample validation

`out/normal_calibration_joint.json` (19 parameters, LHS 60 + Nelder-Mead 70, 132
evaluations, seeds 11–13, warm-started from `lon` and `latlc`), promoted to
`params.TUNED_NORMAL`. Objective **1.524** against 4.689 for the 2026-07-29 fit
and 5.327 for the dataclass defaults. Collision-free in both fitted regimes.

Out of sample on **fresh seeds 21–25**, never used in any fit
(`out/normal_validation.json`; the bar is highD's own leave-one-carriageway-out
distance, a *moment* distance, so compare it with the moment column):

| block | fitted | previous fit | uncalibrated | bar | within 2 s.d. |
|---|---|---|---|---|---|
| train:freeflow | **1.503** | 3.868 | 4.354 | 0.491 | 8/20 |
| train:dense | **0.839** | 3.328 | 3.795 | 0.387 | 11/20 |
| **holdout:freeflow** (locations never fitted) | **1.530** | 3.948 | 4.412 | 0.491 | 7/20 |
| **all:congested** (regime never fitted) | **2.479** | 3.807 | 3.505 | 0.584 | 4/19 |

Four readings:

1. **It transfers.** The held-out *locations* score 1.530 against the fitted
   block's 1.503 — indistinguishable, so this is not overfitting.
2. **The congested regression is reversed.** This is the headline correction to
   the previous session: fitted on free-flow alone, the 2026-07-29 parameters left
   congested *worse than not calibrating at all* (6.366 against 3.304 on its own
   objective). Here the fitted model is **better than uncalibrated** on congested
   (2.479 against 3.505) despite congested being a holdout at 37.8 veh/km/lane,
   more than double the highest fitted density.
3. **It is 2.0–3.1× the data's own variability**, on a 23-observable objective
   that now includes the car-following block — against the 10 lateral observables
   the previous fit was scored on. Not indistinguishable from real traffic; a
   factor 2.6 closer than before, measured out of sample.
4. **Congestion still produces contacts, and that is pre-existing.** 318 contact
   frame-pairs for the fit against 277 for the *untouched defaults* and 329 for
   the previous fit — so the model has always done this at 37.8 veh/km/lane, and
   the fit adds ~15 % more in a regime it was not fitted on. Reported, not hidden.

### 6e. What each term is for: the ablations at the fitted point

`out/term_influence.json`, baseline 1.5232. **Every ablation is run at the fitted
point, and this matters**: at the hand-set pre-fit reference, switching the
discretionary term off *improved* the objective by 2.7, because a badly-tuned
mechanism is worse than none. A term earns its place only once calibrated.

| ablation | Δ objective | what it says |
|---|---|---|
| `no_idm_bound` | **+52.09** | the longitudinal safety layer is decisive |
| `no_pin` | **+49.87** | without lane discipline there are no lanes, only a road |
| `no_sfm_far` | **+10.30** | long-range car–car repulsion is load-bearing |
| `no_lc` | **+2.97** | discretionary lane changing earns its place |
| `no_keep_right` | **+2.95** | …and almost all of that is the keep-right half |
| `no_het` | +1.64 | per-driver dispersion of the lateral block |
| `no_offset` | +0.57 | per-driver lateral offset |
| `no_overtake` | +0.16 | a driver's own wish to overtake contributes little |
| `no_sfm_near` | +0.02 | the near-field body term is nearly inert here |
| `no_density_pin` | +0.005 | invisible to this objective — see below |

Two findings worth more than the ranking itself:

**The discretionary term is really a follower-pressure term.** `no_keep_right`
(+2.951) accounts for essentially all of `no_lc` (+2.967), while `no_overtake` is
+0.160. So what the model needs is not drivers wanting to get past, but drivers
*being pushed aside by someone faster behind them*. That is the same mechanism the
emergency vehicle uses, at a different strength and range — the framework's
central claim, arrived at from the ablation rather than assumed in the design.

**An ablation can only see what the objective contains.** `no_density_pin` costs
+0.005, i.e. `k_rho` is worthless *on the fitted regimes* — free-flow and dense
differ by only ~2× in density. But `k_rho = 0.914` is the parameter credited above
with reversing the congested regression, and congested is a **holdout**, outside
the objective by construction. The ablation and the holdout have to be read
together; either alone would mislead. Tested directly rather than inferred
(`out/log_probe_krho.txt`, congested block, fresh seeds 21–23):

| congested | moment | shape | contact frame-pairs |
|---|---|---|---|
| stage-1 fit (`k_rho` = 0.914) | **2.533** | **0.658** | **173** |
| stage-1 fit with `k_rho` = 0 | 4.404 | 2.231 | **13 809** |
| 2026-07-29 fit | 3.812 | 3.100 | 190 |
| uncalibrated defaults | 5.242 | 1.179 | 160 |

Switching the density modulation off multiplies contacts in congestion by **80×**
and makes the fit worse than the 2026-07-29 one it replaced. So `k_rho` is the
parameter holding the congested regime together — while being worth +0.005 on the
objective that was actually optimised. This is the strongest single argument in the
study for reading ablations and holdouts together.

### 6f. Force budget at the fitted point

Lateral shares, and how they moved from the pre-fit reference of §6b:

| scenario | `lane` | `sfm` | `lc` | `drive` | `corr` | `ev` | IDM binds | adds |
|---|---|---|---|---|---|---|---|---|
| free-flow, no EV | 37 % | 36 % | 14 % | 13 % | — | — | 41 % | 4.92 m/s² |
| free-flow, EV | 39 % | 29 % | 10 % | 13 % | 7 % | 2 % | 41 % | 4.92 |
| dense, no EV | 45 % | 44 % | 4 % | 7 % | — | — | 71 % | 4.80 |
| dense, EV | 46 % | 34 % | 3 % | 9 % | 7 % | 1 % | 73 % | 4.85 |

The calibrated model leans far more on lane keeping and car–car repulsion, and far
less on the discretionary term (14 % against 31 %) and on `drive` (13 % against
29 %), than the hand-set starting point did. The emergency-vehicle terms still take
only **9 % of the lateral budget** while deciding the outcome, and longitudinally
the IDM bound binds 41 % of samples in free flow and 71–73 % in dense, adding
~4.8–4.9 m/s² of braking when it binds. Effort is not importance: `no_sfm_near`
costs 0.02 in the objective, and `corr` at 7 % of the budget is what forms the
rescue lane.

### 6g. Stage 2: four parameters, everything else frozen

`out/ev_calibration.json`, promoted to `params.TUNED_BLUELIGHT`. `TUNED_NORMAL`
frozen; only `A_ev`, `B_ev`, `A_c`, `B_c` fitted, against the pooled dashcam
ground truth (5 videos, 32 997 frames, 144 896 observations, rules-only labels)
plus the safety and comfort guards.

| | objective | fingerprint | collisions | min TTC | EV speed ratio |
|---|---|---|---|---|---|
| **fitted** (seeds 11–13) | **0.9418** | 0.942 | 0 | 2.99 s | 0.976 |
| **out of sample** (fresh seeds 21–23) | **1.0676** | 1.068 | 0 | 3.05 s | 0.973 |
| EV block at dataclass defaults | 1.1310 | 1.131 | 0 | 4.05 s | 0.993 |
| published `TUNED` EV block | 1.2692 | 1.269 | 0 | 3.63 s | 0.992 |
| leave-one-video-out acceptance bar | 0.7252 | | | | |

`A_ev` 5.921, `B_ev` 15.418, `A_c` 2.353, `B_c` 2.857.

**The published `TUNED` EV block scores worse than the dataclass defaults on this
base** (1.269 against 1.131), which is the predicted consequence of stage 1 having
moved the lateral parameters and the reason a stage-2 re-fit was necessary rather
than optional. Out of sample the fit sits ~1.5× the ground truth's own between-video
spread.

Physically the fit inverts the 2026-07-14 shape: the point field becomes *stronger
and shorter* (`A_ev` 5.92 against 2.152, `B_ev` 15.4 m against 21.4 m) while the
corridor field becomes *weaker and broader* (`A_c` 2.35 against 3.758, `B_c` 2.86 m
against 0.748 m). On the stage-1 base lane discipline is stiffer (`a_pin` 1.20
against 0.2236), so a diffuse corridor push no longer depins a driver and the
emergency vehicle has to assert itself closer in.

### 6h. What each emergency-vehicle term is for

`out/ev_ablation.json`, measured at `TUNED_BLUELIGHT` (baseline 0.9420) — i.e. at
the fitted point, for the same reason the stage-1 ablations are. Note these are
*fingerprint* units and are **not comparable in magnitude with §6e**.

| ablation | Δ objective | EV speed ratio | corridor clearance |
|---|---|---|---|
| *(baseline)* | — | 0.976 | 118.1 m |
| `no_ev_at_all` (all six channels) | **+0.683** | 0.858 | 54.2 m |
| **`no_corridor_lat`** | **+0.661** | 0.886 | 68.1 m |
| `no_ev_response` (the three forces) | +0.630 | 0.859 | 54.7 m |
| `no_urgency_relief` | **+0.260** | 0.956 | 100.8 m |
| `no_ev_field` | **+0.088** | 0.974 | 117.4 m |
| `no_corridor_lon` | +0.073 | 0.982 | 123.3 m |
| `no_aware_boost` | +0.008 | 0.975 | 118.3 m |

**Removing the lateral corridor term alone costs almost as much as removing all six
channels** (+0.661 against +0.683), while removing repulsion from the emergency
vehicle *itself* costs +0.088 — a seventh as much. So yielding in this model is
produced by repulsion from the vehicle's **predicted corridor**, not from the
vehicle: `FRAMEWORK.md` §2.5 calls the corridor "the key term", and this is the
first time that has been quantified against real data rather than asserted.

Two further readings:

* **`urgency_pin_relief` matters three times more than the EV point field**
  (+0.260 against +0.088), and it is not an emergency-vehicle force at all — it is
  a host-traffic parameter that only acts on a driver who has perceived a siren.
  Which is exactly why `terms.ABLATIONS` carries an explicit stage per entry.
* **The whole block buys +14 % EV speed and 2.2× corridor clearance** (0.858 →
  0.976 speed ratio, 54 → 118 m clearance) with no collisions at either end. The
  four fitted parameters are what turns an ordinary-traffic model into one that
  clears a path.

---

## 7. Limits, stated rather than tuned away

* **The congested regime rests on 2 carriageways.** `k_rho` is weakly determined
  there by construction. Congested is a holdout, not evidence.
* **Regimes are weighted equally, not by evidence.** Free-flow has 44
  carriageways against dense's 27; weighting by evidence would hand free-flow
  62 % of the objective and risk reproducing the single-density failure this is
  designed to avoid. The price is that dense is over-weighted relative to its
  support.
* **highD has no emergency vehicle.** Stage 1 is validated by it; stage 2 is not
  and cannot be. Stage 2's ground truth is the dashcam dataset.
* **Stage 2 fits four parameters against one pooled dashcam corpus.** That is the
  honest ceiling on what this ground truth supports, which is why the block was
  kept to four rather than widened to the ten that were available.
* **The lateral offset is still static.** `sigma_off` reproduces highD's marginal
  offset distribution but not its ~8 s decorrelation, so the pooled lateral-speed
  p90 stays low against a measured 0.19 m/s (part of which is drone-tracking
  jitter). An Ornstein–Uhlenbeck drift remains on the backlog.
* **Model and highD percentiles are still computed differently** (mean of
  per-carriageway percentiles vs percentile of the pooled sample; agreement
  within 7 % for 7 of 8 observables, 25 % for `peak_alat_p90`). See
  `HIGHD_VALIDATION.md` §7c — the fix is to derive the targets from the stored
  `out/highd_pools.npz`.

---

## 8. Reproduction

```bash
# guards first - all of these must pass unchanged
.venv/bin/python tests/smoke.py
.venv/bin/python tests/test_highd.py
.venv/bin/python tests/test_terms.py

# instrument 3 first: it tells the block decomposition what is identifiable
.venv/bin/python experiments/run_term_influence.py --only sensitivity --n-lhs 96 \
    --out out/term_influence_sens.json

# stage 1: two blocks in parallel, then a joint polish.
# Use `latlc`, NOT `lat` then `lc` - the lateral and discretionary blocks are
# coupled through the depinning threshold (see §The objective rewarded…).
.venv/bin/python experiments/run_normal_calibration.py --block lon
.venv/bin/python experiments/run_normal_calibration.py --block latlc \
    --n-lhs 66 --n-nm 60
.venv/bin/python experiments/run_normal_calibration.py --block joint \
    --n-lhs 60 --n-nm 70 \
    --warm out/normal_calibration_lon.json out/normal_calibration_latlc.json
# paste the merged theta into emv/params.py:TUNED_NORMAL
.venv/bin/python experiments/run_normal_validation.py

# stage 2: four parameters, everything else frozen
.venv/bin/python experiments/run_ev_calibration.py \
    --gt-dir ~/emergency-vehicle-dataset-pipeline/output

# instruments 1 and 2, then figures and tables
.venv/bin/python experiments/run_term_influence.py --only budget ablation \
    --preset normal
.venv/bin/python experiments/run_ev_calibration.py --ablate
.venv/bin/python experiments/make_term_figs.py
```
