# Calibration and validation against real highD trajectories

Companion to `docs/FRAMEWORK.md`. Covers what the real motorway data can and
cannot say about this model, how the two sides are made commensurable, what was
calibrated, and what still does not match.

Code: `emv/highd.py` (reader + observables), `emv/highd_fit.py` (targets +
objective), `emv/empirical.py` (single source of truth for reference constants),
`emv/scenarios.py:make_highd_like`, `experiments/run_highd_{extract,calibration,validation}.py`.

---

## 1. What highD can and cannot validate

highD is 60 drone recordings of German motorway traffic at 25 Hz — 110,516
vehicles, 44,476 veh-km, 440 driven hours. **It contains no emergency vehicles.**

It therefore validates:

* how real drivers *execute* a lateral manoeuvre — duration, centre-to-centre
  time, peak lateral speed and acceleration: the kinematics every yielding
  manoeuvre is built out of;
* how precisely drivers hold a lane between manoeuvres;
* the traffic the EV drives through — per-lane speed distributions, truck share,
  geometry, density, deceleration envelope.

It cannot say anything about the EV *response* — yield onset, compliance rate,
corridor strength. Those keep their dashcam-GT and literature calibration
(`emv/params.py:TUNED`, unchanged). Hence a two-stage calibration.

**The two datasets bracket the model.** highD measures *discretionary* lane
changes (peak |v_lat| median 0.96 m/s); the dashcam GT measures *EV-induced*
yielding (lateral speed p90 0.50 reviewed / 1.01 rules-only). Real yielding is,
if anything, gentler than a discretionary lane change. A realistic model should
sit inside that bracket — a stronger claim than hitting either number alone.

The honest sentence, which supersedes `paper/ieee_emv_yielding.tex`'s
"No empirical trajectory validation":

> The host traffic and its manoeuvre kinematics are calibrated and validated
> against real motorway trajectories; the EV-response layer rides on that
> validated base and remains calibrated against dashcam EV ground truth.

---

## 2. Dataset facts (measured, `out/highd_recordings.json`)

| | |
|---|---|
| recordings / locations | 60 / 6 (44 have 3 lanes per direction) |
| vehicles / veh-km / driven hours | 110,516 / 44,476 / 440.2 |
| sampling | 25 Hz; decimated by 3 → **exactly 0.12 s** = the model's `rec_dt` |
| lane width | median **3.89 m** (337 lanes, 3.12–4.41) vs the model's 3.5 m |
| truck share | 19 % overall; **64 % of the rightmost lane** |
| carriageways by regime (**3-lane subset**, 88 of 120) | free-flow 58, dense 27, congested 2, other 1 |

Regimes are classified **per carriageway**, not per recording: one direction can
be jammed while the other flows. Recording 25 is exactly that — direction 1 at
12 m/s and 29 veh/km/lane, direction 2 free-flowing.

A *carriageway* here is one direction of one recording, so 60 recordings give
**120 carriageways** (`upper` / `lower` in highD's drone frame). Two different
counts of 88 appear in this project and must not be conflated: the table above is
the 88 carriageways of the 44 recordings that are 3-lane in **both** directions
(the subset the 3-lane model scenario mirrors), whereas the acceptance bar's
`n_cells = 88` (§7) is the number of **free-flow** carriageways among all 120.
Over all 120 the split is free-flow 88, dense 29, congested 2, other 1.

| regime | carriageways | median speed | density | criterion |
|---|---|---|---|---|
| `freeflow` | 58 | 31.5 m/s | 10.1 veh/km/lane | ≥28 m/s, <1300 veh/h/lane |
| `dense` | 27 | 30.6 | 13.0 | ≥1300 veh/h/lane |
| `congested` | 2 | 12.1 | 29.2 | <18 m/s |

**highD never fully stops.** Its slowest carriageway sits at 12 m/s, whereas
`make_jam` runs at 2.5 m/s. The model's jam scenario is therefore *outside this
dataset's coverage* and is labelled an extrapolation, not validated.

---

## 3. The coordinate contract (three normalisations)

All three are verified in `tests/test_highd.py` on synthetic tracks.

1. **`x, y` are the bounding-box upper-left corner, not the centre.**
   `xc = x + width/2`, `yc = y + height/2`. `width` is the *longitudinal* extent
   (vehicle length) and `height` the *lateral* one. Check: recording 01, mean
   |y + h/2 − lane centre| = 0.29 m against |y − centre| = 1.04 m.
2. **The upper carriageway travels −x.** Mapping `x, y, vx, vy → −x, y−m₀, −vx,
   vy` (upper) and `x, m_last−y, vx, −vy` (lower) puts travel along +x, lane 0
   rightmost and +y to the left — `emv/road.py`'s convention. Check: the
   truck-heavy, ~15 m/s slower lane comes out as lane 0 in both directions.
3. **`laneId` is global across both carriageways** (recording 01: {2,3} upper,
   {5,6} lower). Lane indices come from `recordingMeta`'s marking arrays, never
   from `laneId` arithmetic.

Independent check on the whole chain: running the model's own lane-change
detector over the transformed data finds **136 lane changes in recording 01
against highD's own `numLaneChanges` count of 139** (2 % agreement).

---

## 4. Making the two sides commensurable

The point is that a model number and a highD number with the same name must be
the *same computation*, not two implementations of one definition.

* `emv/metrics.py:lane_change_from_track` was extracted verbatim from
  `lane_change_events` and is called by **both** sides. Only the lane-index
  geometry differs (uniform lane width for the model, measured markings for
  highD, since highD's lanes vary 3.1–4.4 m within a single carriageway).
* Sampling: 25 Hz / 3 = 0.12 s exactly, so no interpolation is involved.
* Geometry: the *model* is moved to highD (`make_highd_like` sets
  `lane_width = 3.90 m`), rather than rescaling the data, which would have
  deflated real lateral speeds by 3.5/3.9 = 0.90.
* **Observation window.** Per-vehicle peak statistics grow with how long a
  vehicle is watched: a highD track lasts 13.1 s (median), a simulated run 70–90 s.
  `behaviour_stats(track_window=13.08)` cuts model traces into blocks of the
  measured track length before taking per-vehicle peaks.
* **Pooling.** One run yields only a handful of lane changes, so per-run medians
  are sampling noise. `metrics.behaviour_pools` / `stats_from_pools` pool raw
  events across seeds, exactly as the highD side pools across carriageways.
* **Longitudinal acceleration is de-biased** per carriageway: highD's x scale
  drifts slightly per flight, showing up as a direction-antisymmetric offset in
  forward acceleration (recording 01: +0.000 upper vs +0.083 lower).

### Two measurement floors, stated so they are not mistaken for behaviour

* `yVelocity` is stored to 0.01 m/s, so `|Δv_y|/0.12 s` is quantised at
  0.083 m/s². The **median** per-track peak lands on exactly 2 quanta (0.1667)
  in every carriageway tested, and widening the difference to 0.48 s does not
  lift it. The median lateral acceleration is therefore resolution-limited and
  is **not** used as a calibration target; the p90 and p99 carry the signal.
* Part of highD's pooled |v_lat| p90 (0.19 m/s) is drone-tracking jitter rather
  than driver behaviour, so that observable is scored at low weight.

---

## 5. A measurement error found in the model's own metric

`lane_change_events` walked the window start back with

```python
a = k - 1
while a > 0 and abs(y[a] - y_from) < settle:   # settle = 0.6 m
    a -= 1
```

At the frame where the lane index changes, the vehicle is already about half a
lane width from the departure centre — measured 1.72 m against `settle` = 0.6 —
so the condition is false immediately and `a` never moves. The window covered
only the **arrival half** of the manoeuvre. Symptom visible in the stored data:
`dy` came out at 1.29 m on a 3.5 m lane, i.e. half a lane change.

Consequences, and why this matters more than a factor of two:

* The recorded free-flow lane-change duration of 0.93 s was a *half-manoeuvre
  straddle time* compared against NGSIM's *full-manoeuvre* 4.01 ± 2.31 s. The
  resulting headline — "free-flow sidesteps ~4× too fast" (`PROJECT_LOG.md` §5,
  paper 4) — is largely an artefact of that mismatch.
* Measured with identical code on both sides, the same model configuration gives
  **1.80 s against highD's 2.28 s**, i.e. ~26 % fast, not 4×.
* The regime verdict **inverts**: the *jam* is the unrealistic one at 10.6 s
  against a real 2.3–2.5 s, where `PROJECT_LOG.md` §5 previously recorded that
  "the jam regime lands inside every empirical band".

**Nothing was overwritten.** `duration`, `dur_c2c`, `dy` and `peak_vy` keep the
original definition, so every published number stays reproducible (asserted by a
digest test); the corrected window is reported alongside as `duration_full`,
`dur_c2c_full`, `dy_full`, `peak_vy_full`, plus a `truncated` flag for
manoeuvres clipped by the trace boundary. All highD comparisons use the `_full`
variant.

---

## 6. The lane-change rate: three numbers, one definition gap

| number | value | definition |
|---|---|---|
| published | **0.124** /veh-km | 5600 complete lane changes / 45,000 veh-km, as printed in the highD paper |
| all recorded events | **0.314** /veh-km | 13,949 `tracksMeta.numLaneChanges` / 44,476 veh-km over the released 60 recordings |
| like-for-like | **0.284** /veh-km | this repo's own count, two-lane-occupancy timed by the model's function, restricted to manoeuvres entirely inside the tracked section |

The repo previously hard-coded 0.124 in five places as *the* highD rate. The
field-of-view filter does **not** explain the gap: our like-for-like count is
0.284, still 2.3× the published figure. Both are reported, neither is presented
as wrong, and `emv/empirical.py:lc_rate_reconciliation()` is the one place that
states it.

Separately, the model has **no discretionary lane-change mechanism** — its
EV-inert control produces exactly 0.00 lane changes per veh-km — so the *rate* is
structurally unmatchable and is carried at weight 0 in the objective. Every claim
about manoeuvre kinematics is explicitly **conditional on a manoeuvre occurring**.

---

## 7. Calibration and results

**Set from measurement, never fitted** (`make_highd_like`): lane count and width,
per-lane desired-speed means and spreads (right→left 24.0 / 30.8 / 34.2 m/s —
the model's hand-set profile was 24.5 / 27.5 / 30.0, far too flat, and real
left-lane traffic *outruns* the model's EV), per-lane truck share with
class-conditional dimensions, and density (8.4 veh/km/lane, against the 22 that
`make_overtake` assumes — the model's "free-flow" scenario was not free-flowing).

**Fitted** (`experiments/run_highd_calibration.py`, LHS 65 + Nelder–Mead,
126 evaluations, 5 seeds, targets `train:freeflow` = 27 recordings /
44 carriageways). There were **two** calibrations, and both are kept because the
first is the control that motivated the second:

| parameter | default | (a) moments only | (b) moments + shape |
|---|---|---|---|
| `a_pin` | 1.4 | 0.864 | **0.935** |
| `zeta_lat` | 1.0 | 1.684 | **0.995** |
| `v_lat_max` | 2.2 | 1.571 | **0.984** |
| `a_lat_max` | 3.0 | 2.150 | **0.333** |
| `sigma_off` | 0.0 (new) | 0.328 | **0.392** |
| `het_lat` | 0.0 (new) | — (not in the set) | **0.399** |

Records: `out/highd_calibration_moments.json` (a) and
`out/highd_calibration.json` (b). `emv/params.py:TUNED_HIGHD` carries (b).

| | moment dist. | shape dist. |
|---|---|---|
| previous parameters | 2.513 | 1.126 |
| (a) moments only | 0.954 | 0.715 |
| **(b) moments + shape** | **0.256** | **0.500** |
| acceptance bar | 0.408 | 0.178 |

*(Distances on the five fitting seeds. On fresh seeds they are higher — see the
out-of-sample section, which is the number to quote.)*

### (a) matched every median, and could not match its own p90s

| observable | highD | (a) | dev/s.d. | (b) | dev/s.d. |
|---|---|---|---|---|---|
| lane-change duration, median | 2.280 s | 2.280 | 0.00 | 2.280 | 0.00 |
| centre-to-centre, median | 3.480 s | 3.480 | 0.00 | 3.480 | 0.00 |
| peak lateral speed, median | 0.955 m/s | 0.957 | 0.04 | 0.953 | 0.04 |
| lane-keeping offset, mean | 0.274 m | 0.237 | 1.24 | 0.273 | 0.01 |
| **lane-change duration, p90** | 3.360 s | 4.332 | **4.5** | 3.432 | **0.33** |
| **peak lateral speed, p90** | 1.239 m/s | 1.011 | **4.7** | 1.249 | **0.20** |
| **peak lateral accel, p90** | 0.333 m/s² | 0.438 | 2.57 | 0.333 | **0.00** |
| lane-keeping offset, p90 | 0.466 m | 0.455 | 0.23 | 0.536 | 1.53 |
| peak lateral accel, p99 | 0.667 m/s² | 1.887 | 27 | 0.333 | 7.5 |
| pooled lateral speed, p90 | 0.190 m/s | 0.042 | 10 | 0.062 | 8.7 |

The important row is not the medians — both fits get those. It is the **p90s**.
Fit (a) missed the lane-change-duration and peak-lateral-speed p90s by 4.5 and
4.7 standard deviations *while optimising exactly those quantities*. It was not
under-optimised; it lacked the degree of freedom. A single shared lateral block
forces one manoeuvre shape on every driver, so a median and a spread cannot be
set independently. Adding per-driver dispersion (`het_lat`, §7b) improved the
**moment** score by a factor 3.7, which is the strongest evidence that the
missing mechanism was heterogeneity rather than a bad optimum.

**The medians match real motorway data essentially exactly; two tails still do
not.** That split is the result, and it is informative rather than cosmetic:

* `peak_alat_p99` too high — a few vehicles still make violent evasive
  manoeuvres. These are EV-induced, so the residual sits in the *EV-response*
  block (`A_c`, `gamma_c`, `urgency_pin_relief`), which highD cannot identify.
  Stage 2.
* `lat_speed_p90` too low — a static per-driver offset reproduces the *marginal*
  offset distribution but adds no lateral *motion*. highD's offsets decorrelate
  with τ ≈ 8 s; an Ornstein–Uhlenbeck drift would be needed, and part of the
  measured 0.19 m/s is tracking jitter in any case.
* `lc_dur_full_p90` too high — the model has a tail of slow drift-outs that real
  drivers do not produce.
* Deceleration (`peak_decel_p99` 7.4 against a real 1.4) and background speed are
  carried at weight 0 because they are governed by the EV block and by the
  scenario, not by the parameters this stage fits. The deceleration gap is real
  and belongs to stage 2.

`out/fig_highd_cdf.png` shows the whole distributions rather than the fitted
percentiles, and makes the shape of the residual obvious: **the model is
under-dispersed.** Its CDFs are systematically steeper than the real ones — it
reproduces the typical manoeuvre but not the variety of manoeuvres, because every
driver shares one parameter set and only `sigma_off` is drawn per vehicle. Real
driver-to-driver heterogeneity would need per-vehicle draws of the lateral block
as well.

### Out-of-sample validation

The acceptance bar is highD's own **leave-one-carriageway-out distance** — how far
one real carriageway sits from the pooled reference of the others: **0.408** in
free-flow (88 carriageways), 0.324 dense.

All three fits, on fresh seeds (21–25), never used in any calibration:

| target block | previous | (a) moments | (b) +shape | **(c) +per-driver clamp** |
|---|---|---|---|---|
| `train:freeflow` (fitted) | 2.987 | 1.252 (4/10) | 0.810 (4/10) | **0.452 (7/10)** |
| **`holdout:freeflow`** (loc 3/4/6, never fitted) | 2.919 | 1.059 (6/10) | 0.881 (5/10) | **0.708 (5/10)** |
| `all:dense` | 2.812 | 1.187 (6/10) | 0.800 (5/10) | 0.838 (3/10) |
| `all:congested` | 3.304 | 3.188 (1/10) | 2.176 (2/10) | **6.366 (2/10)** ← worse than doing nothing |
| shape distance | 1.239 | — | 0.778 | **0.727** |

### The congested regime got worse, and that is a finding

Fit (c) is the best free-flow model by a wide margin and the **worst** congested
one — worse than the uncalibrated baseline (6.366 against 3.304). It is reported
rather than averaged away, because the mechanism is identifiable.

`a_pin` fell to 0.224, a sixth of the hand-set 1.4, and `zeta_lat` rose to 2.42.
In free flow, where cars are far apart and a lane change is a deliberate,
isolated manoeuvre, weak pinning with heavy damping reproduces the data well. In
a jam, lane discipline is what *holds the structure together*: with pinning that
weak, vehicles drift between lanes under the corridor and neighbour forces
instead of holding station. The analytic depinning half-width moves accordingly,
3.56 → 4.93 m.

So the free-flow fit does not transfer to congestion, and the direction of the
failure is the expected one. Three honest consequences:

1. The calibration is **regime-specific**. A single lateral block fitted on
   free-flow traffic should not be presented as a global improvement, and
   `TUNED_HIGHD` should be read as *the free-flow host-traffic block*.
2. The congested reference rests on **two carriageways** — the weakest evidence
   in this document, and not enough to fit against directly.
3. The natural next step is a regime-aware fit (jointly on free-flow and
   congested targets, accepting a worse free-flow optimum) or a density-dependent
   `a_pin`. Neither is done here.

The calibration **transfers out of sample** — the held-out locations score better
than the fitted block, so this is not overfitting. But at 1.06 the model is still
~2.6× highD's own self-distance: it moved a factor 2.8 closer to real traffic
without becoming indistinguishable from it. The congested block barely moves,
which is expected — the fit was done on free-flow and highD's congested reference
is two carriageways.

**The realism is nearly free.** At the same scenario and seeds: EV speed ratio
0.9256 → 0.9246, min TTC 2.07 → 2.27 s, collisions 0.4 → **0.0**, clearance
125.4 → 122.6 m. The analytic depinning half-width rises 3.557 → 3.918 m exactly
as theory requires when `a_pin` falls — which is also why stage 2 is due.

**Identifiability** (LHS 96 × 5 common seeds, |ρ| > 0.204 significant). Each
fitted parameter has a distinct primary observable, so the set is well posed:

| parameter | strongest response |
|---|---|
| `sigma_off` | `lane_offset_mean` **+0.99** |
| `a_lat_max` | `peak_alat_p99` **+0.96** |
| `zeta_lat` | `lc_c2c_med` +0.79, `lc_dur_full_med` +0.78 |
| `a_pin` | `peak_alat_p90` +0.75, `lat_speed_p90` −0.68 |
| `v_lat_max` | `lc_peak_vy_full_p90` +0.41 |

Manoeuvre *duration* is controlled by the damping `zeta_lat`, not by the speed
clamp — which is why tightening `a_lat_max` alone could never have fixed it.

---

## 7b. Matching moments is not matching distributions

The calibration above scored *percentiles*. That is the standard practice and it
worked — every median landed on the data. But a set of percentiles does not
determine a distribution, and the CDF overlay (`out/fig_highd_cdf.png`) showed
the model's curves rising far more steeply than the real ones. To turn that
observation into a number we added a scipy-free **1-Wasserstein** distance on the
pooled samples (`emv/highd_fit.py:wasserstein1`), normalised by the data's own
interquartile range so it is scale-free, plus an **IQR ratio** (model spread ÷
data spread; 1.0 = matched).

The acceptance bar is constructed the same way as before — how far *one real
recording's* distribution sits from the pooled rest:

| observable | bar (mean W1/IQR) | bar p90 |
|---|---|---|
| lane-change duration | 0.157 | 0.232 |
| peak lateral speed | 0.118 | 0.152 |
| lane-keeping offset | 0.109 | 0.143 |
| per-track peak lateral accel | 0.329 | 0.620 |
| **mean** | **0.178** | |

Measured against it:

| | previous params | moments-only fit |
|---|---|---|
| shape distance (mean W1/IQR) | 1.126 | 0.715 |

So fitting moments improved shape as a side effect (−36 %) but left the model at
four times the bar.

**The residual decomposes cleanly into two different problems**, which is the
useful part:

| observable | IQR ratio | W1/IQR | diagnosis |
|---|---|---|---|
| lane-keeping offset | 1.15 | 0.232 | close |
| lane-change duration | 0.75 | 0.319 | mildly narrow |
| peak lateral speed | **0.23** | 0.456 | **spread far too narrow** |
| peak lateral accel | 1.04 | **1.853** | **spread matched, level displaced** |

A *spread* deficit and a *level* deficit are different failures with different
causes. The displaced peak-lateral-acceleration distribution is the EV-induced
violent-manoeuvre tail (`peak_alat_p99` 2.8× high) and belongs to the
EV-response block — stage 2. The narrow spreads are driver-to-driver variety.

### The heterogeneity mechanism, predicted and tested

Note first the internal evidence: `sigma_off` is the *only* per-driver quantity
in the model, and lane-keeping offset is the *only* observable whose spread
matches (IQR ratio 0.14 → 1.15 when it was introduced). That motivated a
prediction, registered before the test:

> Drawing the lateral block (`a_pin`, `zeta_lat`, `v_lat_max`) per driver rather
> than globally will raise the peak-lateral-speed IQR ratio from 0.23 toward 1.0
> and the lane-change-duration ratio from 0.75 toward 1.0.

`params.het_lat` is the lognormal coefficient of variation of those per-driver
draws (0 = one shared block = the behaviour of record). Sweeping it:

| `het_lat` | shape dist. | moment dist. | IQR ratio: duration | peak speed |
|---|---|---|---|---|
| 0.00 | 0.715 | 0.954 | 0.75 | 0.23 |
| 0.15 | 0.660 | 1.130 | **1.00** | 0.63 |
| 0.30 | 0.694 | 1.201 | 1.31 | **1.26** |
| 0.45 | 0.767 | 1.250 | 1.62 | 1.80 |
| 0.60 | 0.807 | 1.169 | 1.88 | 1.97 |

**The prediction holds**: heterogeneity is the only thing that moves the spreads
at all, and both cross 1.0 within the swept range. But two things it does *not*
do, and both are reported rather than tuned away:

1. The two observables reach a matched spread at *different* dispersions (0.15
   and ~0.25), so one scalar cannot fix both — the per-driver draws would need to
   be correlated, or drawn per observable.
2. Widening a distribution at fixed median moves its percentiles, so the
   *moment* fit degrades (0.954 → 1.20). That is not a defect of the mechanism;
   it is evidence that a moments-only objective was the wrong target.

The objective was therefore changed to score **both** terms — moments plus the
normalised Wasserstein shape distance — with `het_lat` as a sixth fitted
parameter, so the optimiser trades them explicitly instead of optimising one and
degrading the other.

---

## 7c. Two caveats on the fitted result, found by auditing it

Both were found by checking a result that looked *too* good, and both are stated
here rather than left for a reader to discover.

### The lateral-acceleration clamp saturates — found, then fixed

*(Diagnosis below; the fix and its verification are at the end of this
subsection. This is kept in full because the diagnosis is the reusable part.)*

Fit (b) reports `peak_alat_p90` = 0.333 against a target of 0.333 — an apparently
perfect match. It is not one. `a_lat_max` was fitted to 0.3333 and **15.3 % of
model tracks sit exactly at it**, so the p90, p95 and p99 are all the same
number: the distribution is *truncated at the clamp*, not reproduced.

| percentile | model | highD (pooled) |
|---|---|---|
| p50 | 0.033 | 0.167 |
| p75 | 0.150 | 0.250 |
| p90 | **0.333** | 0.417 |
| p95 | **0.333** | 0.500 |
| p99 | **0.333** | 0.667 |

The cause is structural and specific: `a_lat_max` is the one lateral parameter
that is **not** drawn per driver, so the model has a single hard ceiling where
the data has a tail. A hard global clamp cannot produce a distribution whose p99
is twice its p90. The obvious next step is to include `a_lat_max` in the
per-driver draws, which would turn one ceiling into a distribution of ceilings.

**Fix applied.** `a_lat_max` is now drawn per driver, sharing a single factor
with `v_lat_max`: how briskly a driver is willing to move sideways is *one*
trait, and drawing the speed and acceleration ceilings independently would
produce incoherent drivers (a high acceleration ceiling with a low speed
ceiling). Lane-keeping discipline (`a_pin`, `zeta_lat`) keeps its own factor, so
the model now has two per-driver traits rather than four independent draws.
Verified at unchanged parameter values, i.e. before any re-fitting:

| | before fix | after fix | highD |
|---|---|---|---|
| tracks pinned to one clamp value | **15.3 %** | **0.0 %** | — |
| p90 / p95 / p99 | 0.333 / 0.333 / 0.333 | 0.285 / 0.350 / 0.528 | 0.417 / 0.500 / 0.667 |
| p99 / p90 ratio | **1.00** (truncated) | **1.85** | 1.60 |

The distribution has a tail again. The levels were then re-fitted, since
`a_lat_max` and the dispersion had both been fitted against the truncated
behaviour — call that fit **(c)**.

**Removing an artefact is not free, and the bookkeeping matters.** Fit (c)'s
objective value (0.893) is *higher* than fit (b)'s (0.756), which looks like a
regression and is not one: the two numbers were produced under different code,
(b)'s while the clamp still truncated the distribution. Re-scoring both thetas
under the *same* corrected mechanism is the only fair comparison:

| | (b) theta | (c) theta | bar |
|---|---|---|---|
| moment distance | 1.184 | **0.343** | 0.408 |
| shape distance | 0.787 | **0.550** | 0.178 |
| `peak_alat` IQR ratio | 1.68 | **1.02** | 1.00 |
| `peak_alat` p99 | 0.528 | **0.701** | 0.667 |

(c) wins on both terms, and its **moment distance of 0.343 is below highD's own
between-carriageway spread of 0.408** — on the moment observables the model now
sits inside the data's own variability. `TUNED_HIGHD` carries (c).

The general lesson is worth stating because it is easy to get wrong: fit (b)
scored well partly *because* of the defect. A hard clamp let it place
`peak_alat_p90` exactly on target while being 7.5 s.d. out on p99, and the
artificially narrow acceleration distribution let the other parameters take more
extreme values. An aggregate score cannot distinguish "matched the data" from
"saturated against a bound", which is why the per-observable audit in this
section, and not the objective value, is what identified the problem.

### Model and data percentiles are not computed the same way

The highD *targets* are the mean over carriageways of each carriageway's
percentile; the *model* side takes the percentile of the pooled sample. For a
skewed distribution these are different statistics. Measured on the same highD
data:

| observable | target (mean over carriageways) | pooled sample | ratio |
|---|---|---|---|
| lane-change duration, median | 2.280 | 2.280 | 1.00 |
| lane-change duration, p90 | 3.360 | 3.600 | 1.07 |
| peak lateral speed, median | 0.955 | 0.950 | 0.99 |
| peak lateral speed, p90 | 1.239 | 1.250 | 1.01 |
| lane-keeping offset, mean | 0.274 | 0.282 | 1.03 |
| lane-keeping offset, p90 | 0.466 | 0.488 | 1.05 |
| **peak lateral accel, p90** | **0.333** | **0.417** | **1.25** |
| peak lateral accel, p99 | 0.667 | 0.667 | 1.00 |

So the inconsistency is within a few per cent for seven of eight observables and
material for exactly one — `peak_alat_p90`, which is also the observable most
affected by the quantisation floor (§4) and by the clamp saturation above. The
three problems compound on the same quantity, which is why no conclusion in this
document rests on it. Fixing it means deriving the targets from the pooled
samples (they are stored in `out/highd_pools.npz`, so no re-extraction is
needed).

---

## 8. Reproduction

```
python experiments/run_highd_extract.py            # ~2.5 min, needs the zip once
python experiments/run_highd_extract.py --only meta # ~2 s, the rate alone
python experiments/run_highd_calibration.py        # ~10 min
python experiments/run_highd_validation.py         # ~20 min
python experiments/make_highd_figs.py              # figures + tables
```

Only the extractor needs `~/highD-dataset-v1.0.zip` (879 MB; streamed, never
extracted — the 4.78 GB of CSVs are read straight out of the archive). Everything
downstream reads the ~3 MB of committed artifacts: `out/highd_recordings.json`,
`out/highd_targets.json`, `out/highd_pools.npz`. Cache validity is keyed on
`EXTRACTOR_VERSION`, the zip's central-directory CRC digest, and the config dict.

---

## 9. Limitations

1. **No emergency vehicles in highD.** The EV-response block is not validated
   here, and no claim in this repo should say otherwise.
2. **No discretionary lane changes in the model**, so the lane-change *rate* is
   not a calibrated quantity (§6).
3. **The jam regime is extrapolated.** highD's slowest carriageway is 12 m/s;
   `make_jam` runs at 2.5 m/s (§2).
4. **Stage 2 is outstanding.** `a_pin` fell 38 %, which lowers the depinning
   threshold `A_c·u > a_pin(1 − relief·u)`, so the EV block should be re-fitted on
   this base against the dashcam GT.
5. **Tails unmatched** (§7), and a static preferred offset is an
   infinite-correlation-time approximation of a τ ≈ 8 s process.
6. **Six German motorway locations, daytime, dry**, and 37 of 60 recordings share
   one location — which is why the holdout is by *location*, not by recording.
7. **Measurement floors** in lateral acceleration and lateral speed (§4).
