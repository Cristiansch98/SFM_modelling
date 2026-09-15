# A Physics-Informed Repulsion-Force Framework for Emergency-Vehicle Yielding

*Full mathematical specification, analysis, calibration and SUMO coupling.*
*Reference numbers [n] cite `literature_review_ev_repulsion_models.md`.*

---

## 1. Positioning

The framework is a social-force model (Helbing–Molnár [2]) specialised to the
emergency-vehicle (EV) yielding problem, designed to close the four gaps the
literature review identifies:

| Gap | Mechanism here |
|---|---|
| 1. Repulsion from the EV's *predicted corridor*, not its position | §2.5: corridor field on the route-projected path, in the anticipation spirit of [8, 15] |
| 2. Siren/perception onset under-modelled | §2.7: anisotropic detection radius, lognormal reaction delay, compliance rate, two-channel urgency [20] |
| 3. No calibration data for (A, B, λ)-type parameters | §4: behavioural-target loss + LHS/Nelder–Mead machinery; PIDL-ready [24] |
| 4. Ad-hoc force-to-lane mapping | §5: bounded, overshoot-free projection onto SUMO's sublane dynamics with an explicit safety veto [10, 27] |

All quantities are SI; forces are specific forces (m = 1), i.e. accelerations.
The road is straight along $+x$ with right-hand traffic; lane $k$ has centre
$y_k = (k+\tfrac12)w$, lane width $w = 3.5\,$m; drivable band
$y \in [-s_R,\, n w + s_L]$ includes shoulders.

## 2. The model

### 2.1 State

Vehicle $i$: position $(x_i, y_i)$, velocity $(v_{x,i}, v_{y,i})$, length
$L_i$, width $W_i$, desired speed $v_i^0$, relaxation time $\tau_i$, IDM
headway $T_i$, awareness state $S_i \in \{$UNAWARE, NOTICED, YIELDING, HOLD$\}$,
escape side $\sigma_i \in \{-1, +1\}$, urgency $u_i \in [0,1]$. The EV is one
vehicle with its own driver parameters (assertive: $T=0.6\,$s, $b=4\,$m/s²).

### 2.2 Driving force [2]

$$\vec F^{drive}_i = \frac{v^0_i\,\hat x - \vec v_i}{\tau_i},\qquad
\tau = 0.55\,\text{s (cars)},\ \tau_{EV} = 0.40\,\text{s}.$$

The $y$-component contributes $1/\tau$ of lateral damping (used in §2.6).

### 2.3 Car–car interaction

**(a) Elliptic social repulsion** [2, 11, 16]. With bumper gaps
$g_x = \max(|\Delta x| - \bar L, 0.05)$, $g_y = \max(|\Delta y| - \bar W, 0.05)$
($\bar L, \bar W$ = mean half-dimensions) and the elliptic metric
$q = \sqrt{(g_x/B_x)^2 + (g_y/B_y)^2}$, $B_x = 5\,$m $\gg B_y = 0.9\,$m
(elongated equipotentials ≈ vehicle footprint [11]):

$$\vec F^{cc}_{ij} = \Big[A_v e^{-q} + A_n e^{-q/q_n}\Big]\,
w(\varphi_i)\; \hat g_{ij},\qquad
w(\varphi) = \lambda_v + (1-\lambda_v)\tfrac{1+\cos\varphi}{2},$$

where $\hat g_{ij}$ is the (normalised) gradient of $q$ pointing away from $j$
and $\varphi_i$ the angle between $i$'s heading and the direction to $j$
(anisotropy of the receiver [2]). The near-field term
$A_n = 6\,$m/s², $q_n = 0.25$ is the "body force" of [3]: it out-muscles the
corridor force at contact, so squeezing cars compress to a small positive gap
instead of overlapping — packing during rescue-lane formation is a genuine
force balance. The summed force is capped at $F^{cap}_v = 5\,$m/s²; the EV
receives it discounted ×0.5 (assertive driver).

**(b) IDM safety bound** [4]. For every $j$ ahead with smooth lateral overlap
$o_{ij} = \mathrm{clip}\big((m - \text{latgap}_{ij})/m,\,0,\,1\big)$
($m = 0.5\,$m; EV: $0.35\,$m):

$$a^{IDM}_i = \min_j \Big[-a_{max}\Big(\frac{s^*(v_i, \Delta v_{ij})}{\max(s_{ij}, 0.3)}\Big)^2 o_{ij}\Big],
\qquad s^* = s_0 + \max\big(vT + \tfrac{v\,\Delta v}{2\sqrt{a_{max} b}},\,0\big).$$

**Composition rule.** Social terms may never override safety:

$$a_{x,i} = a^{drive}_{x,i} + \min\big(F^{cc}_x + F^{EV}_x + F^{corr}_x,\; a^{IDM}_i\big).$$

The $\min$ both enforces the IDM demand as an upper bound and prevents mild
braking terms from stacking.

### 2.4 EV point repulsion (anisotropic exponential field)

With $\hat n$ from EV to car, $d$ their distance, $\varphi$ the angle between
the EV heading and $\hat n$:

$$\vec F^{EV}_i = A\, e^{(r - d)/B}\,
\Big[\lambda + (1-\lambda)\tfrac{1 + \cos\varphi}{2}\Big]\hat n,
\qquad \lambda = 0.10 .$$

$\lambda = 0.1$ makes the field strongly forward-focused: cars ahead feel the
full field, cars behind 10 % of it. Calibration (§4) softened and lengthened
this field ($A = 2.15$ m/s², $B = 21.4$ m): its main role is the forward
"clear-the-box" nudge near the EV, while the corridor term does the lateral
work. Capped at 4.5 m/s²; felt only by drivers in YIELDING/HOLD (§2.7).

### 2.5 Corridor repulsion from the predicted path (the key term)

The EV's route is projected forward over the anticipation horizon: on the
straight road the corridor is the segment
$s \in [-L_{back},\, L_{pred}]$ at lateral position $y_c$, with

$$L_{pred} = \mathrm{clip}\big(v_{pred} T_{pred},\ 120,\ 260\big)\,\text{m},
\qquad v_{pred} = \max(v_{EV},\ 0.6\, v^0_{EV}),\quad T_{pred} = 8\,\text{s}.$$

$v_{pred}$ floors at a fraction of the EV's *desired* speed: even a blocked EV
projects its corridor (drivers respond to where it is trying to go [20]); the
120 m floor keeps the corridor covering the proximity-urgency range (§2.7).
For a car at longitudinal distance $s$ and signed lateral offset
$d_\perp = y - y_c$, with required clearance
$w_{need} = \tfrac12(W_{EV} + W_i) + 0.5\,$m:

$$F^{corr}_\perp = A_c\,
e^{-\big(|d_\perp| - w_{need}\big)_+ / B_c}\; u_i\; f(s)\; \sigma_i,
\qquad
F^{corr}_\parallel = -\gamma_c\, u_i\,
\Big(1 - \tfrac{|d_\perp|}{w_{need}}\Big)_+ \big(v_i - \kappa v^0_i\big)_+ .$$

The lateral term saturates at $A_c$ inside the required clearance and decays
with length $B_c$ beyond it; $f(s)$ ramps out smoothly over the last 30 m of
the corridor. The longitudinal term is "pull over **and** slow down": while a
car still blocks the corridor it sheds speed toward $\kappa v^0$
($\kappa = 0.82$), letting it drop into adjacent-lane gaps instead of pacing
the EV. The escape side $\sigma_i$ is chosen once at yield onset (position
sign if clearly off-line, else the side with more drivable room with a
right-hand bias) and held with hysteresis — no dithering.

*Curved roads.* Nothing above uses straightness except the projection: for a
general route, $(s, d_\perp)$ are the arc-length and signed normal distance to
the predicted polyline (`emv/road.py` isolates this computation).

### 2.6 Lane keeping as a washboard potential

$$U(y) = -a_{pin}\frac{w}{2\pi} \cos\!\Big(2\pi \frac{y - w/2}{w}\Big)
\;\Rightarrow\;
F^{lane}(y) = -a^{eff}_{pin} \sin\!\Big(2\pi\frac{y - w/2}{w}\Big),$$

with minima at lane centres and maxima on lane markings;
$a^{eff}_{pin} = a_{pin}(1 - 0.5\,u_i)$ — urgency relaxes lane discipline
(panic modulation, [3]). Road edges add exponential walls
$a_{wall} e^{-\delta/B_{wall}}$ ($a_{wall} = 8 > A_c$: cars stop at the wall).

Linearised about a lane centre the pinning frequency is
$\omega^2 = 2\pi a_{pin} / w \approx 2.5\ \text{s}^{-2}$. The drive force
supplies $1/\tau$ of lateral damping; supplementary damping
$c = \max(2\zeta\omega - 1/\tau,\,0)$ sets the damping ratio to
$\zeta = 1$ (critical) — the stability discipline urged by [10] for
force-to-steering mappings. The EV is exempt from the washboard and instead
tracks a target with a critically damped spring
$a_y = \omega_{EV}^2 (y_t - y) - 2\zeta\omega_{EV} v_y$; the target is the
corridor line, offset to clear the nearest vehicle still straddling it
(bounded weave ±1.6 m — EV drivers thread around stragglers [18]).

### 2.6b Density-dependent lane discipline (added 2026-08-06)

The pinning amplitude carries a measured density modulation:

$$a^{eff}_{pin,i} = a_{pin,i}
\Big(\frac{\rho_i}{\rho_{ref}}\Big)^{k_\rho}
\big(1 - \text{relief}\cdot u_i\big),
\qquad \rho_{ref} = 10.61\ \text{veh/km/lane (measured)},$$

with $\rho_i$ counted in the vehicle's own lane within $\pm R_\rho = 100$ m.
$k_\rho = 0$ recovers §2.6 exactly and is the default. The term exists because
fitting the lateral block at a single density drove $a_{pin}$ down 84 % and left
the congested regime *worse than uncalibrated* — in free flow a lane change is an
isolated deliberate manoeuvre and weak pinning fits, whereas in a jam lane
discipline is what holds the structure together. Every result of §3.1 carries
over with $a^{eff}_{pin}$ in place of $a_{pin}$, so **the cleared half-width
$d^*$ is now density-dependent too**. See `TERM_INFLUENCE.md` §2b.

### 2.6c Discretionary lane changing (added 2026-08-06)

Until this term, the model had **no** mechanism for changing lane in the absence
of an emergency vehicle: measured, 0 lane changes over 206 veh-km against a real
highD rate of 0.276 per veh-km. A driver therefore commits to a target lane when
an incentive exceeds the same depinning threshold of §3.1:

$$A_{pass}\,\phi_i\,(1 - \phi^{left}_i) > a^{eff}_{pin,i}
\qquad\text{(overtake)},$$
$$A_{keep}\,\psi_i\,(1 - \phi^{right}_i) > a^{eff}_{pin,i}
\qquad\text{(return right)},$$

$$\phi_i = \Big[\frac{v^0_i - v_{lead}}{v^0_i}\Big]_0^1
\cdot\Big[\frac{T_{frust} - \text{thw}_i}{T_{frust}}\Big]_0^1,$$

where $\phi_i$ is driver $i$'s own frustration with its leader and $\psi_i$ the
same quantity computed for the vehicle *behind* $i$ — keep-right is pressure from
a faster follower, not a preference for the right lane. $\phi^{left/right}$ is the
frustration the prospective leader in the target lane would cause, making the
incentive comparative as in MOBIL. A commitment requires the target lane to be
free within $\pm s_{veto}$ *and* the nearest follower there to be more than the
driver's own headway $T_{hw,i}$ away in closing time.

While committed the force is constant, $a_y = \pm A_{pass/keep}$, and releases
itself when the marking is crossed — so the *shape* of the manoeuvre (duration,
peak lateral speed) remains a property of the lateral block of §2.6 and not of
these amplitudes. The three measurements that forced this design (a gated rather
than latched force gives 24–38 s manoeuvres; a naive keep-right rule gives 0.42 m
lane-keeping offsets; a distance-only merge veto gives collisions between lanes
whose speeds differ by 17 m/s) are recorded in `TERM_INFLUENCE.md` §2a.

Structural consequence worth stating: an ordinary faster follower and an
emergency vehicle now push the vehicle in front sideways through **the same
mechanism**, differing only in the strength and range of the push. The EV terms
of §2.4–2.5 are an extension of a term ordinary traffic already needs.

### 2.7 Perception and urgency (the gate)

Per-driver state machine, addressing gap 2:

```
UNAWARE --(d_EV < R(front/rear))--> NOTICED --(lognormal delay)--> YIELDING
YIELDING --(EV passed by 8 m)--> HOLD --(1.5-4 s)--> UNAWARE (merge back)
```

Detection is anisotropic ($R_{front} = 140$ m via mirror + siren,
$R_{rear} = 60$ m); reaction delays are lognormal (median 1.0 s, σ = 0.45);
a fraction $p_{nc} = 5\%$ never complies [18]. Urgency combines two channels:

$$u_i = \max\Big[\;\underbrace{\varsigma\big((T^{eff}_{react} - t_{arr})/\sigma_t\big)}_{\text{time-to-arrival}},\;
\underbrace{J_i\; \varsigma\big((R_{urgent} - s)/\sigma_s\big)}_{\text{siren proximity}}\Big],
\qquad \varsigma(z) = (1+e^{-z})^{-1},$$

with $t_{arr} = s / \max(v_{app} - v_i,\ 0.5)$,
$v_{app} = \max(v_{EV}, 0.6 v^0_{EV})$, and the *manoeuvrability factor*
$J_i = \mathrm{clip}(1 - v_i / 15,\ 0,\ 1)$ stretching anticipation:

$$T^{eff}_{react} = T_{react}\,(1 + 2.5\, J_i).$$

Two facts motivate the second channel and the stretch, and both emerged from
failed ablations (§3.4): (i) clearing at crawl speed requires shuffling, so
drivers in slow traffic must start far earlier — the German rescue-lane rule
institutionalises exactly this; (ii) urgency based on the EV's *actual*
approach speed deadlocks when the EV is blocked (slow EV → calm traffic →
blocked EV). Drivers react to the siren, not to a velocity estimate of a
vehicle they cannot see.

### 2.8 Constraints and integration

Accelerations are clamped in the road frame
($a_x \in [-8,\ a_{max}\cdot\text{boost}]$, $|a_y| \le 3$ m/s²,
$v_x \ge 0$, $|v_y| \le 2.2$ m/s — a steering-rate proxy standing in for full
non-holonomic kinematics [6]), then integrated with semi-implicit Euler at
$\Delta t = 0.05$ s. The stiffest terms (walls, near-field) satisfy
$\Delta t\,\omega_{stiff} \approx 0.2 \ll 2$; the scheme's symplectic damping
handles the washboard robustly.

## 3. Analysis

### 3.1 Corridor formation is a depinning transition

A car escapes its lane iff the corridor force exceeds the washboard's maximum
restoring force along the path to the lane boundary:
$A_c u > a^{eff}_{pin}$. Balancing the decayed corridor force against the
pinning force gives the **cleared half-width**

$$d^* = w_{need} + B_c \ln\frac{A_c}{a^{eff}_{pin}}
= 2.5 + 0.748\,\ln\frac{3.758}{0.7} \approx 3.8\ \text{m (calibrated, } u = 1).$$

Measured: the median lateral offset plateau at the EV is 3.8–4.2 m
(`out/fig_bowwave_overtake.png`) — quantitative agreement. Below threshold
($A_c u < a^{eff}_{pin}$) cars only shift within-lane by
$\Delta y \approx F_{corr}/(a_{pin}\, 2\pi/w)$, which reproduces the empirical
"nudge" of adjacent-lane traffic.

The phase sweep (81 simulations, density × $A_c$;
`out/fig_phase_diagram.png`) confirms the prediction: EV progress collapses
for $A_c \lesssim a_{pin}$ at all densities and saturates ≥ 95 % above
$A_c \approx 2$, with the failure boundary rising slowly with density as
lateral space becomes contested.

### 3.2 The bow wave

In the EV comoving frame the yield field is a quasi-steady travelling wave:
onset at $s \approx 100$–140 m (set by perception + $T_{react}$, matching the
50–150 m empirical band [18, 20]), maximum displacement $d^*$ at the EV, and
relaxation behind it (HOLD → washboard recapture). `emv/viz.py:plot_bowwave`
aggregates it; the wavefront steepness is controlled by $\sigma_t$.

### 3.3 Headline results (calibrated; mean ± s.d. over 10 seeds per condition, `out/multiseed.json`)

| metric | motorway overtake | rescue lane in jam |
|---|---|---|
| EV mean speed, no yielding | 27.8±0.3 m/s (77±1 %) | 2.8±0.2 m/s (31±2 %) |
| EV mean speed, force model | **34.2±2.7 m/s (95±8 %)** | **8.5±0.3 m/s (95±3 %)** |
| corridor opens (50 m clear) | immediate | 5.8±4.4 s |
| min TTC | 4.6±0.9 s | 1.9±0.6 s |
| collisions (all 40 runs) | 0 | 0 |
| yield onset (pooled over drivers) | 102±33 m ahead | 109±20 m ahead |
| p95 deceleration | 0.07 m/s² | 0.06 m/s² |

### 3.4 Negative results kept in the record

1. Pure time-to-arrival urgency deadlocks in jams (§2.7); fixed by the
   proximity channel. 2. A corridor tied to the EV's instantaneous speed
   ($L = v_{EV} T_{pred}$, no floor) starves in jams — the corridor must
   encode *intent*. 3. A lane-pinned EV cannot pass non-compliers; the
   bounded weave (§2.6) resolves it. Each ablation reproduces the failure by
   construction and is a one-line parameter change.

## 4. Calibration (gap 3)

No public dataset supports fitting these parameters [18], so the loss encodes
behavioural targets: EV progress (→ pre-clearing ideal [19, 20]), yield-onset
distance ≈ 110 m [18, 20], min TTC ≥ 1.2 s, zero collisions, p95 decel ≤ 3.2
m/s², no lateral oscillation, low background disruption [22].
$\theta = (A_{EV}, B_{EV}, A_c, B_c, T_{react}, \gamma_c)$, Latin-hypercube
(41) + Nelder–Mead (60), both scipy-free (`emv/calibrate.py`):

| $\theta$ | bounds | default | **calibrated** |
|---|---|---|---|
| $A_{EV}$ (m/s²) | 1.5–8 | 4.5 | **2.15** |
| $B_{EV}$ (m) | 8–40 | 18 | **21.4** |
| $A_c$ (m/s²) | 1.5–8 | 4.2 | **3.76** |
| $B_c$ (m) | 0.6–3 | 1.6 | **0.75** |
| $T_{react}$ (s) | 4–16 | 9 | **15.9** |
| $\gamma_c$ (1/s) | 0.05–1.2 | 0.55 | **0.72** |

Loss 0.663 → 0.517 (−22 %). Reading: the point field softens/lengthens (the
corridor does the clearing), the corridor decay tightens to ~1 lane of
clearance, anticipation runs as early as the bounds allow — earliness is
cheap in this loss, consistent with pre-clearing theory [20]. $T_{react}$ at
its bound means the empirical onset target (110 m) binds through the
detection radius instead; joint calibration of $R_{front}$ is the natural
next step. The landscape (`out/fig_calibration.png`) shows a broad basin —
behaviour is robust to ±30 % parameter changes above the depinning threshold.

## 5. SUMO coupling (gap 4)

`emv/sumo/bridge.py` — the *same* `forces.ev_field` / `forces.corridor_field`
/ `Perception` code drives both the standalone integrator and SUMO 1.26 via
TraCI (sublane resolution 0.4 m):

* **Longitudinal**: $F_\parallel < 0 \Rightarrow$ `slowDown(v + F·H, H)`,
  H = 0.8 s, only for cars > 15 m ahead of the EV. `speedMode` stays 31, so
  SUMO's safe-speed logic remains the binding constraint — the exact analogue
  of the min() composition of §2.3.
* **Lateral**: target offset $y_t = y_{lane} + \mathrm{clip}(F_\perp/k,\ \pm5.5)$
  with $k = 1$ m/s² per m (≈ $a_{pin}$: sub-threshold forces produce in-lane
  nudges, super-threshold forces full changes — consistent with §3.1),
  tracked by **rate-limited increments** (≤ 0.25 m per 0.1 s step) via
  `changeSublane`. A proportional step toward a bounded target is the
  discrete, overshoot-free equivalent of the critically damped spring —
  unconditionally stable, no oscillation from the controller side [10].
  A per-step **lateral-gap veto** (> 0.45 m to any neighbour with
  longitudinal overlap, else the manoeuvre is cancelled with
  `changeSublane(0)`) prevents forced side-swipes; large one-shot requests
  are avoided precisely because they would keep executing between vetoes.
* EV: SUMO-native driving with speed-gain lane changing
  (`lcSpeedGain=3, lcAssertive=1.6, lcPushy=0.6`).

Three-way comparison, same seeded traffic (1700 veh/h/lane × 3 lanes, 2.2 km
measured segment; `out/sumo_compare.json`):

| mode | segment time | mean speed | corridor clearance | collisions |
|---|---|---|---|---|
| no yielding | 91.2 s | 86.7 km/h | 27 m | 0 |
| SUMO bluelight [17] | 71.9 s | 110.0 km/h | 31 m | 0 |
| **force bridge** | 82.7 s | 95.5 km/h | **73 m** | **0** |

Honest reading: the native bluelight device still wins on raw EV speed — it
grants the EV hard-coded special rights deep inside SUMO's lane-change model.
The bridge recovers ~40 % of the bluelight's improvement while opening a far
wider corridor, staying collision-free, and adding what the device lacks:
perception onset, urgency timing, per-driver compliance, and a continuous,
calibratable field. Known headroom: allow cooperative native LC during
yielding instead of `lcMode 0`, and calibrate $(k, \gamma_c)$ against the
bluelight reference directly.

## 6. Limitations and extensions

* **Real-trajectory calibration (2026-07-28).** The host-traffic block is now
  fitted to and validated against the highD motorway dataset — see
  `docs/HIGHD_VALIDATION.md` and `emv/params.py:TUNED_HIGHD`. What that does
  *not* cover: highD contains no emergency vehicles, so the EV-response block
  keeps its dashcam-GT/literature calibration, the model still has no
  discretionary lane-change mechanism (its lane-change *rate* is therefore not a
  calibrated quantity), and the crawling-jam regime lies outside highD's coverage
  (its slowest carriageway is 12 m/s against `make_jam`'s 2.5).
* **Straight-road demos.** The corridor maths is polyline-ready (§2.5);
  intersections and junction dynamics [17, 23] are not modelled.
* **No acoustic propagation.** Detection radii are effective constants;
  a siren SPL model with building occlusion would refine $R(\varphi)$ (gap 2).
* **PIDL residual** [24, 25]: the analytic fields are the physics prior;
  a neural residual on $(F_\perp, F_\parallel)$ trained on real yielding
  trajectories (e.g. fire-truck GPS + video as in [18]) is the data path.
* **Probabilistic corridor** [15]: replace the deterministic polyline with
  the distribution of predicted EV paths; the field becomes an expectation.
* **Vehicle classes** [16]: fire trucks project larger $(A, B, w_{need})$
  than ambulances; trivially parameterisable per EV type.
* **Benchmarks**: ILP-optimal clearance [19] as an upper bound; MARL [22]
  as a learning baseline — the calibration loss doubles as their reward.

## 7. Reproduction

```
python tests/smoke.py                      # sanity: model vs baselines
python experiments/run_calibration.py      # writes out/calibration.json
python experiments/run_all.py              # figures + GIFs + summary.json
python experiments/run_multiseed.py        # 10-seed stats + seed-avg phase sweep
python experiments/run_sumo_compare.py     # SUMO three-way comparison
python experiments/play_sumo.py            # playable sumo-gui demo (camera on EV)
```

Map: `emv/params.py` (all constants + literature anchors) · `emv/forces.py`
(§2 forces) · `emv/perception.py` (§2.7) · `emv/dynamics.py`, `simulate.py`
(§2.8) · `emv/scenarios.py` · `emv/metrics.py` (§3.3) · `emv/calibrate.py`
(§4) · `emv/sumo/` (§5) · `emv/viz.py` (all figures/animations) ·
`artifact/emv_lab.html` (interactive lab).
