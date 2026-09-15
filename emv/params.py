"""Model parameters, with units and literature anchors.

Reference numbers [n] refer to literature_review_ev_repulsion_models.md.
Defaults marked CAL are refined by emv.calibrate (see out/calibration.json);
the tuned values are baked into TUNED below once calibration has run.
"""
from dataclasses import dataclass, replace, asdict


@dataclass
class Params:
    # ---------------- integration ----------------
    dt: float = 0.05            # s, integration step (semi-implicit Euler)

    # ---------------- driving force [2] ----------------
    tau: float = 0.55           # s, speed relaxation time (cars)
    tau_ev: float = 0.40        # s, EV relaxes faster (decisive driver)

    # ---------------- IDM longitudinal safety layer [4] ----------------
    a_max: float = 2.6          # m/s^2, comfortable acceleration
    b_comf: float = 3.0         # m/s^2, comfortable deceleration
    b_emerg: float = 8.0        # m/s^2, physical braking limit
    s0: float = 2.0             # m, standstill jam gap
    # Per-driver IDM time headway is drawn U(T_hw_lo, T_hw_hi) by the scenario
    # builders. Exposed as parameters (2026-08-06) so the car-following block can
    # be fitted against highD's measured headway distribution (thw_med 1.67 s,
    # p15 0.82, p85 3.745) - the defaults are the literal values the builders
    # used before, so the draw and every existing result are unchanged.
    T_hw_lo: float = 1.1        # s
    T_hw_hi: float = 1.8        # s
    idm_overlap_margin: float = 0.5   # m, smooth lateral-overlap onset width
    # Ablation switch for the term-influence study (emv/terms.py): False removes
    # the IDM upper bound from the composition, leaving the bare social force.
    # Never set False in a production run - it is the model's safety layer.
    idm_bound: bool = True
    aware_amax_boost: float = 1.35    # yielding drivers accept stronger accel

    # EV driver style (assertive, expects yielding) [17, 18]
    ev_T: float = 0.6           # s, EV time headway
    ev_s0: float = 1.5          # m
    ev_b: float = 4.0           # m/s^2
    ev_amax: float = 3.2        # m/s^2
    ev_sfm_receive: float = 0.5 # EV discounts social repulsion (assertive)

    # ---------------- lane keeping: washboard potential ----------------
    # U(y) = -a_pin*(w/2pi)*cos(2pi(y - w/2)/w): minima at lane centres.
    # A car leaves its lane only when the external lateral force exceeds
    # a_pin -> depinning transition (docs/FRAMEWORK.md sec. 4).
    a_pin: float = 1.4          # m/s^2, max lane-restoring acceleration  CAL
    zeta_lat: float = 1.0       # -, lateral damping ratio (critical) [10]
    urgency_pin_relief: float = 0.5   # urgency relaxes lane discipline [3]
    a_wall: float = 8.0         # m/s^2, road-edge repulsion strength
    B_wall: float = 0.35        # m, road-edge decay length
    a_lat_max: float = 3.0      # m/s^2, lateral acceleration cap (comfort/friction)
    v_lat_max: float = 2.2      # m/s, lateral speed cap (steering-rate proxy)
    # Real drivers do not track the lane centre: on highD the mean distance from
    # the centre while lane-keeping is 0.28 m (p90 0.48 m), whereas the washboard
    # potential alone pins cars to within 0.04 m. sigma_off shifts each driver's
    # potential minimum by a per-vehicle draw from N(0, sigma_off), which
    # reproduces that spread. 0.0 = off = the behaviour of record for papers 1-4;
    # the measured value is used by scenarios.make_highd_like / TUNED_HIGHD.
    # (A static per-vehicle draw is an infinite-correlation-time approximation:
    #  highD's offset autocorrelation decays with tau ~ 8 s, which this does not
    #  reproduce - see docs/HIGHD_VALIDATION.md.)
    sigma_off: float = 0.0      # m, s.d. of per-driver preferred lateral offset
    # Driver-to-driver variety in *how* a manoeuvre is executed. With one global
    # lateral block every driver changes lane the same way, which makes the
    # model's distributions far narrower than the measured ones (the peak
    # lateral speed IQR came out 4x too narrow). het_lat is the lognormal
    # coefficient of variation applied per vehicle to a_pin, zeta_lat and
    # v_lat_max. 0.0 = one shared block = the behaviour of record.
    het_lat: float = 0.0        # -, per-driver lateral-block dispersion

    # EV lateral control: critically damped spring to corridor line,
    # with weaving around stragglers (non-compliers) [18]
    ev_omega_lat: float = 1.3   # rad/s
    ev_zeta_lat: float = 1.0    # -
    ev_weave_lookahead: float = 30.0  # m, blocker search range
    ev_weave_margin: float = 0.4      # m, lateral body gap the EV aims for
    ev_weave_max: float = 1.6         # m, max offset from the corridor line
    ev_overlap_margin: float = 0.35   # m, EV accepts tighter passing gaps

    # ---------------- discretionary lane changing (added 2026-08-06) ----
    # Why this exists: with no emergency vehicle in the scenario the model made
    # exactly ZERO lane changes over 206 veh-km, against a measured highD rate of
    # 0.2758 per veh-km (free-flow; 0.2026 dense). Every lane change it
    # produced was EV-induced, so the
    # lateral block could only be identified *because* an EV was present - which
    # makes an "ordinary traffic" calibration circular. This term supplies the
    # missing incentive.
    #
    # Its only job is to DEPIN a car from its lane well; the washboard slope then
    # carries it to the next minimum, so the manoeuvre *shape* stays governed by
    # (a_pin, zeta_lat, v_lat_max, a_lat_max) exactly as before. A discretionary
    # lane change therefore happens iff  A_pass * frust > a_pin_eff  - the same
    # depinning threshold the corridor force must cross (A_c * u > a_pin_eff).
    # One condition, two things that can drive it.
    #
    # 0.0 = off = the behaviour of record for papers 1-6.
    A_pass: float = 0.0        # m/s^2, overtaking (move-left) incentive     CAL
    A_keep_right: float = 0.0  # m/s^2, keep-right (move-right) pressure     CAL
    T_frust: float = 2.5       # s, headway below which a slow leader frustrates
    s_veto: float = 20.0       # m, longitudinal window searched for a blocker
    lc_gate_frac: float = 0.35 # -, commit only within this fraction of a lane
    #                            width of the driver's own preferred position
    T_lc_max: float = 8.0      # s, abandon a commitment that has not completed
    T_lc_cool: float = 3.0     # s, refractory period after a manoeuvre ends
    # Decision threshold for committing to a lane change. None = use a_pin, which
    # is the physically pure reading (escape the well you are in) and the default.
    #
    # It is separable because MEASUREMENT showed a_pin is doing two jobs at once,
    # and they pull in opposite directions. a_pin sets the depinning threshold
    # (hence how OFTEN a manoeuvre happens) *and* the restoring force that shapes
    # it (hence how LONG it takes). Fitted at a_pin=1.14 the model reproduces
    # highD's manoeuvre duration exactly (2.280 s vs 2.280) but makes only a third
    # of the manoeuvres (0.091 vs 0.276 per veh-km); at a_pin=0.22 it makes almost
    # the right number (0.245) but each takes twice as long (4.32 s). One
    # parameter cannot set a rate and a shape independently - the same lesson the
    # 2026-07-29 session learned about medians and spreads (see het_lat).
    a_commit: float | None = None   # m/s^2, commit threshold (None = a_pin)

    # ---------------- density-dependent lane discipline (2026-08-06) ----
    # The free-flow-only fit of 2026-07-29 drove a_pin down 84 % and thereby made
    # the CONGESTED regime worse than no calibration at all (6.37 vs 3.30): weak
    # pinning fits an isolated free-flow manoeuvre, but in a jam lane discipline
    # is what holds the structure together. Physically the missing ingredient is
    # that discipline stiffens as traffic densifies:
    #     a_pin_eff = a_pin * (rho / rho_ref)^k_rho * (1 - relief * u)
    # k_rho = 0 reproduces the previous behaviour exactly. rho_ref is MEASURED,
    # not fitted: highD's train:freeflow density (out/highd_targets.json).
    k_rho: float = 0.0          # -, density exponent of lane discipline     CAL
    rho_ref: float = 10.61      # veh/km/lane, reference density (measured)
    R_rho: float = 100.0        # m, half-window of the local density estimate

    # ---------------- car-car social force (elliptic) [2, 11, 16] -------
    A_v: float = 2.5            # m/s^2, strength
    Bx_v: float = 5.0           # m, longitudinal decay length
    By_v: float = 0.9           # m, lateral decay length (elongated field [11])
    A_near: float = 6.0         # m/s^2, short-range (body) term [3]
    q_near: float = 0.25        # -, short-range decay in ellipse units
    lam_v: float = 0.25         # -, anisotropy: rear stimuli weight [2]
    F_cap_v: float = 5.0        # m/s^2, per-vehicle cap on summed force
    pair_cutoff: float = 70.0   # m, interaction cutoff (|dx|)

    # ---------------- EV point repulsion (user term 2) [2, 12, 16] ------
    A_ev: float = 4.5           # m/s^2, strength                          CAL
    B_ev: float = 18.0          # m, decay length (siren heard early)      CAL
    lam_ev: float = 0.10        # -, anisotropy: forward-focused field
    r_ev: float = 3.0           # m, combined size offset
    F_cap_ev: float = 4.5       # m/s^2

    # ---------------- corridor repulsion (user term 3; gap 1) -----------
    A_c: float = 4.2            # m/s^2, lateral strength                  CAL
    B_c: float = 1.6            # m, lateral decay length                  CAL
    margin_c: float = 0.5       # m, extra clearance margin
    T_pred: float = 8.0         # s, path prediction horizon [8]
    L_pred_min: float = 120.0   # m, corridor length bounds (covers R_urgent:
    L_pred_max: float = 260.0   # m   a blocked EV still projects its path)
    L_back: float = 6.0         # m, corridor extends slightly behind EV nose
    end_ramp: float = 30.0      # m, smooth fade at far corridor end
    gamma_c: float = 0.55       # 1/s, longitudinal yield-brake gain       CAL
    kappa_merge: float = 0.82   # -, target speed fraction while merging out
    v_app_floor: float = 0.6    # -, assumed EV approach speed >= this * v0_ev

    # urgency = sigmoid((T_react_eff - t_arrival)/sigma_t): timing knob (gap 2, [20])
    T_react: float = 9.0        # s, EV time-to-arrival that triggers action CAL
    sigma_t: float = 2.2        # s, softness of the timing threshold
    # anticipation grows as manoeuvrability shrinks: clearing at crawl speed
    # requires shuffling, so drivers in slow/standing traffic pre-form the
    # corridor (Rettungsgasse rule) instead of waiting for the EV [17, 20, 21]
    jam_anticipation: float = 2.5   # -, T_react multiplier at standstill
    v_manoeuvre_ref: float = 15.0   # m/s, speed above which anticipation is off
    # proximity urgency channel: in slow traffic drivers cannot estimate the
    # EV's arrival (it may itself be blocked) and respond to siren distance
    # instead; breaks the slow-EV/slow-corridor feedback deadlock
    R_urgent: float = 90.0          # m, siren-proximity urgency range at standstill
    sigma_s: float = 12.0           # m, softness of the proximity threshold
    # floor on the jam gate of the proximity channel. 0 = literature behaviour
    # (proximity urgency only in slow traffic); 1 = drivers respond to siren
    # distance at any speed, which is what a human-driven EV needs because its
    # closing speed is low and t_arrival stays large until it is right behind.
    prox_gate_floor: float = 0.0    # -, in [0, 1]

    # ---------------- perception / awareness (gap 2) --------------------
    R_front: float = 140.0      # m, detection radius when EV approaches from behind
    R_rear: float = 60.0        # m, detection radius for cars behind the EV
    delay_med: float = 1.0      # s, median driver reaction delay (lognormal)
    delay_sig: float = 0.45     # -, lognormal sigma of reaction delay
    p_noncomply: float = 0.05   # -, fraction of drivers that never yield [18]
    T_hold_lo: float = 1.5      # s, hold yielded position after EV passes
    T_hold_hi: float = 4.0      # s
    s_passed: float = 8.0       # m, EV counts as 'passed' this far beyond car
    v_passed: float = 0.3       # m/s, min EV recede speed for 'passed' (a
                                #   parked EV never 'passes' the traffic)
    side_right_bias: float = 1.2  # m, extra perceived room to the right (US rule)

    def copy(self, **over) -> "Params":
        return replace(self, **over)

    def to_dict(self) -> dict:
        return asdict(self)


#: Calibrated parameter overrides (experiments/run_calibration.py, 2026-07-14;
#: LHS 41 + Nelder-Mead 60, loss 0.517 vs 0.663 for the hand-set defaults;
#: full trace in out/calibration.json). Physically: the point field softens
#: and lengthens (the corridor term does the clearing), the corridor decay
#: tightens (~one lane width of clearance), anticipation starts earlier.
TUNED: dict = dict(
    A_ev=2.152, B_ev=21.406, A_c=3.758, B_c=0.748, T_react=15.944, gamma_c=0.720,
)


#: Calibrated against **real highD trajectories** (experiments/run_highd_calibration.py,
#: 2026-07-28; LHS 65 + Nelder-Mead, 126 evaluations, seeds 11-15; objective
#: emv.highd_fit.loss against out/highd_targets.json, block train:freeflow =
#: 27 recordings / 44 carriageways of German motorway traffic; loss 0.952 vs
#: 2.513 for TUNED on the same objective, i.e. -62 %; trace in
#: out/highd_calibration.json).
#:
#: What this is: the *host-traffic* block. highD contains no emergency vehicle,
#: so it can only identify how a lateral manoeuvre is executed and how precisely
#: drivers hold a lane. Every EV-response parameter therefore keeps its
#: behavioural-target / dashcam-GT calibration - TUNED above is inherited
#: unchanged, and TUNED itself is deliberately left alone because papers 1-4 and
#: PROJECT_LOG sec. 5 quote it.
#:
#: Physically: real drivers change lane a little more slowly and much less
#: sharply than the hand-set defaults assumed (v_lat_max 2.2 -> 1.57, a_pin
#: 1.4 -> 0.86, damping 1.0 -> 1.68 = overdamped), and they do not track the
#: lane centre at all - sigma_off 0.33 m reproduces the 0.27 m mean offset
#: measured on highD, which the pinned model put at 0.04 m. With these values
#: the median manoeuvre matches the data almost exactly (lane-change duration
#: 2.28 s vs 2.28 measured, centre-to-centre 3.48 vs 3.48, peak lateral speed
#: 0.96 vs 0.96). The tails do not - see docs/HIGHD_VALIDATION.md sec. 7.
#:
#: KNOWN NEXT STEP: a_pin fell by **84 %** (1.4 -> 0.2236 in the shipped fit (c);
#: the "38 %" this note used to quote was fit (a)'s 0.864 and understates the
#: shift by more than 4x), which lowers the depinning threshold
#: A_c*u > a_pin*(1 - relief*u), so the EV block should be re-fitted on this base
#: (stage 2) against the dashcam ground truth. Until then TUNED_HIGHD is
#: calibrated host traffic carrying an EV response fitted for a stiffer lane.
#: SUPERSEDED 2026-07-28 by the joint fit below, kept because
#: docs/HIGHD_VALIDATION.md sec. 7b uses it as the "moments-only" comparison and
#: out/highd_calibration_moments.json is its record:
#:   a_pin=0.864 zeta_lat=1.6843 v_lat_max=1.5705 a_lat_max=2.1501
#:   sigma_off=0.3277   (moment 0.954, shape 0.715)
#: It matched every median while leaving the distributions up to 4x too narrow,
#: and could not reach its own p90 targets (lane-change duration p90 was 4.5 s.d.
#: out) because a single shared lateral block cannot hold a median and a spread
#: at the same time.
#: (b) superseded 2026-07-29. Fitted while a_lat_max was still a single GLOBAL
#: clamp, which truncated the lateral-acceleration distribution (15 % of tracks
#: pinned to it, p90 = p95 = p99). Once a_lat_max is drawn per driver, (b)'s
#: theta scores moment 1.184 / shape 0.787 against (c)'s 0.343 / 0.550:
#:   a_pin=0.9352 zeta_lat=0.9952 v_lat_max=0.9839 a_lat_max=0.3333
#:   sigma_off=0.3924 het_lat=0.3994          (out/highd_calibration_b.json)
#:
#: (c) fitted 2026-07-29 with the corrected mechanism: a_lat_max and v_lat_max
#: share one per-driver "comfort envelope" factor, a_pin and zeta_lat have their
#: own, so the model carries two driver traits rather than one shared block.
#: Moment distance 0.343 is BELOW highD's own between-carriageway spread of
#: 0.408 - on the moment observables the model is within the data's own
#: variability. Shape distance 0.550 against a 0.178 bar is not yet there.
TUNED_HIGHD: dict = dict(
    TUNED,
    a_pin=0.2236, zeta_lat=2.4229, v_lat_max=1.3428, a_lat_max=0.7043,
    sigma_off=0.4479, het_lat=0.4257,
)


#: ==========================================================================
#: Two-stage calibration presets (2026-08-06).
#:
#: TUNED_HIGHD above is the 2026-07-29 fit. It has two properties this pair is
#: designed to replace: it was scored with a live emergency vehicle in the
#: scenario (so its lane-change kinematics came from EV-induced manoeuvres, the
#: model having no discretionary ones), and it was fitted on free-flow alone -
#: which drove `a_pin` down 84 % and left the congested regime worse than no
#: calibration at all.
#:
#:   TUNED_NORMAL     stage 1: ordinary German motorway traffic, **no emergency
#:                    vehicle anywhere in the scenario**, fitted on highD across
#:                    two densities (experiments/run_normal_calibration.py).
#:   TUNED_BLUELIGHT  stage 2: TUNED_NORMAL frozen, plus the four emergency-
#:                    vehicle repulsion parameters (A_ev, B_ev, A_c, B_c) fitted
#:                    against the dashcam ground truth
#:                    (experiments/run_ev_calibration.py).
#:
#: Both are None until their fit has run, and the accessors say so rather than
#: silently returning something else - a preset that quietly falls back to other
#: parameters is how a result gets misattributed.
#: ==========================================================================
#: Stage 1, fitted 2026-08-06 (`experiments/run_normal_calibration.py --block joint`,
#: LHS 60 + Nelder-Mead 70, 132 evaluations recorded, seeds 11-13, warm-started
#: from the `lon` and `latlc` blocks; record `out/normal_calibration_joint.json`).
#:
#: Objective: `highd_fit.loss_normal` with **no emergency vehicle in the
#: scenario**, summed over the free-flow (44 carriageways) and dense (27) target
#: blocks; congested (2) held out. 1.524 against 4.689 for TUNED_HIGHD and 5.327
#: for the dataclass defaults on the same objective. Collision-free in both
#: regimes. Per-regime moment distance 1.371 free-flow / 0.775 dense against
#: highD's own leave-one-carriageway-out spread of 0.491 / 0.387, i.e. the model
#: sits 2.0-2.8x the data's own variability on a 23-observable objective.
#:
#: What moved, and why it is not comparable with TUNED_HIGHD term by term: this
#: objective scores the *car-following* block (highD's headway and TTC
#: distributions, never scored before) and a *discretionary* lane-change
#: mechanism that did not exist before, so the lateral parameters are no longer
#: absorbing EV-induced manoeuvres. `a_pin` lands at 1.20 rather than
#: TUNED_HIGHD's 0.2236 - but see the KNOWN LIMITATION below before reading that
#: as "lane discipline is strong after all".
#:
#: KNOWN LIMITATION, measured and not tuned away: the lane-change **rate** comes
#: out 0.092 per veh-km against a measured 0.276, because `a_pin` sets both the
#: depinning threshold (how often a manoeuvre starts) and the restoring force
#: (how it is executed). `params.a_commit` separates the two and does raise the
#: rate to 0.18-0.23 with the manoeuvre shape preserved, but the collision counts
#: near that setting are non-monotonic at 3 seeds (5 / 0 / 15 contact frame-pairs
#: at a_commit 1.0 / 0.9 / 0.8), so the safety boundary is unresolved and no value
#: is shipped. `a_commit` belongs in the fitted vector next time, with more seeds.
#: Full evidence: docs/TERM_INFLUENCE.md sec. 6c.
TUNED_NORMAL: dict | None = dict(
    # --- car-following (N-lon)
    tau=0.6006, T_hw_lo=1.3479, T_hw_hi=1.7136, s0=2.6101, A_v=5.9237,
    Bx_v=6.7825, a_max=1.1603, b_comf=4.1777,
    # --- lateral block (N-lat)
    a_pin=1.2012, zeta_lat=0.902, v_lat_max=1.0501, a_lat_max=0.4158,
    sigma_off=0.3796, het_lat=0.3371, k_rho=0.9142,
    # --- discretionary lane changing (N-lc)
    A_pass=5.511, A_keep_right=9.2441, T_frust=3.7412, s_veto=27.7238,
)

#: Stage 2, fitted 2026-08-06 (`experiments/run_ev_calibration.py`, LHS 40 +
#: Nelder-Mead 45, seeds 11-13; record `out/ev_calibration.json`).
#:
#: **TUNED_NORMAL frozen, exactly four parameters fitted** - the repulsion the
#: emergency vehicle exerts (`A_ev`, `B_ev`) and the repulsion from its predicted
#: corridor (`A_c`, `B_c`). This is the requested experiment: model and tune
#: ordinary traffic first, then change only the parameters describing how other
#: cars are repelled by the emergency vehicle. Perception timing, urgency and the
#: longitudinal merge term keep their published values.
#:
#: Objective: behaviour-fingerprint distance to the pooled dashcam ground truth
#: (5 videos, 32 997 frames, 144 896 observations), rules-only labels, plus the
#: safety/comfort guards of `calibrate.loss`. 0.9418 against
#: 1.131 for the dataclass EV defaults and
#: 1.2692 for the published `TUNED` EV block on this
#: base - note the published block scores *worse* than the defaults here, which is
#: the predicted consequence of stage 1 having moved the lateral parameters, and
#: the reason a stage-2 re-fit was needed at all.
#:
#: Out of sample on fresh seeds [21, 22, 23]: **1.0676**, against a
#: leave-one-video-out acceptance bar of None - i.e. ~1.5x the ground
#: truth's own between-video spread. Collision-free, min TTC 3.05 s,
#: EV speed ratio 0.973, mean corridor clearance 123 m,
#: yield onset 105 m.
#:
#: Physically: the point field is *stronger and shorter* than the 2026-07-14 fit
#: (A_ev 5.9209 vs 2.152, B_ev 15.4181 vs 21.406) while the corridor
#: field is *weaker and broader* (A_c 2.3529 vs 3.758, B_c 2.8568 vs
#: 0.748). On the stage-1 base, lane discipline is stiffer (a_pin 1.20 vs 0.2236),
#: so a diffuse corridor push no longer suffices to depin a driver and the
#: emergency vehicle has to assert itself closer in.
TUNED_BLUELIGHT: dict | None = dict(
    TUNED_NORMAL,
    A_ev=5.9209, B_ev=15.4181, A_c=2.3529, B_c=2.8568,
)


def _preset(name: str, d: dict | None, script: str, **over) -> Params:
    if d is None:
        raise RuntimeError(
            f"{name} has not been calibrated yet - run {script} first, then "
            f"paste its theta into emv/params.py:{name}.")
    return Params().copy(**{**d, **over})


def normal_params(**over) -> Params:
    """Stage-1 parameters: ordinary traffic, emergency vehicle absent."""
    return _preset("TUNED_NORMAL", TUNED_NORMAL,
                   "experiments/run_normal_calibration.py", **over)


def bluelight_params(**over) -> Params:
    """Stage-2 parameters: TUNED_NORMAL plus the fitted EV repulsion block."""
    return _preset("TUNED_BLUELIGHT", TUNED_BLUELIGHT,
                   "experiments/run_ev_calibration.py", **over)


def highd_params(**over) -> Params:
    """Params calibrated against real highD trajectories (see TUNED_HIGHD).

    Mirrors tuned_params(): the preset on top of the dataclass defaults, keyword
    overrides last. Road geometry is deliberately *not* here - highD's 3.90 m
    median lane width is a property of the road, not the driver, and lives in
    emv.scenarios.make_highd_like.
    """
    d = dict(TUNED_HIGHD)
    d.update(over)
    return Params().copy(**d)


def tuned_params(**over) -> Params:
    d = dict(TUNED)
    d.update(over)
    return Params().copy(**d)
