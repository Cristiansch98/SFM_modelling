# physics_behaviour_EMV

**A physics-informed repulsion-force framework for simulating how traffic
yields to an emergency vehicle** — social-force core, perception gating,
depinning analysis, calibration machinery, and a hybrid SUMO coupling.

The repository contains source code, tests, mathematical documentation, an
interactive browser lab, and the Unreal Engine project source. Generated
figures, videos, calibration outputs, logs, papers, local Python environments,
and Unreal build caches are deliberately excluded; all results can be regenerated
from the supplied experiments.

Every car is a Newtonian particle under superimposed specific forces
(Helbing-style [2]): a **driving force** $(v^0\hat x - \vec v)/\tau$ holds
normal flow; an **anisotropic exponential field**
$A e^{(r-d)/B}\hat n\,[\lambda + (1{-}\lambda)\frac{1+\cos\varphi}{2}]$
radiates from the EV; and the key term, a **corridor repulsion**, repels from
the EV's *predicted path* (its route projected 8 s ahead) — the lateral
component pushes cars out of the corridor, the longitudinal one makes them
"pull over and slow down". An IDM bound keeps safety authoritative, a
washboard lane-keeping potential makes lane changes a **depinning
transition** with an analytic threshold, and a per-driver **perception state
machine** (detection radius, lognormal reaction delay, compliance, two-channel
urgency) decides *when* each driver feels the field.

Full math + analysis: **[docs/FRAMEWORK.md](docs/FRAMEWORK.md)**.
Literature anchors [1]–[27]: `literature_review_ev_repulsion_models.md`.
Interactive lab (live sim + results): published Claude artifact *EMV
Yield-Field Lab* (`artifact/emv_lab.html`).

## Headline results (calibrated, mean ± s.d. over 10 seeds per condition)

| | no yielding | force model |
|---|---|---|
| Motorway overtake — EV mean speed | 100 km/h (77±1 %) | **123±10 km/h (95±8 %)** |
| Rescue lane in jam — EV mean speed | 10 km/h (31±2 %) | **31±1 km/h (95±3 %)**, corridor opens in 5.8±4.4 s |
| Collisions (all 40 runs) / min TTC | 0 / 3.4±0.4 s | **0** / 1.9±0.6 s |
| Yield onset ahead of EV (pooled) | — | 102±33 m / 109±20 m (empirical band 50–150 m [18, 20]) |

Full per-seed data: `out/multiseed.json` (`python experiments/run_multiseed.py`).

**Drivable UE5.8 demo** — you drive the ambulance, and you pick *who decides how
traffic reacts*: `python experiments/start_demo.py` or double-click
`START_DEMO.bat`. One command starts the NPC brain, opens a live top-down 2D view
of the same scene, waits until the bridge socket is accepting, *then* launches
Unreal. Press Play in the editor, or pass `--game` to drop straight into driving.
Controls: W/S throttle+brake, A/D steer, Space handbrake, L siren, C camera.

| `--mode` | NPCs driven by | |
|---|---|---|
| `physics` *(default)* | this project's force model | `emv/ue/server.py` |
| `bluelight` | SUMO's native rescue-lane device, reaction distance 100 m | `emv/ue/sumo_server.py` |
| `none` | SUMO with no yielding behaviour — the control condition | `emv/ue/sumo_server.py` |

Same road, same density, same car — the only difference is the yielding logic,
so the modes are directly comparable from the driver's seat. Driving a scripted
EV at 119 km/h down the rescue-gap line for 45 s gives mean corridor clearance
ahead of **18.5 m** (`none`), **18.7 m** (`bluelight`) and **43.6 m**
(`physics`): SUMO's device is built for jams and does very little in free-flowing
motorway traffic, which is precisely the regime this project's corridor term
targets. (The device's other half — the EV's own special rights — does not apply
here, because the EV is you.)

Add `--no-view` for a headless brain, `--no-ue` to attach Unreal yourself,
`--hz` to set the brain tick rate (default 50). Wire protocol and UE-side setup:
**[docs/UE_BRIDGE.md](docs/UE_BRIDGE.md)**.

**Live parameter lab** — the endless version of the animations, with the eight
parameters that decide the outcome on sliders you drag while it runs:
`python experiments/live_lab.py` (`--scenario jam`, `--preset default|tuned|game`).
The sliders mutate the same `Params` object the force model reads every step, so
a change lands within one integration step; the readout shows the depinning
verdict live ($A_c$ vs $a^{eff}_{pin}$ and the resulting $d^*$), which makes the
phase transition something you can drag across. `g` records what you are watching
to `out/anim_live_lab.gif`.

**Playable SUMO demo** (needs SUMO + sumo-gui): `python experiments/play_sumo.py`
or double-click `PLAY_SUMO_DEMO.bat` — opens sumo-gui with the force bridge live,
camera locked to the EV, vehicles recoloured by awareness state
(`--mode bluelight|none` for the reference behaviours).

* **Parameter sensitivity + behavioural realism** (`python experiments/run_param_stats.py`,
  ~30 min → `out/fig_param_*.png`, `out/param_stats_tables.md`): free-flow EV
  progress is driven by $A_c$ (ρ = +0.59, saturating by $A_c$ ≈ 2.4) and
  $T_{react}$; the jam is driven by non-compliance (ρ = −0.81). Measured
  iso-performance contours over the $(a_{pin}, A_c)$ plane run **parallel to the
  analytic escape boundary**. Memo:
  [research artifact](https://claude.ai/code/artifact/c690070b-258e-4b5a-89bc-9d618fab147a).
  *Superseded 2026-07-28*: that comparison used cited anchors and a lane-change
  window that measured only the arrival half of a manoeuvre, so the "free-flow is
  ~4× too fast" reading was largely an artefact — see below and
  `docs/HIGHD_VALIDATION.md` §5.
* **Calibrated and validated on real trajectories** (highD, 60 recordings,
  110,516 vehicles, 44,476 veh-km): with model and data measured by *the same
  code*, the fitted median manoeuvre matches the real one almost exactly —
  lane-change duration 2.28 s vs 2.28 measured, centre-to-centre 3.48 vs 3.48,
  peak lateral speed 0.96 vs 0.96 m/s — at an objective of 0.95 against 2.51 for
  the previous parameters. The tails do not match, and the lane-change *rate*
  cannot (no discretionary lane-change mechanism). `emv/params.py:TUNED_HIGHD`;
  method and limits in `docs/HIGHD_VALIDATION.md`.
* **Depinning validated**: predicted cleared half-width
  $d^* = w_{need} + B_c\ln(A_c/a^{eff}_{pin}) \approx 3.8$ m vs measured
  3.8–4.2 m; the phase diagram's corridor-failure boundary follows
  $A_c \approx a_{pin}$ across densities.
* **Calibration** (LHS + Nelder–Mead, scipy-free): loss −22 % vs hand-set
  defaults; tuned values baked into `emv/params.py:TUNED`.
* **SUMO 1.26 coupling** (same force/perception code via TraCI): collision-free,
  +10 % EV speed over no-yield with a 73 m corridor; SUMO's native bluelight
  device remains faster (hard-coded special rights) — comparison in
  `out/fig_sumo_compare.png`.

## Quickstart

```bash
pip install -r requirements.txt        # numpy, matplotlib, pillow
python tests/smoke.py                  # ~20 s: model vs baselines, asserts
python experiments/live_lab.py         # real-time window, parameters on sliders
python experiments/run_all.py          # figures + GIFs + out/summary.json (~5 min)
python experiments/run_calibration.py  # re-calibrate (~7 min)
python experiments/run_sumo_compare.py # needs SUMO_HOME; three-way comparison
```

```python
from emv.params import tuned_params
from emv.scenarios import make_overtake, make_jam
from emv.metrics import evaluate, summary_line

sim = make_jam(seed=3, p=tuned_params())      # or make_overtake(...)
hist = sim.run(70.0)                          # History of trajectories
print(summary_line(evaluate(hist)))
```

## State & logging

Everything that runs writes a structured entry to the append-only ledger
`logs/runs.jsonl` (via `emv/runlog.py`); `logs/LATEST.md` shows the recent
tail. The complete project state — results of record, environment quirks,
decisions, known failure modes — lives in **`PROJECT_LOG.md`** (new sessions
start there; `CLAUDE.md` enforces it).

## Layout

```
emv/                 the framework
  params.py            all parameters, units, literature anchors, TUNED values
  forces.py            drive · car-car (elliptic SFM + IDM bound) · EV field · corridor
  perception.py        awareness state machine + two-channel urgency
  lanechange.py        discretionary lane changing: latched commitment when the
                       incentive beats the depinning threshold (off by default)
  terms.py             per-term influence: registry, ablations, force budget
  road.py, state.py    geometry (corridor projection), state arrays
  dynamics.py          constraint projection + semi-implicit Euler
  simulate.py          runner + trajectory History
  scenarios.py         motorway overtake · rescue lane in jam (+ baselines)
  metrics.py           progress, safety (TTC), comfort, timing, disruption
  calibrate.py         behavioural-target loss, LHS, Nelder-Mead
  viz.py               animations, force field, space-time, bow wave, phase diagram
  sumo/                scenario generator + hybrid TraCI bridge (slowDown/sublane)
experiments/         run_all · run_calibration · run_sumo_compare
                     run_normal_calibration (stage 1: ordinary traffic vs highD,
                     emergency vehicle absent) · run_ev_calibration (stage 2:
                     only A_ev, B_ev, A_c, B_c) · run_term_influence
docs/FRAMEWORK.md    full mathematical specification & analysis
docs/TERM_INFLUENCE.md  two-stage calibration + what each force term contributes
artifact/            interactive lab (published artifact)
out/                 generated figures, animations, JSON results
```
