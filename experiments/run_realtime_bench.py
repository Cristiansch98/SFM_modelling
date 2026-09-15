"""How much wall time one simulation step costs -> out/realtime_bench.json.

    python experiments/run_realtime_bench.py [--reps 400] [--quick]

The claim this repo makes for the training simulator is that the model is cheap
enough to run *inside* the frame loop of a first-person simulator rather than in
a separate process. That claim needs a number, so this measures one:

  * wall time per `dynamics.step` call over a range of fleet sizes,
  * the real-time factor  (simulated seconds) / (wall seconds)  at dt = 0.05 s,
  * the share of a 60 Hz frame budget (16.67 ms) one physics update takes,
  * how the cost scales - the force evaluation is O(N^2) over the pair cutoff,
    so the fit of a quadratic tells us where the linear term stops dominating.

It is deliberately a *bare* measurement: no rendering, no history recording, no
perception sampling beyond what a step needs, single thread, and the timed
region is exactly what an engine would call per tick. Reported as a lower bound
on cost, not as a benchmark of anything else's implementation.

Self-logs kind='bench'.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv import scenarios
from emv.params import Params, TUNED_BLUELIGHT
from emv.runlog import log_run

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
FRAME_60 = 1000.0 / 60.0                      # ms, one frame at 60 Hz


def _world(n_veh, seed=7):
    """A highD-like carriageway holding about `n_veh` vehicles, EV included.

    Fleet size is set by the *measured* density (8.4 veh/km/lane over 3 lanes),
    so the road length is what varies - keeping the traffic itself realistic
    instead of packing an unrealistic number of cars into a fixed stretch.
    """
    per_km = 3 * 8.4
    sim = scenarios.make_highd_like(p=Params(**TUNED_BLUELIGHT), seed=seed,
                                    road_len=1000.0 * n_veh / per_km)
    return sim


def _time_steps(sim, reps):
    """Wall time of one full update, in milliseconds, sorted ascending.

    One update = corridor rebuild + perception + lane-change decisions + all
    force terms + integration, i.e. exactly what an engine tick has to pay for
    (`Sim.step`). No recording, no rendering.
    """
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sim.step()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    reps = 60 if a.quick else a.reps
    sizes = [20, 40, 60, 80, 120, 160, 240] if not a.quick else [40, 120]

    rows = []
    for n in sizes:
        sim = _world(n)
        _time_steps(sim, 25)                          # warm-up: let numpy settle
        s = _time_steps(sim, reps)
        med = statistics.median(s)
        n_act = int(sim.st.n)
        row = dict(n_veh=n_act, ms_med=med, ms_p95=s[int(0.95 * len(s)) - 1],
                   ms_min=s[0], rt_factor=(sim.p.dt * 1e3) / med,
                   frame_share_60=med / FRAME_60, us_per_veh=1e3 * med / n_act)
        rows.append(row)
        print(f"N={row['n_veh']:4d}  {med:6.3f} ms/step  "
              f"p95 {row['ms_p95']:6.3f}  x{row['rt_factor']:6.1f} real time  "
              f"{100 * row['frame_share_60']:5.2f} % of a 60 Hz frame  "
              f"{row['us_per_veh']:5.1f} us/veh", flush=True)

    # cost model: quadratic in N (pairwise forces inside the cutoff)
    n = np.array([r["n_veh"] for r in rows], float)
    ms = np.array([r["ms_med"] for r in rows], float)
    c2, c1, c0 = np.polyfit(n, ms, 2)
    # fleet that still fits one 60 Hz frame, from that fit
    roots = np.roots([c2, c1, c0 - FRAME_60])
    n_frame = float(max(r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0))

    rec = dict(rows=rows, fit=dict(c2=float(c2), c1=float(c1), c0=float(c0)),
               n_veh_one_frame_60hz=n_frame, dt=0.05, frame_ms_60=FRAME_60,
               reps=reps, note="single thread, no rendering, no history")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "realtime_bench.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)

    print(f"\ncost(N) = {c2:.3e} N^2 + {c1:.3e} N + {c0:.3e}  ms")
    print(f"one 60 Hz frame holds ~{n_frame:.0f} vehicles of physics")
    log_run("bench", params=dict(sizes=sizes, reps=reps),
            metrics=dict(ms_med_120=[r["ms_med"] for r in rows][len(rows) // 2],
                         rt_factor_min=min(r["rt_factor"] for r in rows),
                         n_veh_one_frame_60hz=n_frame),
            outputs=["out/realtime_bench.json"],
            note="real-time cost of one model update vs fleet size")


if __name__ == "__main__":
    main()
