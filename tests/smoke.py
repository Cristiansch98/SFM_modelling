"""Quick sanity run: model vs baseline in both scenarios."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from emv.scenarios import make_overtake, make_jam
from emv.metrics import evaluate, summary_line

t0 = time.time()
runs = []
for build, kw, T, stop in [
    (make_overtake, dict(seed=1, yielding=True), 75.0, 2300.0),
    (make_overtake, dict(seed=1, yielding=False), 75.0, 2300.0),
    (make_jam, dict(seed=3, yielding=True), 70.0, 700.0),
    (make_jam, dict(seed=3, yielding=False), 70.0, 700.0),
]:
    sim = build(**kw)
    n = sim.st.n
    h = sim.run(T, stop_when_ev_x=stop)
    m = evaluate(h)
    runs.append((h, m))
    assert np.isfinite(h.x).all() and np.isfinite(h.vx).all(), "NaN in trajectories"
    print(f"N={n:4d} frames={h.n_frames:4d}  " + summary_line(m), flush=True)

print(f"\ntotal wall time {time.time()-t0:.1f}s")
(m_o, m_ob, m_j, m_jb) = [r[1] for r in runs]
print(f"overtake EV speed: model {m_o['ev_mean_speed']:.1f} vs baseline {m_ob['ev_mean_speed']:.1f} m/s")
print(f"jam      EV speed: model {m_j['ev_mean_speed']:.1f} vs baseline {m_jb['ev_mean_speed']:.1f} m/s")
assert m_o["ev_mean_speed"] > m_ob["ev_mean_speed"] + 1.0, "yielding should speed up EV (overtake)"
assert m_j["ev_mean_speed"] > m_jb["ev_mean_speed"] + 1.0, "yielding should speed up EV (jam)"
assert m_o["collisions"] == 0 and m_j["collisions"] == 0, "collisions in model runs"
from emv.runlog import log_run
log_run("smoke", metrics=dict(overtake_v=m_o["ev_mean_speed"], jam_v=m_j["ev_mean_speed"],
                              base_o=m_ob["ev_mean_speed"], base_j=m_jb["ev_mean_speed"]))
print("SMOKE OK")
