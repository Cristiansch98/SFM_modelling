"""Diagnose why the jam corridor is not opening: dump per-car force balance
near the EV mid-run."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from emv.scenarios import make_jam
from emv.state import STATE_NAMES

sim = make_jam(seed=3)
for _ in range(int(40.0 / sim.p.dt)):
    sim.step()

st = sim.st
corr, dg = sim.snapshot_forces()
ev = st.ev
s = st.x - st.x[ev]
sel = np.flatnonzero((s > -15) & (s < 70) & ~st.is_ev)
sel = sel[np.argsort(s[sel])]

print(f"t=40s  EV: x={st.x[ev]:.1f} y={st.y[ev]:.2f} vx={st.vx[ev]:.2f}  "
      f"corr y_c={corr.y_c:.2f} L={corr.L:.0f}")
print(f"EV idm bound = {dg['idm'][ev]:.2f}")
hdr = f"{'s':>6} {'y':>6} {'vx':>5} {'vy':>6} {'st':>8} {'side':>4} {'u':>5} " \
      f"{'Fc_lat':>7} {'Fsfm_y':>7} {'Flane':>7} {'Fev_y':>6} {'ay_sum':>7} {'idm':>7}"
print(hdr)
for i in sel:
    ay_sum = dg['corr'][1][i] + dg['sfm'][1][i] + dg['lane'][1][i] + dg['ev'][1][i]
    idm = dg['idm'][i]
    print(f"{s[i]:6.1f} {st.y[i]:6.2f} {st.vx[i]:5.2f} {st.vy[i]:6.2f} "
          f"{STATE_NAMES[int(st.aware[i])]:>8} {st.side[i]:4d} {st.u[i]:5.2f} "
          f"{dg['corr'][1][i]:7.2f} {dg['sfm'][1][i]:7.2f} {dg['lane'][1][i]:7.2f} "
          f"{dg['ev'][1][i]:6.2f} {ay_sum:7.2f} {min(idm, 99.0):7.2f}")

# lateral occupancy histogram ahead of EV
ahead = (s > 0) & (s < 60) & ~st.is_ev
print("\nlateral positions ahead (60m):",
      np.array2string(np.sort(st.y[ahead]), precision=2))
print("w_need =", 0.5 * (st.W[ev] + 1.8) + sim.p.margin_c)
