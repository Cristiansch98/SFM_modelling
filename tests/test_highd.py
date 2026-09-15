"""Offline checks for the highD calibration work (no dataset required).

Guards three things, in order of how expensive they would be to discover late:

1. **The existing record still reproduces.** `TUNED` is untouched, no
   pre-existing `Params` default changed value, and `lane_change_events` /
   `behaviour_stats` return exactly what they returned before the shared
   `lane_change_from_track` refactor. Papers 1-4 and PROJECT_LOG sec. 5 quote
   these numbers.
2. **The two measurement windows behave as documented** - the originally
   shipped one covers the arrival half of a manoeuvre, the `_full` one the whole
   thing (see emv/metrics.py:lane_change_from_track).
3. **The highD reader's coordinate contract**, on synthetic tracks: bbox corner
   to centre, direction flip, marking-based lane indexing, and a hand-built lane
   change of known duration.

Also asserts the package never imports pandas or scipy - the Windows box has
neither, and emv/highd.py is numpy-only on purpose.

    python tests/test_highd.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv import highd as hd
from emv import metrics
from emv.params import Params, TUNED, TUNED_HIGHD, highd_params, tuned_params
from emv.scenarios import make_highd_like, make_jam, make_overtake

FAIL = []


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAIL.append(msg)


# ---------------------------------------------------------------- 1. record
print("1. existing record")
check(TUNED == dict(A_ev=2.152, B_ev=21.406, A_c=3.758, B_c=0.748,
                    T_react=15.944, gamma_c=0.720),
      "params.TUNED is byte-identical to the calibration of record")

#: Every Params default as of 2026-07-28, before the highD work. sigma_off is
#: the only field added since, and it defaults to 0.0 = mechanism off.
NEW_FIELDS = {"sigma_off"}
p, t = Params(), tuned_params()
check(p.sigma_off == 0.0, "sigma_off defaults to 0 (preferred-offset mechanism off)")
check(p.a_lat_max == 3.0 and p.v_lat_max == 2.2 and p.a_pin == 1.4
      and p.zeta_lat == 1.0,
      "lateral defaults unchanged (a_lat_max 3.0, v_lat_max 2.2, a_pin 1.4)")
check(t.A_c == 3.758 and t.T_react == 15.944,
      "tuned_params() still returns the calibrated EV block")

#: Reference values captured from the pre-refactor code on 2026-07-28.
REF = {
    "overtake_seed1": dict(
        n_events=2, lc_dur_med=1.05, lc_peak_vy_med=1.2136828291892114,
        peak_alat_p90=0.7396643380033648),
    "jam_seed3": dict(n_events=12, lc_dur_med=2.5),
}
h_o = make_overtake(seed=1, p=Params()).run(20.0, rec_dt=0.12)
h_j = make_jam(seed=3, p=Params()).run(20.0, rec_dt=0.12)
m_o, m_j = metrics.behaviour_stats(h_o), metrics.behaviour_stats(h_j)
lc_o, lc_j = metrics.lane_change_events(h_o), metrics.lane_change_events(h_j)
check(len(lc_o) == REF["overtake_seed1"]["n_events"]
      and len(lc_j) == REF["jam_seed3"]["n_events"],
      "lane_change_events finds the same events after the refactor")
for key, ref in REF["overtake_seed1"].items():
    if key == "n_events":
        continue
    check(abs(m_o[key] - ref) < 1e-12, f"behaviour_stats[{key}] bit-identical")
check(abs(m_j["lc_dur_med"] - REF["jam_seed3"]["lc_dur_med"]) < 1e-12,
      "behaviour_stats[lc_dur_med] bit-identical (jam)")

# ------------------------------------------------------- 2. the two windows
print("\n2. measurement windows")
e = lc_o[0]
check(e["duration_full"] > e["duration"],
      f"full window is longer ({e['duration_full']:.2f} s vs {e['duration']:.2f} s)")
w = h_o.road.lane_width
check(abs(e["dy_full"] - (w - 2 * 0.6)) < 0.35,
      f"dy_full ~ lane width minus the two settle bands "
      f"({e['dy_full']:.2f} m, lane {w} m)")
check(e["dy"] < 0.75 * e["dy_full"],
      f"the shipped dy is the arrival half only ({e['dy']:.2f} m)")
check(all(k in m_o for k in ("lc_dur_full_med", "lc_c2c_med",
                             "lane_offset_mean", "lc_truncated_share")),
      "behaviour_stats exposes the full-window and lane-keeping keys")

# ------------------------------------------------- 3. highD reader contract
print("\n3. highD coordinate contract (synthetic)")
marks = np.array([0.0, 3.5, 7.0, 10.5])
dt = 0.12
n = 120
t = np.arange(n) * dt
#: a clean 4 s lane change from lane 0 centre (1.75) to lane 1 centre (5.25)
y = np.full(n, 1.75)
k0, k1 = 40, 40 + int(round(4.0 / dt))
y[k0:k1] = np.linspace(1.75, 5.25, k1 - k0)
y[k1:] = 5.25
vy = np.gradient(y, dt)
evs = metrics.lane_change_from_track(t, y, vy, W=1.8, marks=marks, veh=0)
check(len(evs) == 1, f"one lane change detected ({len(evs)})")
if evs:
    ev = evs[0]
    check(ev["from_lane"] == 0 and ev["to_lane"] == 1, "lane indices from markings")
    # body straddles the 3.5 m marking over |y-3.5| < 0.9 m, i.e. 1.8 m of a
    # 3.5 m traverse taking 4 s
    expect = 4.0 * (1.8 / 3.5)
    check(abs(ev["duration_full"] - expect) < 3 * dt,
          f"two-lane-occupancy duration {ev['duration_full']:.2f} s "
          f"vs {expect:.2f} s expected")
    check(abs(ev["dy_full"] - (3.5 - 2 * 0.6)) < 0.3,
          f"dy_full {ev['dy_full']:.2f} m")

check(hd.classify_regime(31.0, 900.0) == "freeflow"
      and hd.classify_regime(12.0, 900.0) == "congested"
      and hd.classify_regime(25.0, 1700.0) == "dense",
      "regime classification thresholds")

# ------------------------------------------------------------ 4. new preset
print("\n4. highD preset and scenario")
ph = highd_params()
check(ph.A_c == TUNED["A_c"] and ph.T_react == TUNED["T_react"],
      "TUNED_HIGHD inherits the EV block from TUNED unchanged")
check(ph.sigma_off > 0 and ph.a_pin < Params().a_pin,
      "TUNED_HIGHD carries the fitted lateral block")
check(set(TUNED).issubset(TUNED_HIGHD), "TUNED_HIGHD is a superset of TUNED")

sim = make_highd_like(seed=11, p=ph, regime="freeflow", road_len=1200.0)
check(sim.st.n > 10, f"highD scenario builds ({sim.st.n} vehicles)")
check((sim.st.L > 9).any(), "the fleet contains trucks")
check(abs(sim.road.lane_width - 3.9) < 0.3,
      f"lane width from measurement ({sim.road.lane_width:.2f} m)")
inert = make_highd_like(seed=11, p=ph, regime="freeflow", road_len=1200.0,
                        ev_inert=True)
check(inert.st.x[inert.st.ev] > 1200.0 and inert.st.v0[inert.st.ev] == 0.0,
      "ev_inert parks the EV outside pair_cutoff for an EV-free control")

# ------------------------------------------- 4b. shape metrics + heterogeneity
print("\n4b. distribution metrics and per-driver heterogeneity")
from emv import highd_fit as hfit

rng = np.random.default_rng(0)
a = rng.normal(0.0, 1.0, 20000)
check(abs(hfit.wasserstein1(a, a)) < 1e-9, "W1 of a sample with itself is 0")
check(abs(hfit.wasserstein1(a, a + 1.0) - 1.0) < 0.02,
      "W1 recovers a pure location shift of 1.0")
check(abs(hfit.dispersion_ratio(a, 2.0 * a) - 0.5) < 0.02,
      "IQR ratio recovers a factor-2 spread difference")
check(abs(hfit.wasserstein1(a, 2.0 * a)) > 0.5,
      "W1 is non-zero for equal-median, different-spread samples "
      "(the case percentile matching misses)")

sim0 = make_highd_like(seed=5, p=highd_params().copy(het_lat=0.0),
                       road_len=1200.0)
sim1 = make_highd_like(seed=5, p=highd_params().copy(het_lat=0.3),
                       road_len=1200.0)
check(sim0.st.a_pin is None,
      "het_lat=0 leaves the per-driver lateral arrays unset (mechanism off)")
check(sim1.st.a_pin is not None and float(np.std(sim1.st.a_pin[1:])) > 0,
      "het_lat>0 gives each driver its own lateral block")
check(sim1.st.a_lat_max is not None and float(np.std(sim1.st.a_lat_max[1:])) > 0,
      "a_lat_max is drawn per driver too - a single global clamp truncated the "
      "lateral-acceleration distribution instead of reproducing its tail")
check(abs(float(sim1.st.a_pin[0]) - highd_params().a_pin) < 1e-12
      and abs(float(sim1.st.a_lat_max[0]) - highd_params().a_lat_max) < 1e-12,
      "the EV keeps the global lateral block")
check(abs(float(np.median(sim1.st.a_pin[1:])) / highd_params().a_pin - 1.0) < 0.25,
      "per-driver draws keep the fitted value as their median")
# the speed and acceleration ceilings are one trait, so they must covary; drawn
# independently they would produce incoherent drivers
r = np.corrcoef(sim1.st.v_lat_max[1:], sim1.st.a_lat_max[1:])[0, 1]
check(r > 0.99, f"v_lat_max and a_lat_max share one per-driver factor (r={r:.3f})")
r2 = abs(np.corrcoef(sim1.st.a_pin[1:], sim1.st.v_lat_max[1:])[0, 1])
check(r2 < 0.4, f"lane-keeping discipline is a separate trait (|r|={r2:.3f})")

# ------------------------------------------------------------ 5. no pandas
print("\n5. dependency guard")
import emv.empirical  # noqa: F401
import emv.highd_fit  # noqa: F401
check("pandas" not in sys.modules, "emv never imports pandas")
check("scipy" not in sys.modules, "emv never imports scipy")

lit = emv.empirical.figure_bands()
check(abs(lit["lc_dur_med"]["mid"] - 4.01) < 1e-12
      and abs(lit["lc_per_veh_km"]["mid"] - 5600.0 / 45000.0) < 1e-12,
      "empirical.figure_bands() reproduces the cited anchors exactly")

print("\n" + ("FAILED: " + "; ".join(FAIL) if FAIL else "ALL CHECKS PASS"))
sys.exit(1 if FAIL else 0)
