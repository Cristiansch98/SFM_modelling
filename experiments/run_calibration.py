"""Calibrate (A_ev, B_ev, A_c, B_c, T_react, gamma_c) against behavioural
targets. Writes out/calibration.json. See emv/calibrate.py for the loss."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from emv.calibrate import run_calibration

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lhs", type=int, default=40)
    ap.add_argument("--nm", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11])
    args = ap.parse_args()
    from emv.runlog import RunLogger
    t0 = time.time()
    with RunLogger("calibration",
                   note=f"lhs={args.lhs} nm={args.nm} seeds={args.seeds}") as rl:
        res = run_calibration(n_lhs=args.lhs, n_nm=args.nm, seeds=tuple(args.seeds),
                              out_path=os.path.join(os.path.dirname(__file__), "..",
                                                    "out", "calibration.json"))
        rl.finish(metrics=dict(loss=res["loss"], loss_default=res["loss_default"],
                               **res["theta"]),
                  outputs=["out/calibration.json"])
    print(f"wall time {time.time()-t0:.0f}s")
