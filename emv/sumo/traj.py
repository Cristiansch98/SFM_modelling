"""Compact full-trajectory recording for SUMO bridge runs.

The bridge (record_traj=True) captures, at every 0.1 s step of the EV-active
window, the id/x/y/speed/awareness of every vehicle on the edge plus the
corridor centre line. Frames are ragged (vehicles enter and leave), so the
npz stores one concatenated array per field with per-frame offsets:

    t (F,)  corr_y (F,)  ptr (F+1,)          frame f = slice(ptr[f], ptr[f+1])
    idx (K,) x (K,) y (K,) v (K,) state (K,)  K = total observations
    veh_ids (N,) L (N,) W (N,)                idx values index these tables
    meta                                      json: mode/seed/geometry/ev_idx

`state` uses the emv.state awareness codes (UNAWARE..HOLD); modes without a
perception model record UNAWARE everywhere (rule mode maps its scripted
active flag to YIELDING so videos can show activation).
"""
import json
import os
import numpy as np


def blank_traj() -> dict:
    """Mutable accumulator the bridge appends to (one entry per step)."""
    return dict(t=[], corr_y=[], frames=[], vindex={}, veh_ids=[], L=[], W=[])


def save_traj(path: str, traj: dict, meta: dict) -> str:
    """Flatten the ragged accumulator and write a single .npz."""
    counts = [f[0].size for f in traj["frames"]]
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    cat = lambda i: np.concatenate([f[i] for f in traj["frames"]])
    meta = dict(meta, ev_idx=traj["vindex"].get("EV", -1))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        t=np.asarray(traj["t"], np.float32),
        corr_y=np.asarray(traj["corr_y"], np.float32),
        ptr=ptr, idx=cat(0).astype(np.int32),
        x=cat(1).astype(np.float32), y=cat(2).astype(np.float32),
        v=cat(3).astype(np.float32), state=cat(4).astype(np.int8),
        veh_ids=np.asarray(traj["veh_ids"]),
        L=np.asarray(traj["L"], np.float32),
        W=np.asarray(traj["W"], np.float32),
        meta=np.asarray(json.dumps(meta)))
    return path


def load_traj(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    tr = {k: z[k] for k in ("t", "corr_y", "ptr", "idx", "x", "y", "v",
                            "state", "veh_ids", "L", "W")}
    tr["meta"] = json.loads(str(z["meta"]))
    tr["ev"] = int(tr["meta"]["ev_idx"])
    return tr


def frame(tr: dict, f: int) -> dict:
    """One frame as a dict of aligned arrays (idx/x/y/v/state)."""
    s = slice(int(tr["ptr"][f]), int(tr["ptr"][f + 1]))
    return dict(idx=tr["idx"][s], x=tr["x"][s], y=tr["y"][s],
                v=tr["v"][s], state=tr["state"][s])


def dense(tr: dict, stride: int = 1):
    """Vehicle-major dense matrices X/Y/V (N, Fs), NaN where absent, plus the
    strided time vector - convenient for line-based trajectory plots."""
    fs = np.arange(0, tr["t"].size, stride)
    n = tr["veh_ids"].size
    X = np.full((n, fs.size), np.nan, np.float32)
    Y = np.full((n, fs.size), np.nan, np.float32)
    V = np.full((n, fs.size), np.nan, np.float32)
    for c, f in enumerate(fs):
        s = slice(int(tr["ptr"][f]), int(tr["ptr"][f + 1]))
        i = tr["idx"][s]
        X[i, c] = tr["x"][s]
        Y[i, c] = tr["y"][s]
        V[i, c] = tr["v"][s]
    return tr["t"][fs], X, Y, V
