"""Single source of truth for the empirical reference constants.

Before this module the same anchors were hard-coded in five independent places
(`experiments/make_param_stats_figs.py`, `experiments/make_param_paper_extra.py`,
`artifact/make_param_memo.py`, `paper/ieee_emv_param_sensitivity.tex`,
`PROJECT_LOG.md`/`README.md`) and had already drifted apart - the naturalistic
deceleration anchor read 2.85 in one file and 3.4 in another.

Two namespaces, kept apart deliberately:

* `literature()` - values **cited** from other work and other datasets (NGSIM,
  AASHTO, comfort studies, and the lane-change rate as *printed* in the highD
  paper). Secondary references: a different road, country and era. They are
  never silently replaced by our own measurement.
* `measured()` - values **measured** on highD by this repo's own extractor
  (`emv/highd.py` -> `out/highd_targets.json`), under definitions identical to
  the model's own metrics. These are the calibration targets and validation
  references.

`measured()` degrades gracefully: on a machine without the dataset it returns
`{}` and every figure script still works off `literature()`.

On the lane-change rate specifically - both numbers are right about different
things, and both are reported:

* **0.124/veh-km** = 5600 *complete* lane changes / 45 000 veh-km as printed in
  the highD paper, where a manoeuvre must lie entirely inside the ~420 m field of
  view and pass their extraction filter.
* **0.314/veh-km** = 13 949 `tracksMeta.numLaneChanges` / 44 476 veh-km over the
  released 60 recordings - every lane-change event a track records.

The 2.5x ratio is a definition gap, not an error in either source, which is why
the extractor also emits `lc_per_veh_km_complete` as the like-for-like third
number.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGETS_PATH = os.path.join(ROOT, "out", "highd_targets.json")

#: Cited anchors. `lo`/`hi` bound a band, `mid` marks a point value, `note` is
#: the label figures and tables print. Values are byte-identical to the `LIT`
#: dict they replace, so migrating a consumer changes nothing.
LITERATURE: dict = {
    "lc_dur_med": dict(
        lo=4.01 - 2.31, hi=4.01 + 2.31, mid=4.01,
        note="NGSIM, two-lane occupancy\n4.01 +/- 2.31 s (Thiemann+ 2008)",
        cite="thiemann08",
        dataset="NGSIM US-101/I-80, 2005 - congested US freeway"),
    "lc_per_veh_km": dict(
        lo=None, hi=None, mid=5600.0 / 45000.0,
        note="highD: 5600 lane changes\nper 45 000 veh-km",
        cite="krajewski18",
        dataset="highD as printed; complete manoeuvres only"),
    "lc_peak_vy_med": dict(
        lo=None, hi=None, mid=None,
        note="no directly comparable\nnaturalistic statistic"),
    "yield_peak_decel_med": dict(
        lo=1.0, hi=2.85, mid=2.85,
        note="naturalistic braking:\n99th pct 2.85 m/s2;\ncomfort 1.0-1.3 m/s2",
        caveat="the 2.85 figure could not be verified in the source it was "
               "attributed to (PROJECT_LOG sec. 10); paper 4 uses AASHTO's "
               "3.4 m/s2 stopping-sight design deceleration instead, and highD "
               "measures 1.4-2.0 m/s2 for the per-track p99"),
    "yield_peak_alat_med": dict(
        lo=0.0, hi=2.0, mid=1.8,
        note="motorway lateral accel\nmostly < 1.8-2.0 m/s2"),
    "react_dist_mean": dict(
        lo=50.0, hi=150.0, mid=None,
        note="EV yield onset 50-150 m\n(refs [18], [20])"),
    "decel_design": dict(
        lo=None, hi=None, mid=3.4,
        note="AASHTO stopping-sight-distance\ndesign deceleration 3.4 m/s2",
        cite="aashto"),
}

#: Dashcam ground truth (out/survey_compare.json, reviewed and rules-only
#: relabelled). Ego-frame, 1 Hz, bounded by the annotator's 50 m range.
GT: dict = dict(onset_med=10.98, onset_med_rules=7.845, lat_speed_med=0.06,
                lat_speed_med_rules=0.47, lat_speed_p90=0.503, range_m=50.0)

#: Which highD target block a figure should quote by default.
DEFAULT_BLOCK = "all:freeflow"

_cache: dict = {}


def has_dataset(path: str | None = None) -> bool:
    return os.path.exists(path or TARGETS_PATH)


def _load(path: str | None = None) -> dict:
    p = path or TARGETS_PATH
    if p not in _cache:
        try:
            with open(p) as fh:
                _cache[p] = json.load(fh)
        except (OSError, ValueError):
            _cache[p] = {}
    return _cache[p]


def measured(block: str = DEFAULT_BLOCK, path: str | None = None) -> dict:
    """Measured highD targets for one pooled block, `{key: {...}}`.

    Returns `{}` when the dataset has never been extracted on this machine, so
    callers must treat a measured value as optional.
    """
    js = _load(path)
    return js.get("targets", {}).get(block, {}) if js else {}


def measured_value(key: str, block: str = DEFAULT_BLOCK,
                   default=None, path: str | None = None):
    m = measured(block, path).get(key)
    return m["value"] if m else default


def measured_band(key: str, block: str = DEFAULT_BLOCK,
                  path: str | None = None):
    """`(lo, hi, mid)` from the measured 10th-90th percentile spread over
    carriageways, or `(None, None, None)` when unavailable."""
    m = measured(block, path).get(key)
    if not m:
        return (None, None, None)
    return (m.get("spread_lo"), m.get("spread_hi"), m.get("value"))


def dataset(path: str | None = None) -> dict:
    """Dataset-level facts (vehicle count, veh-km, lane-change rate, geometry)."""
    return _load(path).get("dataset", {})


def provenance(path: str | None = None) -> dict:
    return _load(path).get("provenance", {})


def literature(key: str | None = None) -> dict:
    return dict(LITERATURE) if key is None else dict(LITERATURE.get(key, {}))


def band(key: str) -> tuple:
    """`(lo, hi, mid)` of the *cited* band - the accessor the existing figure
    scripts want."""
    d = LITERATURE.get(key, {})
    return (d.get("lo"), d.get("hi"), d.get("mid"))


def note(key: str) -> str:
    return LITERATURE.get(key, {}).get("note", "")


def figure_bands() -> dict:
    """Drop-in replacement for `make_param_stats_figs.LIT`.

    Cited values only, byte-identical to the dict it replaces: migrating a
    consumer to this function must not change any existing figure. Measured
    highD values are additive and come from `measured()` / `measured_band()`.
    """
    return {k: dict(lo=v.get("lo"), hi=v.get("hi"), mid=v.get("mid"),
                    note=v.get("note", ""))
            for k, v in LITERATURE.items()
            if k != "decel_design"}


def gt() -> dict:
    """Drop-in replacement for `make_param_stats_figs.GT`."""
    return dict(GT)


def lc_rate_reconciliation(path: str | None = None) -> dict:
    """The three lane-change-rate numbers side by side, with their definitions.

    Used by the figures, tables and papers so the 0.124-vs-0.314 discrepancy is
    stated once, in one place, and never presented as one number being wrong.
    """
    ds = dataset(path)
    m = measured(path=path)
    return dict(
        published=dict(
            value=round(5600.0 / 45000.0, 4), cite="krajewski18",
            label="highD paper, complete manoeuvres",
            note="5600 complete lane changes / 45 000 veh-km, as printed"),
        all_events=dict(
            value=ds.get("lc_per_veh_km"),
            n=ds.get("n_lane_changes"), veh_km=ds.get("veh_km"),
            label="all recorded events",
            note="sum of tracksMeta.numLaneChanges over the released "
                 "recordings, divided by total driven distance"),
        complete_like_for_like=dict(
            value=(m.get("lc_per_veh_km_complete", {}) or {}).get("value"),
            label="this repo, manoeuvre fully inside the tracked section",
            note="two-lane-occupancy timed by the same function the model "
                 "uses (metrics.lane_change_from_track), restricted to "
                 "manoeuvres that neither enter nor leave the field of view"),
    )
