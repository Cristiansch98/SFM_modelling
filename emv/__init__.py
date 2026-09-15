"""
emv - physics-informed repulsion-force framework for emergency-vehicle yielding.

A social-force / artificial-potential-field model (Helbing-Molnar [2], Wolf &
Burdick [11]) of how surrounding traffic gives way to an emergency vehicle
(EV), with:

  * driving force with relaxation time tau                    [2, 4]
  * anisotropic exponential repulsion from the EV             [2, 16]
  * corridor repulsion from the EV's *predicted path*         (gap 1 of the
    literature review; anticipation in the spirit of [8, 15])
  * siren-perception onset: detection radius + reaction delay (gap 2, [20])
  * washboard lane-keeping potential -> lane changes are a
    depinning transition with an analytic threshold           [10, 11]
  * IDM longitudinal safety layer                             [4]
  * calibration machinery (LHS + Nelder-Mead)                 (gap 3, [9, 18])
  * hybrid force->SUMO sublane projection                     (gap 4, [17, 26, 27])

Reference numbers [n] refer to literature_review_ev_repulsion_models.md.
All quantities are SI (m, s, m/s, m/s^2); forces are per unit mass.
"""

from .params import Params
from .road import Road, Corridor
from .state import VehState, UNAWARE, NOTICED, YIELDING, HOLD
from .simulate import Sim, History
from .scenarios import make_overtake, make_jam
from . import metrics

__version__ = "1.0.0"
