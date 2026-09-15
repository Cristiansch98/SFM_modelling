"""Unreal Engine 5.8 game bridge.

The physics-inspired force model (emv/) is the authoritative "brain" for the
surrounding traffic; a first-person UE5.8 game renders a realistic highway and
lets the user drive the emergency vehicle. UE streams the player EV's state to
Python each tick; Python steps the NPC social-force model (reacting to the live
EV exactly as in out/anim_surrogate_sumo.gif) and streams NPC transforms back.

Mirror of emv/sumo/bridge.py, with the authority reversed: UE owns the EV,
Python owns every NPC. See docs/UE_BRIDGE.md for the wire + coordinate contract.
"""
from .server import UEBridge

__all__ = ["UEBridge"]
