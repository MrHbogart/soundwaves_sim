"""Sound wave room simulation package."""

from .analysis import (
    ImpulseResponseResult,
    AcousticMetrics,
    BandMetrics,
    ReflectionArrival,
    WallDamper,
    WallInteraction,
    DamperEffect,
    compute_band_metrics,
    compute_metrics,
    energy_decay_curve,
    estimate_rt60,
    first_order_wall_interactions,
    impulse_response_with_dampers,
    receiver_positions_for_wall,
    top_wall_interactions,
    octave_band_centers,
    required_panel_area,
    required_absorption_area,
    room_surface_area,
    sabine_rt60,
    damper_effects,
)
from .config import SimulationConfig
from .geometry import RoomBox
from .source import SoundSource
from .simulation import WaveSimulation

__all__ = [
    "ImpulseResponseResult",
    "AcousticMetrics",
    "BandMetrics",
    "ReflectionArrival",
    "WallDamper",
    "WallInteraction",
    "DamperEffect",
    "compute_band_metrics",
    "compute_metrics",
    "SimulationConfig",
    "RoomBox",
    "SoundSource",
    "WaveSimulation",
    "energy_decay_curve",
    "estimate_rt60",
    "first_order_wall_interactions",
    "impulse_response_with_dampers",
    "receiver_positions_for_wall",
    "top_wall_interactions",
    "octave_band_centers",
    "required_panel_area",
    "required_absorption_area",
    "room_surface_area",
    "sabine_rt60",
    "damper_effects",
]
