"""Configuration objects for the simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SimulationConfig:
    nx: int = 28
    ny: int = 28
    nz: int = 16
    room_size: Tuple[float, float, float] = (5.0, 5.0, 2.7)  # meters
    c: float = 343.0  # m/s
    dt: float = 1 / 1200.0  # seconds
    duration: float = 0.02  # seconds
    attenuation: float = 0.0  # per meter
    wall_absorption: float = 0.0  # 0 (rigid) to 1 (fully absorbing)
    reflection_order: int = 1  # number of image-source reflections

    @property
    def grid_spacing(self) -> Tuple[float, float, float]:
        lx, ly, lz = self.room_size
        return (lx / (self.nx - 1), ly / (self.ny - 1), lz / (self.nz - 1))

    @property
    def steps(self) -> int:
        return int(self.duration / self.dt)
