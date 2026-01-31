"""Simulation core for room sound propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .config import SimulationConfig
from .geometry import RoomBox
from .source import SoundSource
from .analysis import ImpulseResponseResult, ReflectionArrival


@dataclass
class WaveSimulation:
    room: RoomBox
    source: SoundSource
    config: SimulationConfig

    def grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lx, ly, lz = self.config.room_size
        x = np.linspace(0.0, lx, self.config.nx)
        y = np.linspace(0.0, ly, self.config.ny)
        z = np.linspace(0.0, lz, self.config.nz)
        return np.meshgrid(x, y, z, indexing="ij")

    def field_at(self, t: float) -> np.ndarray:
        """Return pressure field at time t on the configured grid."""
        xx, yy, zz = self.grid()
        field = np.zeros_like(xx)
        images = self.room.image_sources(self.source.position, self.config.reflection_order)
        absorption = self.config.wall_absorption

        for image in images:
            dx = xx - image.position[0]
            dy = yy - image.position[1]
            dz = zz - image.position[2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            reflection_gain = (1.0 - absorption) ** image.reflection_count
            field += reflection_gain * self.source.pressure(
                r, t, self.config.c, self.config.attenuation
            )

        return field

    def impulse_response(
        self,
        receiver_position: np.ndarray,
        normalize: bool = True,
        progress: bool = False,
    ) -> ImpulseResponseResult:
        """Compute a broadband impulse response at a receiver location."""
        steps = self.config.steps
        times = np.arange(steps) * self.config.dt
        response = np.zeros(steps, dtype=float)
        absorption = self.config.wall_absorption
        images = self.room.image_sources(self.source.position, self.config.reflection_order)

        for image in _progress_iter(images, len(images), progress, "Impulse response"):
            r = float(np.linalg.norm(receiver_position - image.position))
            if r <= 0.0:
                continue
            delay = r / self.config.c
            idx = int(round(delay / self.config.dt))
            if idx >= steps:
                continue
            reflection_gain = (1.0 - absorption) ** image.reflection_count
            amplitude = reflection_gain * self.source.amplitude
            amplitude *= np.exp(-self.config.attenuation * r) / max(r, 1e-3)
            response[idx] += amplitude

        if normalize and np.max(np.abs(response)) > 0:
            response = response / np.max(np.abs(response))
        return ImpulseResponseResult(time=times, response=response)

    def reflection_arrivals(self, receiver_position: np.ndarray) -> list[ReflectionArrival]:
        """Return arrival times and amplitudes for each image source."""
        arrivals: list[ReflectionArrival] = []
        absorption = self.config.wall_absorption
        images = self.room.image_sources(self.source.position, self.config.reflection_order)

        for image in images:
            r = float(np.linalg.norm(receiver_position - image.position))
            if r <= 0.0:
                continue
            delay = r / self.config.c
            reflection_gain = (1.0 - absorption) ** image.reflection_count
            amplitude = reflection_gain * self.source.amplitude
            amplitude *= np.exp(-self.config.attenuation * r) / max(r, 1e-3)
            arrivals.append(
                ReflectionArrival(time=delay, amplitude=amplitude, reflection_count=image.reflection_count)
            )
        arrivals.sort(key=lambda arrival: arrival.time)
        return arrivals


def _progress_iter(iterable: Iterable, total: int, progress: bool, desc: str):
    if not progress:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc)
