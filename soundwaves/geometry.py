"""Room geometry and reflection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class ImageSource:
    position: np.ndarray
    reflection_count: int
    indices: Tuple[int, int, int]


@dataclass(frozen=True)
class RoomBox:
    size: Tuple[float, float, float]

    def image_sources(self, source_position: np.ndarray, order: int) -> List[ImageSource]:
        """Generate image sources for axis-aligned room boundaries.

        Uses the method of images for parallel planes at 0 and L on each axis.
        """
        if order <= 0:
            return [ImageSource(position=source_position, reflection_count=0)]

        lx, ly, lz = self.size
        images: List[ImageSource] = []
        for nx in range(-order, order + 1):
            for ny in range(-order, order + 1):
                for nz in range(-order, order + 1):
                    rx = (-1) ** nx * source_position[0] + 2 * nx * lx
                    ry = (-1) ** ny * source_position[1] + 2 * ny * ly
                    rz = (-1) ** nz * source_position[2] + 2 * nz * lz
                    reflections = abs(nx) + abs(ny) + abs(nz)
                    images.append(
                        ImageSource(
                            position=np.array([rx, ry, rz]),
                            reflection_count=reflections,
                            indices=(nx, ny, nz),
                        )
                    )
        return images
