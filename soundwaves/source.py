"""Sound source definitions."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SoundSource:
    position: np.ndarray
    frequencies: np.ndarray
    amplitude: float = 1.0
    phase: float = 0.0

    @staticmethod
    def guitar_amp_spectrum(
        position: np.ndarray,
        freq_min: float = 82.0,
        freq_max: float = 880.0,
        bands: int = 6,
        amplitude: float = 1.0,
        phase: float = 0.0,
    ) -> "SoundSource":
        """Create a broadband source using log-spaced bands."""
        freqs = np.geomspace(freq_min, freq_max, bands)
        return SoundSource(position=position, frequencies=freqs, amplitude=amplitude, phase=phase)

    def pressure(self, r: np.ndarray, t: float, c: float, attenuation: float) -> np.ndarray:
        """Compute a spherical wave pressure field at distance r (array)."""
        time_delay = r / c
        # Avoid divide-by-zero at the source by clamping r.
        r_safe = np.maximum(r, 1e-3)
        envelope = self.amplitude * np.exp(-attenuation * r_safe) / r_safe
        signal = np.zeros_like(r_safe)
        for freq in self.frequencies:
            omega = 2.0 * np.pi * freq
            signal += np.sin(omega * (t - time_delay) + self.phase)
        return envelope * signal
