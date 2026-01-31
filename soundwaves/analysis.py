"""Analysis utilities for room acoustics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ImpulseResponseResult:
    time: np.ndarray
    response: np.ndarray


@dataclass(frozen=True)
class AcousticMetrics:
    edt: float | None
    t20: float | None
    t30: float | None
    rt60: float | None
    c50: float | None
    c80: float | None
    d50: float | None


@dataclass(frozen=True)
class BandMetrics:
    center_hz: float
    metrics: AcousticMetrics


@dataclass(frozen=True)
class ReflectionArrival:
    time: float
    amplitude: float
    reflection_count: int


@dataclass(frozen=True)
class WallInteraction:
    wall: str
    position: np.ndarray
    time: float
    amplitude: float
    energy: float
    image_index: Tuple[int, int, int]


@dataclass(frozen=True)
class WallDamper:
    wall: str
    center: np.ndarray
    radius: float
    absorption: float


@dataclass(frozen=True)
class DamperEffect:
    receiver_label: str
    damper: WallDamper
    baseline_energy: float
    damped_energy: float
    energy_reduction_db: float
    energy_reduction_pct: float
    metrics_before: AcousticMetrics
    metrics_after: AcousticMetrics


def energy_decay_curve(response: np.ndarray) -> np.ndarray:
    """Compute Schroeder energy decay curve in dB."""
    energy = np.cumsum(response[::-1] ** 2)[::-1]
    energy = np.maximum(energy, 1e-12)
    energy /= energy[0]
    return 10.0 * np.log10(energy)


def estimate_rt60(
    decay_db: np.ndarray,
    dt: float,
    fit_range_db: Tuple[float, float] = (-5.0, -35.0),
) -> float | None:
    """Estimate RT60 from a linear fit of the decay curve."""
    upper, lower = fit_range_db
    if upper > 0 or lower >= upper:
        raise ValueError("fit_range_db must be negative and ordered (upper, lower).")
    idx = np.where((decay_db <= upper) & (decay_db >= lower))[0]
    if idx.size < 2:
        return None
    times = idx * dt
    coeffs = np.polyfit(times, decay_db[idx], 1)
    slope = coeffs[0]
    if slope >= 0:
        return None
    return -60.0 / slope


def _estimate_decay_slope(
    decay_db: np.ndarray, dt: float, fit_range_db: Tuple[float, float]
) -> float | None:
    upper, lower = fit_range_db
    if upper > 0 or lower >= upper:
        raise ValueError("fit_range_db must be negative and ordered (upper, lower).")
    idx = np.where((decay_db <= upper) & (decay_db >= lower))[0]
    if idx.size < 2:
        return None
    times = idx * dt
    coeffs = np.polyfit(times, decay_db[idx], 1)
    slope = coeffs[0]
    if slope >= 0:
        return None
    return slope


def compute_metrics(response: np.ndarray, dt: float) -> AcousticMetrics:
    decay_db = energy_decay_curve(response)
    edt_slope = _estimate_decay_slope(decay_db, dt, (0.0, -10.0))
    t20_slope = _estimate_decay_slope(decay_db, dt, (-5.0, -25.0))
    t30_slope = _estimate_decay_slope(decay_db, dt, (-5.0, -35.0))

    def _rt60_from_slope(slope: float | None) -> float | None:
        if slope is None:
            return None
        return -60.0 / slope

    edt = _rt60_from_slope(edt_slope)
    t20 = _rt60_from_slope(t20_slope)
    t30 = _rt60_from_slope(t30_slope)
    rt60 = estimate_rt60(decay_db, dt, (-5.0, -35.0))

    sample_rate = 1.0 / dt
    n50 = int(round(0.05 * sample_rate))
    n80 = int(round(0.08 * sample_rate))
    energy = response**2
    total_energy = np.sum(energy) + 1e-12
    early50 = np.sum(energy[:n50])
    early80 = np.sum(energy[:n80])
    late50 = np.sum(energy[n50:])
    late80 = np.sum(energy[n80:])

    c50 = 10.0 * np.log10((early50 + 1e-12) / (late50 + 1e-12))
    c80 = 10.0 * np.log10((early80 + 1e-12) / (late80 + 1e-12))
    d50 = 100.0 * early50 / total_energy

    return AcousticMetrics(edt=edt, t20=t20, t30=t30, rt60=rt60, c50=c50, c80=c80, d50=d50)


def octave_band_centers() -> np.ndarray:
    return np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)


def _bandpass_fft(signal: np.ndarray, sample_rate: float, low_hz: float, high_hz: float) -> np.ndarray:
    n = signal.size
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    filtered = np.zeros_like(spectrum)
    filtered[mask] = spectrum[mask]
    return np.fft.irfft(filtered, n=n)


def compute_band_metrics(
    response: np.ndarray,
    dt: float,
    centers_hz: Iterable[float] | None = None,
    progress: bool = False,
) -> Dict[float, AcousticMetrics]:
    sample_rate = 1.0 / dt
    centers = np.array(list(centers_hz or octave_band_centers()), dtype=float)
    band_metrics: Dict[float, AcousticMetrics] = {}
    for center in _progress_iter(centers, len(centers), progress, "Octave bands"):
        low = center / np.sqrt(2.0)
        high = center * np.sqrt(2.0)
        filtered = _bandpass_fft(response, sample_rate, low, high)
        band_metrics[center] = compute_metrics(filtered, dt)
    return band_metrics


def sabine_rt60(volume_m3: float, absorption_area_m2: float) -> float | None:
    if absorption_area_m2 <= 0.0:
        return None
    return 0.161 * volume_m3 / absorption_area_m2


def required_absorption_area(volume_m3: float, target_rt60: float) -> float | None:
    if target_rt60 <= 0.0:
        return None
    return 0.161 * volume_m3 / target_rt60


def room_surface_area(room_size: Tuple[float, float, float]) -> float:
    lx, ly, lz = room_size
    return 2.0 * (lx * ly + lx * lz + ly * lz)


def required_panel_area(absorption_area_m2: float, panel_absorption: float) -> float | None:
    if panel_absorption <= 0.0:
        return None
    return absorption_area_m2 / panel_absorption


def wall_plane(wall: str, room_size: Tuple[float, float, float]) -> Tuple[int, float]:
    """Return (axis_index, plane_value) for a wall identifier."""
    wall = wall.lower()
    lx, ly, lz = room_size
    mapping = {
        "x0": (0, 0.0),
        "xl": (0, lx),
        "y0": (1, 0.0),
        "yl": (1, ly),
        "z0": (2, 0.0),
        "zl": (2, lz),
    }
    if wall not in mapping:
        raise ValueError("wall must be one of x0, xL, y0, yL, z0, zL")
    return mapping[wall]


def wall_from_indices(indices: Tuple[int, int, int]) -> str | None:
    nx, ny, nz = indices
    if abs(nx) + abs(ny) + abs(nz) != 1:
        return None
    if nx != 0:
        return "xL" if nx > 0 else "x0"
    if ny != 0:
        return "yL" if ny > 0 else "y0"
    if nz != 0:
        return "zL" if nz > 0 else "z0"
    return None


def reflection_point(
    receiver_position: np.ndarray,
    image_position: np.ndarray,
    wall: str,
    room_size: Tuple[float, float, float],
) -> np.ndarray | None:
    axis, plane_value = wall_plane(wall, room_size)
    direction = image_position - receiver_position
    denom = direction[axis]
    if np.isclose(denom, 0.0):
        return None
    t = (plane_value - receiver_position[axis]) / denom
    if t <= 0.0 or t >= 1.0:
        return None
    point = receiver_position + t * direction
    return point


def first_order_wall_interactions(
    simulation: "WaveSimulation", receiver_position: np.ndarray
) -> List[WallInteraction]:
    """Compute specular first-order reflection points and strengths."""
    interactions: List[WallInteraction] = []
    images = simulation.room.image_sources(simulation.source.position, simulation.config.reflection_order)
    absorption = simulation.config.wall_absorption
    for image in images:
        if image.reflection_count != 1:
            continue
        wall = wall_from_indices(image.indices)
        if wall is None:
            continue
        point = reflection_point(receiver_position, image.position, wall, simulation.config.room_size)
        if point is None:
            continue
        r = float(np.linalg.norm(receiver_position - image.position))
        if r <= 0.0:
            continue
        delay = r / simulation.config.c
        reflection_gain = (1.0 - absorption) ** image.reflection_count
        amplitude = reflection_gain * simulation.source.amplitude
        amplitude *= np.exp(-simulation.config.attenuation * r) / max(r, 1e-3)
        interactions.append(
            WallInteraction(
                wall=wall,
                position=point,
                time=delay,
                amplitude=amplitude,
                energy=amplitude * amplitude,
                image_index=image.indices,
            )
        )
    return interactions


def top_wall_interactions(
    interactions: Sequence[WallInteraction], wall: str, top_n: int = 3
) -> List[WallInteraction]:
    wall = wall.lower()
    filtered = [interaction for interaction in interactions if interaction.wall.lower() == wall]
    filtered.sort(key=lambda item: item.energy, reverse=True)
    return filtered[: max(0, int(top_n))]


def receiver_positions_for_wall(
    room_size: Tuple[float, float, float],
    wall: str,
    offset: float = 0.5,
    height: float | None = None,
) -> Dict[str, np.ndarray]:
    """Return receiver positions inside and behind the selected wall."""
    lx, ly, lz = room_size
    z = height if height is not None else 0.5 * lz
    center = np.array([0.5 * lx, 0.5 * ly, z], dtype=float)
    axis, plane_value = wall_plane(wall, room_size)
    behind = center.copy()
    if plane_value <= 0.0:
        behind[axis] = -abs(offset)
    else:
        behind[axis] = plane_value + abs(offset)
    return {
        "center": center,
        f"behind_{wall}": behind,
    }


def impulse_response_with_dampers(
    simulation: "WaveSimulation",
    receiver_position: np.ndarray,
    dampers: Sequence[WallDamper] | None = None,
    normalize: bool = True,
    progress: bool = False,
) -> ImpulseResponseResult:
    """Impulse response with optional wall damper patches applied."""
    steps = simulation.config.steps
    times = np.arange(steps) * simulation.config.dt
    response = np.zeros(steps, dtype=float)
    absorption = simulation.config.wall_absorption
    dampers = list(dampers or [])
    images = simulation.room.image_sources(simulation.source.position, simulation.config.reflection_order)

    for image in _progress_iter(images, len(images), progress, "Impulse response (damped)"):
        r = float(np.linalg.norm(receiver_position - image.position))
        if r <= 0.0:
            continue
        delay = r / simulation.config.c
        idx = int(round(delay / simulation.config.dt))
        if idx >= steps:
            continue
        reflection_gain = (1.0 - absorption) ** image.reflection_count
        amplitude = reflection_gain * simulation.source.amplitude
        amplitude *= np.exp(-simulation.config.attenuation * r) / max(r, 1e-3)
        if image.reflection_count == 1 and dampers:
            wall = wall_from_indices(image.indices)
            if wall is not None:
                point = reflection_point(
                    receiver_position, image.position, wall, simulation.config.room_size
                )
                if point is not None:
                    for damper in dampers:
                        if damper.wall.lower() != wall.lower():
                            continue
                        distance = float(np.linalg.norm(point - damper.center))
                        if distance <= damper.radius:
                            amplitude *= (1.0 - damper.absorption)
        response[idx] += amplitude

    if normalize and np.max(np.abs(response)) > 0:
        response = response / np.max(np.abs(response))
    return ImpulseResponseResult(time=times, response=response)


def damper_effects(
    simulation: "WaveSimulation",
    receiver_positions: Dict[str, np.ndarray],
    dampers: Sequence[WallDamper],
    progress: bool = False,
) -> List[DamperEffect]:
    results: List[DamperEffect] = []
    receiver_items = list(receiver_positions.items())
    damper_list = list(dampers)
    total = len(receiver_items) * (len(damper_list) + 1)
    progress_bar = _progress_bar(total, progress, "Damper sweep")
    try:
        for label, receiver in receiver_items:
            baseline = simulation.impulse_response(receiver, normalize=False)
            baseline_metrics = compute_metrics(baseline.response, simulation.config.dt)
            baseline_energy = float(np.sum(baseline.response**2))
            progress_bar.update(1)
            for damper in damper_list:
                damped = impulse_response_with_dampers(
                    simulation, receiver, dampers=[damper], normalize=False
                )
                damped_metrics = compute_metrics(damped.response, simulation.config.dt)
                damped_energy = float(np.sum(damped.response**2))
                reduction_ratio = (baseline_energy - damped_energy) / max(baseline_energy, 1e-12)
                energy_reduction_db = 10.0 * np.log10(max(baseline_energy, 1e-12) / max(damped_energy, 1e-12))
                results.append(
                    DamperEffect(
                        receiver_label=label,
                        damper=damper,
                        baseline_energy=baseline_energy,
                        damped_energy=damped_energy,
                        energy_reduction_db=energy_reduction_db,
                        energy_reduction_pct=100.0 * reduction_ratio,
                        metrics_before=baseline_metrics,
                        metrics_after=damped_metrics,
                    )
                )
                progress_bar.update(1)
    finally:
        progress_bar.close()
    return results


def _progress_iter(iterable: Iterable, total: int, progress: bool, desc: str):
    if not progress:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


class _NullProgressBar:
    def update(self, _=1) -> None:
        return None

    def close(self) -> None:
        return None


def _progress_bar(total: int, progress: bool, desc: str):
    if not progress:
        return _NullProgressBar()
    try:
        from tqdm.auto import tqdm
    except Exception:
        return _NullProgressBar()
    return tqdm(total=total, desc=desc)
