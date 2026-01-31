"""Visualization utilities for the simulation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analysis import (
    ImpulseResponseResult,
    DamperEffect,
    WallDamper,
    compute_band_metrics,
    compute_metrics,
    energy_decay_curve,
    estimate_rt60,
    first_order_wall_interactions,
    impulse_response_with_dampers,
    required_panel_area,
    required_absorption_area,
    room_surface_area,
    sabine_rt60,
    wall_plane,
)
from .simulation import WaveSimulation


def room_wireframe(size: Tuple[float, float, float]) -> go.Scatter3d:
    lx, ly, lz = size
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [lx, 0.0, 0.0],
            [lx, ly, 0.0],
            [0.0, ly, 0.0],
            [0.0, 0.0, lz],
            [lx, 0.0, lz],
            [lx, ly, lz],
            [0.0, ly, lz],
        ]
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [corners[i, 0], corners[j, 0], None]
        ys += [corners[i, 1], corners[j, 1], None]
        zs += [corners[i, 2], corners[j, 2], None]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.35)", width=3),
        name="Room",
        showlegend=False,
    )


def source_marker(position: np.ndarray) -> go.Scatter3d:
    return go.Scatter3d(
        x=[position[0]],
        y=[position[1]],
        z=[position[2]],
        mode="markers",
        marker=dict(size=6, color="gold"),
        name="Source",
        showlegend=False,
    )


def receiver_marker(position: np.ndarray) -> go.Scatter3d:
    return go.Scatter3d(
        x=[position[0]],
        y=[position[1]],
        z=[position[2]],
        mode="markers",
        marker=dict(size=5, color="deepskyblue"),
        name="Receiver",
        showlegend=False,
    )


def _axis_style(title: str, axis_color: str, grid_color: str, show_grid: bool) -> Dict[str, Any]:
    return dict(
        title=title,
        showgrid=show_grid,
        gridcolor=grid_color,
        color=axis_color,
        zeroline=False,
        showbackground=False,
    )


def _scene_layout(
    room_size: Tuple[float, float, float],
    background: str,
    axis_color: str,
    grid_color: str,
    show_grid: bool,
    camera: Dict[str, Any] | None,
) -> Dict[str, Any]:
    return dict(
        xaxis=_axis_style("X (m)", axis_color, grid_color, show_grid),
        yaxis=_axis_style("Y (m)", axis_color, grid_color, show_grid),
        zaxis=_axis_style("Z (m)", axis_color, grid_color, show_grid),
        aspectmode="data",
        bgcolor=background,
        camera=camera or {},
    )


def _camera_from_preset(preset: str) -> Dict[str, Any]:
    preset = preset.lower()
    if preset == "top":
        return dict(eye=dict(x=0.0, y=0.0, z=2.2))
    if preset == "front":
        return dict(eye=dict(x=0.0, y=2.0, z=0.6))
    if preset == "side":
        return dict(eye=dict(x=2.0, y=0.0, z=0.6))
    if preset == "iso":
        return dict(eye=dict(x=1.6, y=1.6, z=0.9))
    return dict(eye=dict(x=1.4, y=1.4, z=0.8))


def _apply_camera_motion(
    frames: List[go.Frame],
    motion: str,
    radius: float,
    elevation: float,
    start_angle: float = 0.0,
) -> None:
    if not frames:
        return
    motion = motion.lower()
    total = len(frames)
    if motion == "orbit":
        angles = np.linspace(0.0, 2.0 * np.pi, max(1, total))
    elif motion == "sweep":
        angles = np.linspace(-0.261799, 0.261799, max(1, total))
    else:
        return
    for idx, frame in enumerate(frames):
        angle = start_angle + angles[idx]
        eye = dict(x=radius * np.cos(angle), y=radius * np.sin(angle), z=elevation)
        frame.layout = go.Layout(scene_camera=dict(eye=eye))


def _animation_controls(
    frames: Iterable[go.Frame],
    frame_duration: int,
    transition_duration: int,
    easing: str,
    slider_prefix: str,
    show_slider: bool,
) -> Dict[str, Any]:
    play_button = dict(
        label="Play",
        method="animate",
        args=[
            None,
            {
                "frame": {"duration": frame_duration, "redraw": True},
                "transition": {"duration": transition_duration, "easing": easing},
                "fromcurrent": True,
                "mode": "immediate",
            },
        ],
    )
    pause_button = dict(
        label="Pause",
        method="animate",
        args=[
            [None],
            {
                "frame": {"duration": 0, "redraw": True},
                "mode": "immediate",
            },
        ],
    )
    sliders = []
    if show_slider:
        sliders.append(
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [frame.name],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                        label=frame.name,
                    )
                    for frame in frames
                ],
                currentvalue=dict(prefix=slider_prefix),
            )
        )
    return {
        "updatemenus": [dict(type="buttons", showactive=False, buttons=[play_button, pause_button])],
        "sliders": sliders,
    }

def build_frames(
    simulation: WaveSimulation,
    times: Iterable[float],
    iso_level: float,
    render_stride: int = 2,
    colorscale: str = "RdBu",
    opacity: float = 0.6,
    surface_count: int = 2,
    show_colorbar: bool = False,
    lighting: Dict[str, float] | None = None,
    lightposition: Dict[str, float] | None = None,
    progress: bool = False,
) -> List[go.Frame]:
    frames: List[go.Frame] = []
    times_list = list(times)
    xx, yy, zz = simulation.grid()
    stride = max(1, int(render_stride))
    xx = xx[::stride, ::stride, ::stride]
    yy = yy[::stride, ::stride, ::stride]
    zz = zz[::stride, ::stride, ::stride]
    flat_x = xx.flatten()
    flat_y = yy.flatten()
    flat_z = zz.flatten()

    for t in _progress_iter(times_list, len(times_list), progress, "Isosurface frames"):
        field = simulation.field_at(t)[::stride, ::stride, ::stride]
        frames.append(
            go.Frame(
                data=[
                    go.Isosurface(
                        x=flat_x,
                        y=flat_y,
                        z=flat_z,
                        value=field.flatten(),
                        isomin=-iso_level,
                        isomax=iso_level,
                        surface_count=surface_count,
                        opacity=opacity,
                        colorscale=colorscale,
                        lighting=lighting
                        or dict(ambient=0.4, diffuse=0.6, roughness=0.8, specular=0.2),
                        lightposition=lightposition or dict(x=1.0, y=1.0, z=2.0),
                        caps=dict(x_show=False, y_show=False, z_show=False),
                        showscale=show_colorbar,
                    )
                ],
                name=f"{t:.4f}",
            )
        )
    return frames


def build_slice_frames(
    simulation: WaveSimulation,
    times: Iterable[float],
    iso_level: float,
    slice_axis: str = "z",
    slice_index: int | None = None,
    render_stride: int = 2,
    colorscale: str = "RdBu",
    opacity: float = 0.6,
    show_colorbar: bool = False,
    progress: bool = False,
) -> List[go.Frame]:
    axis = slice_axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError("slice_axis must be 'x', 'y', or 'z'")
    xx, yy, zz = simulation.grid()
    stride = max(1, int(render_stride))
    xx = xx[::stride, ::stride, ::stride]
    yy = yy[::stride, ::stride, ::stride]
    zz = zz[::stride, ::stride, ::stride]

    if axis == "x":
        max_index = xx.shape[0] - 1
    elif axis == "y":
        max_index = yy.shape[1] - 1
    else:
        max_index = zz.shape[2] - 1
    if slice_index is None:
        slice_index = max_index // 2
    slice_index = max(0, min(slice_index, max_index))

    frames: List[go.Frame] = []
    times_list = list(times)
    for t in _progress_iter(times_list, len(times_list), progress, "Slice frames"):
        field = simulation.field_at(t)[::stride, ::stride, ::stride]
        if axis == "x":
            x_plane = xx[slice_index, :, :]
            y_plane = yy[slice_index, :, :]
            z_plane = zz[slice_index, :, :]
            values = field[slice_index, :, :]
        elif axis == "y":
            x_plane = xx[:, slice_index, :]
            y_plane = yy[:, slice_index, :]
            z_plane = zz[:, slice_index, :]
            values = field[:, slice_index, :]
        else:
            x_plane = xx[:, :, slice_index]
            y_plane = yy[:, :, slice_index]
            z_plane = zz[:, :, slice_index]
            values = field[:, :, slice_index]
        frames.append(
            go.Frame(
                data=[
                    go.Surface(
                        x=x_plane,
                        y=y_plane,
                        z=z_plane,
                        surfacecolor=values,
                        colorscale=colorscale,
                        cmin=-iso_level,
                        cmax=iso_level,
                        opacity=opacity,
                        showscale=show_colorbar,
                    )
                ],
                name=f"{t:.4f}",
            )
        )
    return frames


def make_animation(
    simulation: WaveSimulation,
    times: Iterable[float],
    iso_level: float,
    render_stride: int = 2,
    colorscale: str = "RdBu",
    opacity: float = 0.6,
    surface_count: int = 2,
    show_colorbar: bool = False,
    background: str = "rgba(10,10,15,1.0)",
    axis_color: str = "rgba(220,220,220,0.7)",
    grid_color: str = "rgba(255,255,255,0.08)",
    show_grid: bool = True,
    frame_duration: int = 40,
    transition_duration: int = 0,
    easing: str = "linear",
    camera_preset: str = "iso",
    camera_motion: str = "static",
    camera_radius: float = 1.6,
    camera_elevation: float = 0.8,
    show_slider: bool = True,
    progress: bool = False,
) -> go.Figure:
    frames = build_frames(
        simulation,
        times,
        iso_level,
        render_stride=render_stride,
        colorscale=colorscale,
        opacity=opacity,
        surface_count=surface_count,
        show_colorbar=show_colorbar,
        progress=progress,
    )
    _apply_camera_motion(frames, camera_motion, camera_radius, camera_elevation)
    controls = _animation_controls(
        frames, frame_duration, transition_duration, easing, "t = ", show_slider
    )
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            scene=_scene_layout(
                simulation.config.room_size,
                background,
                axis_color,
                grid_color,
                show_grid,
                _camera_from_preset(camera_preset),
            ),
            updatemenus=controls["updatemenus"],
            sliders=controls["sliders"],
        ),
    )
    return fig


def make_pro_animation(
    simulation: WaveSimulation,
    times: Iterable[float],
    iso_level: float,
    render_stride: int = 2,
    slice_axis: str = "z",
    slice_index: int | None = None,
    colorscale: str = "RdBu",
    opacity: float = 0.6,
    surface_count: int = 2,
    show_colorbar: bool = False,
    background: str = "rgba(10,10,15,1.0)",
    axis_color: str = "rgba(220,220,220,0.7)",
    grid_color: str = "rgba(255,255,255,0.08)",
    show_grid: bool = True,
    frame_duration: int = 40,
    transition_duration: int = 0,
    easing: str = "linear",
    camera_preset: str = "iso",
    camera_motion: str = "static",
    camera_radius: float = 1.7,
    camera_elevation: float = 0.9,
    show_room: bool = True,
    show_source: bool = True,
    show_receiver: bool = False,
    receiver_position: np.ndarray | None = None,
    show_slider: bool = True,
    progress: bool = False,
) -> go.Figure:
    frames_iso = build_frames(
        simulation,
        times,
        iso_level,
        render_stride=render_stride,
        colorscale=colorscale,
        opacity=opacity,
        surface_count=surface_count,
        show_colorbar=show_colorbar,
        progress=progress,
    )
    frames_slice = build_slice_frames(
        simulation,
        times,
        iso_level,
        slice_axis=slice_axis,
        slice_index=slice_index,
        render_stride=render_stride,
        colorscale=colorscale,
        opacity=opacity,
        show_colorbar=False,
        progress=progress,
    )
    combined_frames: List[go.Frame] = []
    for iso_frame, slice_frame in _progress_iter(
        zip(frames_iso, frames_slice),
        min(len(frames_iso), len(frames_slice)),
        progress,
        "Combine frames",
    ):
        combined_frames.append(
            go.Frame(
                data=[iso_frame.data[0], slice_frame.data[0]],
                traces=[0, 1],
                name=iso_frame.name,
            )
        )

    _apply_camera_motion(combined_frames, camera_motion, camera_radius, camera_elevation)
    controls = _animation_controls(
        combined_frames, frame_duration, transition_duration, easing, "t = ", show_slider
    )
    base_iso = frames_iso[0].data[0]
    base_slice = frames_slice[0].data[0]
    extra_traces = []
    if show_room:
        extra_traces.append(room_wireframe(simulation.config.room_size))
    if show_source:
        extra_traces.append(source_marker(simulation.source.position))
    if show_receiver and receiver_position is not None:
        extra_traces.append(receiver_marker(receiver_position))
    fig = go.Figure(
        data=[
            base_iso,
            base_slice,
            *extra_traces,
        ],
        frames=combined_frames,
        layout=go.Layout(
            scene=_scene_layout(
                simulation.config.room_size,
                background,
                axis_color,
                grid_color,
                show_grid,
                _camera_from_preset(camera_preset),
            ),
            updatemenus=controls["updatemenus"],
            sliders=controls["sliders"],
        ),
    )
    return fig


def make_slice_animation(
    simulation: WaveSimulation,
    times: Iterable[float],
    iso_level: float,
    slice_axis: str = "z",
    slice_index: int | None = None,
    render_stride: int = 2,
    colorscale: str = "RdBu",
    opacity: float = 0.6,
    show_colorbar: bool = False,
    background: str = "rgba(10,10,15,1.0)",
    axis_color: str = "rgba(220,220,220,0.7)",
    grid_color: str = "rgba(255,255,255,0.08)",
    show_grid: bool = True,
    frame_duration: int = 40,
    transition_duration: int = 0,
    easing: str = "linear",
    camera_preset: str = "iso",
    camera_motion: str = "static",
    camera_radius: float = 1.6,
    camera_elevation: float = 0.8,
    show_slider: bool = True,
    progress: bool = False,
) -> go.Figure:
    frames = build_slice_frames(
        simulation,
        times,
        iso_level,
        slice_axis=slice_axis,
        slice_index=slice_index,
        render_stride=render_stride,
        colorscale=colorscale,
        opacity=opacity,
        show_colorbar=show_colorbar,
        progress=progress,
    )
    _apply_camera_motion(frames, camera_motion, camera_radius, camera_elevation)
    controls = _animation_controls(
        frames, frame_duration, transition_duration, easing, "t = ", show_slider
    )
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            scene=_scene_layout(
                simulation.config.room_size,
                background,
                axis_color,
                grid_color,
                show_grid,
                _camera_from_preset(camera_preset),
            ),
            updatemenus=controls["updatemenus"],
            sliders=controls["sliders"],
        ),
    )
    return fig

def build_sparse_frames(
    simulation: WaveSimulation,
    times: Iterable[float],
    iso_level: float,
    render_stride: int = 2,
    max_points: int = 3000,
    colorscale: str = "RdBu",
    opacity: float = 0.6,
    point_size: int = 3,
    show_colorbar: bool = False,
) -> List[go.Frame]:
    frames: List[go.Frame] = []
    xx, yy, zz = simulation.grid()
    stride = max(1, int(render_stride))
    xx = xx[::stride, ::stride, ::stride]
    yy = yy[::stride, ::stride, ::stride]
    zz = zz[::stride, ::stride, ::stride]

    flat_x = xx.flatten()
    flat_y = yy.flatten()
    flat_z = zz.flatten()

    for t in times:
        field = simulation.field_at(t)[::stride, ::stride, ::stride]
        values = field.flatten()
        mask = np.abs(values) >= iso_level
        idx = np.where(mask)[0]
        if idx.size > max_points:
            idx = np.random.choice(idx, size=max_points, replace=False)
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=flat_x[idx],
                        y=flat_y[idx],
                        z=flat_z[idx],
                        mode="markers",
                        marker=dict(
                            size=point_size,
                            color=values[idx],
                            colorscale=colorscale,
                            cmin=-iso_level,
                            cmax=iso_level,
                            opacity=opacity,
                            showscale=show_colorbar,
                        ),
                    )
                ],
                name=f"{t:.4f}",
            )
        )
    return frames


def build_volume_frames(
    simulation: WaveSimulation,
    times: Iterable[float],
    iso_level: float,
    render_stride: int = 2,
    colorscale: str = "RdBu",
    opacity: float = 0.12,
    surface_count: int = 12,
    show_colorbar: bool = False,
) -> List[go.Frame]:
    frames: List[go.Frame] = []
    xx, yy, zz = simulation.grid()
    stride = max(1, int(render_stride))
    xx = xx[::stride, ::stride, ::stride]
    yy = yy[::stride, ::stride, ::stride]
    zz = zz[::stride, ::stride, ::stride]

    flat_x = xx.flatten()
    flat_y = yy.flatten()
    flat_z = zz.flatten()

    for t in times:
        field = simulation.field_at(t)[::stride, ::stride, ::stride]
        frames.append(
            go.Frame(
                data=[
                    go.Volume(
                        x=flat_x,
                        y=flat_y,
                        z=flat_z,
                        value=field.flatten(),
                        isomin=-iso_level,
                        isomax=iso_level,
                        opacity=opacity,
                        surface_count=surface_count,
                        colorscale=colorscale,
                        showscale=show_colorbar,
                    )
                ],
                name=f"{t:.4f}",
            )
        )
    return frames


def make_sparse_animation(
    simulation: WaveSimulation,
    times: Iterable[float],
    iso_level: float,
    render_stride: int = 2,
    max_points: int = 3000,
    colorscale: str = "RdBu",
    opacity: float = 0.6,
    point_size: int = 3,
    show_colorbar: bool = False,
    background: str = "rgba(10,10,15,1.0)",
    axis_color: str = "rgba(220,220,220,0.7)",
    grid_color: str = "rgba(255,255,255,0.08)",
    show_grid: bool = True,
    frame_duration: int = 40,
    transition_duration: int = 0,
    easing: str = "linear",
    camera_preset: str = "iso",
    camera_motion: str = "static",
    camera_radius: float = 1.6,
    camera_elevation: float = 0.8,
    show_slider: bool = True,
) -> go.Figure:
    frames = build_sparse_frames(
        simulation,
        times,
        iso_level,
        render_stride=render_stride,
        max_points=max_points,
        colorscale=colorscale,
        opacity=opacity,
        point_size=point_size,
        show_colorbar=show_colorbar,
    )
    _apply_camera_motion(frames, camera_motion, camera_radius, camera_elevation)
    controls = _animation_controls(
        frames, frame_duration, transition_duration, easing, "t = ", show_slider
    )
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            scene=_scene_layout(
                simulation.config.room_size,
                background,
                axis_color,
                grid_color,
                show_grid,
                _camera_from_preset(camera_preset),
            ),
            updatemenus=controls["updatemenus"],
            sliders=controls["sliders"],
        ),
    )
    return fig


def make_volume_animation(
    simulation: WaveSimulation,
    times: Iterable[float],
    iso_level: float,
    render_stride: int = 2,
    colorscale: str = "RdBu",
    opacity: float = 0.12,
    surface_count: int = 12,
    show_colorbar: bool = False,
    background: str = "rgba(10,10,15,1.0)",
    axis_color: str = "rgba(220,220,220,0.7)",
    grid_color: str = "rgba(255,255,255,0.08)",
    show_grid: bool = True,
    frame_duration: int = 40,
    transition_duration: int = 0,
    easing: str = "linear",
    camera_preset: str = "iso",
    camera_motion: str = "static",
    camera_radius: float = 1.7,
    camera_elevation: float = 0.9,
    show_slider: bool = True,
) -> go.Figure:
    frames = build_volume_frames(
        simulation,
        times,
        iso_level,
        render_stride=render_stride,
        colorscale=colorscale,
        opacity=opacity,
        surface_count=surface_count,
        show_colorbar=show_colorbar,
    )
    _apply_camera_motion(frames, camera_motion, camera_radius, camera_elevation)
    controls = _animation_controls(
        frames, frame_duration, transition_duration, easing, "t = ", show_slider
    )
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            scene=_scene_layout(
                simulation.config.room_size,
                background,
                axis_color,
                grid_color,
                show_grid,
                _camera_from_preset(camera_preset),
            ),
            updatemenus=controls["updatemenus"],
            sliders=controls["sliders"],
        ),
    )
    return fig


def make_impulse_response_figure(
    result: ImpulseResponseResult,
    show_decay: bool = True,
    show_rt60: bool = True,
    background: str = "rgba(10,10,15,1.0)",
    axis_color: str = "rgba(220,220,220,0.7)",
    grid_color: str = "rgba(255,255,255,0.08)",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.time,
            y=result.response,
            mode="lines",
            line=dict(color="gold"),
            name="Impulse response",
        )
    )
    if show_decay:
        decay_db = energy_decay_curve(result.response)
        fig.add_trace(
            go.Scatter(
                x=result.time,
                y=decay_db,
                mode="lines",
                line=dict(color="deepskyblue"),
                name="Energy decay (dB)",
                yaxis="y2",
            )
        )
        if show_rt60:
            rt60 = estimate_rt60(decay_db, result.time[1] - result.time[0])
            if rt60 is not None:
                fig.add_annotation(
                    x=0.02 * result.time[-1],
                    y=0.95,
                    xref="x",
                    yref="paper",
                    text=f"RT60 approx {rt60:.2f}s",
                    showarrow=False,
                    font=dict(color="deepskyblue"),
                )

    fig.update_layout(
        plot_bgcolor=background,
        paper_bgcolor=background,
        xaxis=dict(
            title="Time (s)",
            showgrid=True,
            gridcolor=grid_color,
            color=axis_color,
            zeroline=False,
        ),
        yaxis=dict(
            title="Amplitude",
            showgrid=True,
            gridcolor=grid_color,
            color=axis_color,
            zeroline=False,
        ),
        yaxis2=dict(
            title="Decay (dB)",
            overlaying="y",
            side="right",
            showgrid=False,
            color=axis_color,
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=axis_color),
        ),
    )
    return fig


def make_acoustic_report_figure(
    simulation: WaveSimulation,
    receiver_position: np.ndarray,
    target_rt60: float | None = None,
    panel_absorption: float = 0.7,
    bands_hz: Iterable[float] | None = None,
    background: str = "rgba(10,10,15,1.0)",
    axis_color: str = "rgba(220,220,220,0.7)",
    grid_color: str = "rgba(255,255,255,0.08)",
) -> go.Figure:
    rir = simulation.impulse_response(receiver_position)
    dt = rir.time[1] - rir.time[0]
    metrics = compute_metrics(rir.response, dt)
    band_metrics = compute_band_metrics(rir.response, dt, bands_hz)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Impulse Response",
            "Energy Decay",
            "RT60 by Octave Band",
            "Clarity (C50/C80)",
        ),
    )

    fig.add_trace(
        go.Scatter(x=rir.time, y=rir.response, mode="lines", line=dict(color="gold")),
        row=1,
        col=1,
    )

    decay_db = energy_decay_curve(rir.response)
    fig.add_trace(
        go.Scatter(x=rir.time, y=decay_db, mode="lines", line=dict(color="deepskyblue")),
        row=1,
        col=2,
    )

    centers = list(band_metrics.keys())
    rt60s = [
        band_metrics[center].rt60 if band_metrics[center].rt60 is not None else 0.0
        for center in centers
    ]
    fig.add_trace(
        go.Bar(
            x=[str(int(center)) for center in centers],
            y=rt60s,
            marker_color="mediumorchid",
            name="RT60",
        ),
        row=2,
        col=1,
    )
    if target_rt60 is not None:
        fig.add_trace(
            go.Scatter(
                x=[str(int(center)) for center in centers],
                y=[target_rt60] * len(centers),
                mode="lines",
                line=dict(color="gold", dash="dash"),
                name="Target RT60",
            ),
            row=2,
            col=1,
        )

    c50s = [
        band_metrics[center].c50 if band_metrics[center].c50 is not None else 0.0
        for center in centers
    ]
    c80s = [
        band_metrics[center].c80 if band_metrics[center].c80 is not None else 0.0
        for center in centers
    ]
    fig.add_trace(
        go.Bar(
            x=[str(int(center)) for center in centers],
            y=c50s,
            marker_color="lightsalmon",
            name="C50",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=[str(int(center)) for center in centers],
            y=c80s,
            marker_color="lightseagreen",
            name="C80",
        ),
        row=2,
        col=2,
    )

    surface_area = room_surface_area(simulation.config.room_size)
    current_absorption = surface_area * simulation.config.wall_absorption
    volume = np.prod(simulation.config.room_size)
    sabine_estimate = sabine_rt60(volume, current_absorption)
    target_absorption = None
    panel_area = None
    if target_rt60 is not None:
        target_absorption = required_absorption_area(volume, target_rt60)
        if target_absorption is not None:
            panel_area = required_panel_area(target_absorption, panel_absorption)

    metrics_lines = [
        f"EDT: {metrics.edt:.2f}s" if metrics.edt is not None else "EDT: n/a",
        f"T20: {metrics.t20:.2f}s" if metrics.t20 is not None else "T20: n/a",
        f"T30: {metrics.t30:.2f}s" if metrics.t30 is not None else "T30: n/a",
        f"RT60: {metrics.rt60:.2f}s" if metrics.rt60 is not None else "RT60: n/a",
        f"C50: {metrics.c50:.1f} dB" if metrics.c50 is not None else "C50: n/a",
        f"C80: {metrics.c80:.1f} dB" if metrics.c80 is not None else "C80: n/a",
        f"D50: {metrics.d50:.1f} %" if metrics.d50 is not None else "D50: n/a",
    ]
    if sabine_estimate is not None:
        metrics_lines.append(f"Sabine RT60: {sabine_estimate:.2f}s")
    if target_absorption is not None:
        metrics_lines.append(f"Target absorption area: {target_absorption:.2f} m^2")
    if panel_area is not None:
        metrics_lines.append(f"Panel area needed (@{panel_absorption:.2f}): {panel_area:.2f} m^2")

    fig.add_annotation(
        text="<br>".join(metrics_lines),
        xref="paper",
        yref="paper",
        x=0.52,
        y=0.98,
        showarrow=False,
        align="left",
        font=dict(color=axis_color, size=11),
        bgcolor="rgba(0,0,0,0)",
    )

    fig.update_layout(
        plot_bgcolor=background,
        paper_bgcolor=background,
        font=dict(color=axis_color),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        bargap=0.2,
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid_color, color=axis_color)
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, color=axis_color)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Decay (dB)", row=1, col=2)
    fig.update_yaxes(title_text="RT60 (s)", row=2, col=1)
    fig.update_yaxes(title_text="Clarity (dB)", row=2, col=2)

    return fig


def make_reflection_arrivals_figure(
    simulation: WaveSimulation,
    receiver_position: np.ndarray,
    background: str = "rgba(10,10,15,1.0)",
    axis_color: str = "rgba(220,220,220,0.7)",
    grid_color: str = "rgba(255,255,255,0.08)",
) -> go.Figure:
    arrivals = simulation.reflection_arrivals(receiver_position)
    times = [arrival.time for arrival in arrivals]
    amplitudes = [arrival.amplitude for arrival in arrivals]
    orders = [arrival.reflection_count for arrival in arrivals]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=times,
                y=amplitudes,
                mode="markers",
                marker=dict(
                    size=8,
                    color=orders,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Reflection order"),
                ),
            )
        ],
        layout=go.Layout(
            plot_bgcolor=background,
            paper_bgcolor=background,
            xaxis=dict(
                title="Arrival time (s)",
                showgrid=True,
                gridcolor=grid_color,
                color=axis_color,
                zeroline=False,
            ),
            yaxis=dict(
                title="Amplitude",
                showgrid=True,
                gridcolor=grid_color,
                color=axis_color,
                zeroline=False,
            ),
            font=dict(color=axis_color),
        ),
    )
    return fig


def make_wall_interaction_heatmap(
    simulation: WaveSimulation,
    receiver_position: np.ndarray,
    wall: str,
    grid_size: int = 40,
    spread: float = 0.3,
    dampers: Iterable[WallDamper] | None = None,
    background: str = "rgba(10,10,15,1.0)",
    axis_color: str = "rgba(220,220,220,0.7)",
    grid_color: str = "rgba(255,255,255,0.08)",
) -> go.Figure:
    axis, _ = wall_plane(wall, simulation.config.room_size)
    lx, ly, lz = simulation.config.room_size
    if axis == 0:
        u = np.linspace(0.0, ly, grid_size)
        v = np.linspace(0.0, lz, grid_size)
        u_label, v_label = "Y (m)", "Z (m)"
    elif axis == 1:
        u = np.linspace(0.0, lx, grid_size)
        v = np.linspace(0.0, lz, grid_size)
        u_label, v_label = "X (m)", "Z (m)"
    else:
        u = np.linspace(0.0, lx, grid_size)
        v = np.linspace(0.0, ly, grid_size)
        u_label, v_label = "X (m)", "Y (m)"
    uu, vv = np.meshgrid(u, v, indexing="xy")
    heat = np.zeros_like(uu)
    interactions = [
        interaction
        for interaction in first_order_wall_interactions(simulation, receiver_position)
        if interaction.wall.lower() == wall.lower()
    ]
    for interaction in interactions:
        if axis == 0:
            du = uu - interaction.position[1]
            dv = vv - interaction.position[2]
        elif axis == 1:
            du = uu - interaction.position[0]
            dv = vv - interaction.position[2]
        else:
            du = uu - interaction.position[0]
            dv = vv - interaction.position[1]
        heat += interaction.energy * np.exp(-(du * du + dv * dv) / max(spread, 1e-3) ** 2)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=u,
            y=v,
            z=heat,
            colorscale="Inferno",
            colorbar=dict(title="Interaction energy"),
        )
    )
    for interaction in interactions:
        if axis == 0:
            x_pt, y_pt = interaction.position[1], interaction.position[2]
        elif axis == 1:
            x_pt, y_pt = interaction.position[0], interaction.position[2]
        else:
            x_pt, y_pt = interaction.position[0], interaction.position[1]
        fig.add_trace(
            go.Scatter(
                x=[x_pt],
                y=[y_pt],
                mode="markers",
                marker=dict(size=10, color="deepskyblue"),
                name="Reflection point",
                showlegend=False,
            )
        )
    for damper in dampers or []:
        if damper.wall.lower() != wall.lower():
            continue
        if axis == 0:
            x_pt, y_pt = damper.center[1], damper.center[2]
        elif axis == 1:
            x_pt, y_pt = damper.center[0], damper.center[2]
        else:
            x_pt, y_pt = damper.center[0], damper.center[1]
        fig.add_trace(
            go.Scatter(
                x=[x_pt],
                y=[y_pt],
                mode="markers",
                marker=dict(size=12, color="gold", symbol="x"),
                name="Damper",
                showlegend=False,
            )
        )

    fig.update_layout(
        plot_bgcolor=background,
        paper_bgcolor=background,
        xaxis=dict(title=u_label, showgrid=True, gridcolor=grid_color, color=axis_color),
        yaxis=dict(title=v_label, showgrid=True, gridcolor=grid_color, color=axis_color),
        font=dict(color=axis_color),
        title=f"Wall interaction heatmap ({wall})",
    )
    return fig


def make_damper_overview_figure(
    simulation: WaveSimulation,
    receiver_positions: Dict[str, np.ndarray],
    wall: str,
    dampers: Iterable[WallDamper] | None = None,
    background: str = "rgba(10,10,15,1.0)",
    axis_color: str = "rgba(220,220,220,0.7)",
    grid_color: str = "rgba(255,255,255,0.08)",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(room_wireframe(simulation.config.room_size))
    fig.add_trace(source_marker(simulation.source.position))
    for label, receiver in receiver_positions.items():
        fig.add_trace(
            go.Scatter3d(
                x=[receiver[0]],
                y=[receiver[1]],
                z=[receiver[2]],
                mode="markers+text",
                marker=dict(size=5, color="deepskyblue"),
                text=[label],
                textposition="top center",
                showlegend=False,
            )
        )
    interactions = []
    for receiver in receiver_positions.values():
        interactions.extend(first_order_wall_interactions(simulation, receiver))
    for interaction in interactions:
        if interaction.wall.lower() != wall.lower():
            continue
        fig.add_trace(
            go.Scatter3d(
                x=[interaction.position[0]],
                y=[interaction.position[1]],
                z=[interaction.position[2]],
                mode="markers",
                marker=dict(size=6, color="salmon"),
                showlegend=False,
            )
        )
    for damper in dampers or []:
        fig.add_trace(
            go.Scatter3d(
                x=[damper.center[0]],
                y=[damper.center[1]],
                z=[damper.center[2]],
                mode="markers",
                marker=dict(size=7, color="gold", symbol="diamond"),
                showlegend=False,
            )
        )
    fig.update_layout(
        scene=_scene_layout(
            simulation.config.room_size,
            background,
            axis_color,
            grid_color,
            True,
            _camera_from_preset("iso"),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Damper study overview ({wall})",
    )
    return fig


def make_damper_effects_figure(
    effects: Iterable[DamperEffect],
    background: str = "rgba(10,10,15,1.0)",
    axis_color: str = "rgba(220,220,220,0.7)",
    grid_color: str = "rgba(255,255,255,0.08)",
) -> go.Figure:
    effects = list(effects)
    if not effects:
        return go.Figure()
    damper_labels = []
    for effect in effects:
        center = effect.damper.center
        damper_labels.append(f"{effect.damper.wall}@{center[0]:.2f},{center[1]:.2f},{center[2]:.2f}")
    fig = go.Figure()
    unique_receivers = sorted({effect.receiver_label for effect in effects})
    for receiver_label in unique_receivers:
        receiver_effects = [effect for effect in effects if effect.receiver_label == receiver_label]
        fig.add_trace(
            go.Bar(
                x=[
                    f"{effect.damper.wall}@{effect.damper.center[0]:.2f},{effect.damper.center[1]:.2f}"
                    for effect in receiver_effects
                ],
                y=[effect.energy_reduction_pct for effect in receiver_effects],
                name=receiver_label,
            )
        )
    fig.update_layout(
        plot_bgcolor=background,
        paper_bgcolor=background,
        xaxis=dict(title="Damper location", showgrid=True, gridcolor=grid_color, color=axis_color),
        yaxis=dict(
            title="Energy reduction (%)", showgrid=True, gridcolor=grid_color, color=axis_color
        ),
        font=dict(color=axis_color),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        title="Damper impact by receiver position",
    )
    return fig


def make_impulse_response_comparison_figure(
    simulation: WaveSimulation,
    receiver_position: np.ndarray,
    damper: WallDamper,
    background: str = "rgba(10,10,15,1.0)",
    axis_color: str = "rgba(220,220,220,0.7)",
    grid_color: str = "rgba(255,255,255,0.08)",
) -> go.Figure:
    baseline = simulation.impulse_response(receiver_position, normalize=True)
    damped = impulse_response_with_dampers(
        simulation, receiver_position, dampers=[damper], normalize=True
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=baseline.time,
            y=baseline.response,
            mode="lines",
            line=dict(color="deepskyblue"),
            name="Baseline",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=damped.time,
            y=damped.response,
            mode="lines",
            line=dict(color="gold"),
            name="With damper",
        )
    )
    fig.update_layout(
        plot_bgcolor=background,
        paper_bgcolor=background,
        xaxis=dict(title="Time (s)", showgrid=True, gridcolor=grid_color, color=axis_color),
        yaxis=dict(title="Normalized amplitude", showgrid=True, gridcolor=grid_color, color=axis_color),
        font=dict(color=axis_color),
        title=f"Impulse response comparison ({damper.wall})",
    )
    return fig


def _progress_iter(iterable: Iterable, total: int, progress: bool, desc: str):
    if not progress:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc)
