# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Presentation layout: the camera above, a minimal HD map below.

The DEFAULT layout is a debug view. It strokes every lane's centreline and both
its edges, then adds agent ids, the ground-truth ghost car, prediction
trajectories and a metrics table. That is the right amount of information for
reading a regression and far too much for showing a rollout to someone outside
the team, who sees a thicket of lines rather than a street.

This layout keeps the same drawing and removes what does not earn its place:

  * the camera on top, whole and uncropped, at its own resolution,
  * the map below it at the same width, turned so the car drives upwards,
  * lane boundaries drawn ONCE -- a divider is the right edge of one lane and
    the left edge of the next, and the debug view draws both,
  * no centrelines, no agent ids, no ghost car, no predictions, no table,
  * people as small dots, so a crowded pavement does not read as traffic.

Everything is post-hoc: `eval.main` re-renders from a rollout's ASL log without
re-simulating, so a style change costs one CPU job and no GPU time.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import shapely
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from trajdata import maps
from trajdata.maps import VectorMap

from eval.data import SimulationResult
from eval.schema import EvalConfig

logger = logging.getLogger(__name__)

EGO_ID = "EGO"


@dataclasses.dataclass(frozen=True)
class PresentationStyle:
    """Every colour and weight in the layout, in one place.

    Kept as a dataclass rather than config keys so the look can be iterated on
    from a render script without touching the Hydra schema.
    """

    background: str = "#000000"
    # A wash under the linework, so the carriageway reads as a surface against
    # the background. None draws the boundaries alone.
    road_fill: str | None = "#161616"
    lane_line: str = "#ffffff"
    lane_line_width: float = 0.9
    lane_line_alpha: float = 0.75
    # Vehicles. The ego is the one thing that has to be found at a glance, so it
    # is the only coloured object in the frame; everything else is white.
    ego: str = "#ffbe28"
    actor: str = "#ffffff"
    actor_alpha: float = 0.92
    # Drawn between neighbouring cars so a queue does not merge into one shape.
    vehicle_edge: str = "#000000"
    # People, drawn small and dimmer. A station forecourt has dozens of them and
    # boxing each one the way a car is boxed buries the scene.
    pedestrian: str = "#ffffff"
    pedestrian_alpha: float = 0.55
    pedestrian_radius_m: float = 0.45
    # Anything whose bounding box fits inside this is treated as a person.
    pedestrian_max_extent_m: float = 1.2
    # The path the ego drives, in its own colour.
    trail: str = "#ffbe28"
    trail_width: float = 1.8
    # A hairline between the two panels.
    divider: str = "#2a2a2a"
    # Where the car sits in the map panel, as fractions from left and bottom.
    # Low, because the road ahead is what there is to look at.
    map_ego_x_frac: float = 0.5
    map_ego_y_frac: float = 0.28


def _lane_polygon(lane) -> shapely.Polygon | None:
    """One lane's drivable area, from its two edges."""
    if lane.left_edge is None or lane.right_edge is None:
        return None
    left = np.asarray(lane.left_edge.xy)
    right = np.asarray(lane.right_edge.xy)
    if left.ndim != 2 or right.ndim != 2 or len(left) < 2 or len(right) < 2:
        return None
    polygon = shapely.Polygon(np.vstack([left[:, :2], right[::-1, :2]]))
    if not polygon.is_valid:
        # Self-intersecting ribbons happen on tight curves; buffer(0) repairs them.
        polygon = polygon.buffer(0)
    return polygon if (not polygon.is_empty and polygon.area > 0.0) else None


def lane_boundaries(vec_map: VectorMap) -> list[shapely.LineString]:
    """Every lane boundary, each drawn once.

    Neighbouring lanes share a boundary, and the map stores it twice: once as
    one lane's right edge and once as the next lane's left edge, digitised a few
    centimetres apart. Drawing both is what gives the debug view its doubled,
    cluttered look, so near-duplicates are dropped here.
    """
    boundaries: list[shapely.LineString] = []
    seen: set[tuple] = set()
    lanes = vec_map.elements[maps.vec_map_elements.MapElementType.ROAD_LANE]
    for lane in lanes.values():
        for edge in (lane.left_edge, lane.right_edge):
            if edge is None:
                continue
            xy = np.asarray(edge.xy)
            if xy.ndim != 2 or len(xy) < 2:
                continue
            xy = xy[:, :2]
            start, end = xy[0], xy[-1]
            # Two spellings of the same boundary agree on where it starts, where
            # it ends and how long it is, to well under half a metre.
            key = (
                tuple(np.round(np.minimum(start, end) * 2).astype(int)),
                tuple(np.round(np.maximum(start, end) * 2).astype(int)),
                int(round(shapely.LineString(xy).length)),
            )
            if key in seen:
                continue
            seen.add(key)
            boundaries.append(shapely.LineString(xy))
    return boundaries


def _compound_path(geometry) -> Path | None:
    """Matplotlib path for a (Multi)Polygon, holes included."""
    if geometry is None or geometry.is_empty:
        return None
    polygons = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    paths = []
    for polygon in polygons:
        if not isinstance(polygon, shapely.Polygon) or polygon.is_empty:
            continue
        # Matplotlib fills by winding number, so an outline and the holes in it
        # must wind opposite ways; shapely does not promise either orientation.
        polygon = shapely.geometry.polygon.orient(polygon, sign=1.0)
        for ring in [polygon.exterior, *polygon.interiors]:
            xy = np.asarray(ring.coords)
            if len(xy) < 3:
                continue
            codes = np.full(len(xy), Path.LINETO, dtype=np.uint8)
            codes[0] = Path.MOVETO
            codes[-1] = Path.CLOSEPOLY
            paths.append(Path(xy[:, :2], codes))
    if not paths:
        return None
    return Path.make_compound_path(*paths)


def _fps_from_timestamps(timestamps_us: np.ndarray) -> float:
    if len(timestamps_us) < 2:
        return 10.0
    step_us = float(np.median(np.diff(np.asarray(timestamps_us, dtype=float))))
    return 1e6 / step_us if step_us > 0 else 10.0


def render_presentation_video(
    sim_result: SimulationResult,
    cfg: EvalConfig,
    output_path: str,
    style: PresentationStyle | None = None,
    width_px: int = 1920,
    map_height_px: int = 1080,
    dpi: int = 120,
    camera_id: str | None = None,
    frame_indices: list[int] | None = None,
) -> None:
    """Render one rollout in the presentation layout and write it to `output_path`.

    The frame is `width_px` across. The camera keeps its own shape, so its panel
    is as tall as that width makes it, and the map panel adds `map_height_px`
    below. Nothing is cropped or stretched.

    Args:
        sim_result: The rollout to draw.
        cfg: Eval config; `video.map_video.map_radius_m` sets how far ahead of
            and behind the car the map reaches.
        output_path: mp4 to write.
        style: Colours and weights.
        width_px: Width of the frame; both panels use all of it.
        map_height_px: Height of the map panel.
        dpi: Matplotlib dpi; only the product with figsize matters.
        camera_id: Logical camera to show; defaults to the configured one.
        frame_indices: Which frames to render; all of them by default. A single
            index renders a one-frame clip, which is the fast way to judge a
            style change without waiting for a whole rollout.
    """
    style = style or PresentationStyle()
    timestamps_us = np.asarray(sim_result.timestamps_us)
    if len(timestamps_us) == 0:
        raise ValueError("rollout has no timestamps to render")

    camera_id = camera_id or cfg.video.camera_id_to_render
    camera = sim_result.cameras.camera_by_logical_id[camera_id]
    first_image = camera.image_at_time(int(timestamps_us[0]))
    if first_image is None:
        raise ValueError(f"camera {camera_id} has no image to size the frame with")

    image_w, image_h = first_image.size
    camera_height_px = round(width_px * image_h / image_w)
    total_height_px = camera_height_px + map_height_px

    fig = plt.figure(
        figsize=(width_px / dpi, total_height_px / dpi),
        dpi=dpi,
        facecolor=style.background,
    )
    map_frac = map_height_px / total_height_px
    ax_cam = fig.add_axes([0.0, map_frac, 1.0, 1.0 - map_frac])
    ax_map = fig.add_axes([0.0, 0.0, 1.0, map_frac])
    for ax in (ax_cam, ax_map):
        ax.set_facecolor(style.background)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_map.set_aspect("equal")
    fig.add_artist(
        plt.Line2D([0, 1], [map_frac, map_frac], color=style.divider, linewidth=1.0)
    )

    camera.render_image_at_time(int(timestamps_us[0]), ax_cam)

    # --- static map drawing, laid out once in world coordinates -------------
    static_artists: list[plt.Artist] = []

    if style.road_fill is not None:
        lanes = sim_result.vec_map.elements[maps.vec_map_elements.MapElementType.ROAD_LANE]
        polygons = [p for p in (_lane_polygon(lane) for lane in lanes.values()) if p is not None]
        # Dilate and erode to close the hairline gaps where lanes that should
        # touch were digitised a few centimetres apart.
        surface_path = _compound_path(shapely.union_all(polygons).buffer(0.3).buffer(-0.3))
        if surface_path is not None:
            fill = PathPatch(
                surface_path, facecolor=style.road_fill, edgecolor="none", zorder=0
            )
            ax_map.add_patch(fill)
            static_artists.append(fill)

    for boundary in lane_boundaries(sim_result.vec_map):
        xy = np.asarray(boundary.coords)
        (line,) = ax_map.plot(
            xy[:, 0],
            xy[:, 1],
            color=style.lane_line,
            linewidth=style.lane_line_width,
            alpha=style.lane_line_alpha,
            solid_capstyle="round",
            zorder=1,
        )
        static_artists.append(line)

    ego_trajectory = sim_result.actor_trajectories[EGO_ID].interpolate_to_timestamps(timestamps_us)
    ego_xy = np.asarray(ego_trajectory.positions)[:, :2]
    (driven,) = ax_map.plot(
        [], [], color=style.trail, linewidth=style.trail_width,
        solid_capstyle="round", zorder=4,
    )

    agent_patches: list[plt.Artist] = []

    def draw_frame(frame_index: int) -> list[plt.Artist]:
        time = int(timestamps_us[frame_index])

        # Turn the world so the car drives upwards, the way the camera looks.
        yaw = sim_result.actor_polygons.get_yaw_for_agent_at_time(EGO_ID, time)
        rotation = np.pi / 2 - yaw
        world_to_axes = transforms.Affine2D().rotate(rotation) + ax_map.transData
        for artist in static_artists:
            artist.set_transform(world_to_axes)
        driven.set_data(ego_xy[: frame_index + 1, 0], ego_xy[: frame_index + 1, 1])
        driven.set_transform(world_to_axes)

        for patch in agent_patches:
            patch.remove()
        agent_patches.clear()

        polygons_at_time = sim_result.actor_polygons.get_polygons_at_time(time)
        for agent_id, polygon in zip(polygons_at_time.agent_ids, polygons_at_time.bbox_polygons):
            is_ego = agent_id == EGO_ID
            min_x, min_y, max_x, max_y = polygon.bounds
            is_person = (
                not is_ego
                and max(max_x - min_x, max_y - min_y) <= style.pedestrian_max_extent_m
            )
            if is_person:
                centre = polygon.centroid
                patch = plt.Circle(
                    (centre.x, centre.y),
                    radius=style.pedestrian_radius_m,
                    facecolor=style.pedestrian,
                    edgecolor="none",
                    alpha=style.pedestrian_alpha,
                    zorder=5,
                    transform=world_to_axes,
                )
            else:
                path = _compound_path(polygon)
                if path is None:
                    continue
                patch = PathPatch(
                    path,
                    facecolor=style.ego if is_ego else style.actor,
                    edgecolor=style.vehicle_edge,
                    linewidth=0.6,
                    alpha=1.0 if is_ego else style.actor_alpha,
                    zorder=10 if is_ego else 6,
                    transform=world_to_axes,
                )
            ax_map.add_patch(patch)
            agent_patches.append(patch)

        # The map is drawn to scale, so metres per pixel is the same in both
        # directions: the radius fixes the vertical span and the panel's shape
        # decides how much ground that leaves across it.
        radius = cfg.video.map_video.map_radius_m
        centre = transforms.Affine2D().rotate(rotation).transform(ego_xy[frame_index])
        span_y = 2 * radius
        span_x = span_y * width_px / map_height_px
        ax_map.set_xlim(
            centre[0] - span_x * style.map_ego_x_frac,
            centre[0] + span_x * (1 - style.map_ego_x_frac),
        )
        ax_map.set_ylim(
            centre[1] - span_y * style.map_ego_y_frac,
            centre[1] + span_y * (1 - style.map_ego_y_frac),
        )

        camera.render_image_at_time(time, ax_cam)
        return [*static_artists, driven, *agent_patches]

    anim = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=frame_indices if frame_indices is not None else len(timestamps_us),
        blit=False,
    )
    anim.save(output_path, fps=_fps_from_timestamps(timestamps_us), dpi=dpi, writer="ffmpeg")
    plt.close(fig)
    logger.info("Wrote presentation video %s", output_path)
