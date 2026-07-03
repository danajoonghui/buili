from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .presentation_renderer import (
    IsoRenderer,
    _add_dimension,
    _add_floor_grid,
    _add_tag,
    _font,
)

PointPX = tuple[float, float]
ObjectKind = Literal[
    "bathtub",
    "cabinet_run",
    "ceiling_light",
    "door",
    "duplex_outlet",
    "fixture_tag",
    "sink",
    "shower",
    "smoke_detector",
    "switch",
    "toilet",
    "washer_dryer",
    "water_heater",
    "window",
]


@dataclass(frozen=True)
class SourceTransform:
    """Single source-pixel to scene-meter transform for geometry, labels, tags, and QA."""

    width_px: float
    height_px: float
    meters_per_px: float = 0.016
    offset_x_px: float = 0.0
    offset_y_px: float = 0.0

    def point(self, point_px: PointPX) -> tuple[float, float]:
        px, py = point_px
        return (
            (px - self.offset_x_px) * self.meters_per_px,
            (self.height_px - py + self.offset_y_px) * self.meters_per_px,
        )

    def distance(self, value_px: float) -> float:
        return value_px * self.meters_per_px


@dataclass(frozen=True)
class SemanticWall:
    id: str
    start_px: PointPX
    end_px: PointPX
    wall_type: Literal["exterior", "interior", "partition"] = "interior"
    source_note: str = ""


@dataclass(frozen=True)
class SemanticOpening:
    id: str
    kind: Literal["door", "window"]
    center_px: PointPX
    length_px: float
    angle_deg: float
    mark: str = ""
    source_note: str = ""


@dataclass(frozen=True)
class SemanticObject:
    id: str
    kind: ObjectKind
    center_px: PointPX
    width_px: float
    depth_px: float
    angle_deg: float = 0.0
    label: str = ""
    source_note: str = ""


@dataclass(frozen=True)
class SemanticRoomLabel:
    name: str
    number: str
    center_px: PointPX
    source_text: str


@dataclass(frozen=True)
class SemanticIssueTag:
    label: str
    center_px: PointPX
    color: Literal["red", "blue", "yellow"]
    side: Literal["left", "right"] = "right"
    source_note: str = ""


@dataclass(frozen=True)
class SemanticDimension:
    start_px: PointPX
    end_px: PointPX
    label: str


@dataclass(frozen=True)
class SemanticScene:
    source_pdf: str
    source_page_png: str
    source_crop_png: str
    transform: SourceTransform
    walls: list[SemanticWall]
    openings: list[SemanticOpening]
    objects: list[SemanticObject]
    labels: list[SemanticRoomLabel]
    tags: list[SemanticIssueTag]
    dimensions: list[SemanticDimension]
    source_scope: str

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["counts"] = {
            "walls": len(self.walls),
            "openings": len(self.openings),
            "objects": len(self.objects),
            "labels": len(self.labels),
            "tags": len(self.tags),
            "dimensions": len(self.dimensions),
        }
        return payload


def _rotated_corners(
    center: tuple[float, float],
    width: float,
    depth: float,
    angle_deg: float,
) -> list[tuple[float, float]]:
    cx, cz = center
    angle = math.radians(angle_deg)
    ux = math.cos(angle)
    uz = math.sin(angle)
    vx = -uz
    vz = ux
    corners: list[tuple[float, float]] = []
    for sx, sz in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]:
        corners.append(
            (
                cx + ux * width * sx + vx * depth * sz,
                cz + uz * width * sx + vz * depth * sz,
            )
        )
    return corners


def _rect_points(
    center: tuple[float, float],
    width: float,
    depth: float,
    angle_deg: float,
    *,
    offset_x: float = 0.0,
    offset_z: float = 0.0,
) -> list[tuple[float, float]]:
    cx, cz = center
    angle = math.radians(angle_deg)
    ux = math.cos(angle)
    uz = math.sin(angle)
    vx = -uz
    vz = ux
    shifted = (cx + ux * offset_x + vx * offset_z, cz + uz * offset_x + vz * offset_z)
    return _rotated_corners(shifted, width, depth, angle_deg)


def _ellipse_points(
    center: tuple[float, float],
    width: float,
    depth: float,
    angle_deg: float,
    *,
    segments: int = 14,
    offset_x: float = 0.0,
    offset_z: float = 0.0,
) -> list[tuple[float, float]]:
    cx, cz = center
    angle = math.radians(angle_deg)
    ux = math.cos(angle)
    uz = math.sin(angle)
    vx = -uz
    vz = ux
    points: list[tuple[float, float]] = []
    for index in range(segments):
        theta = math.tau * index / segments
        local_x = offset_x + math.cos(theta) * width / 2
        local_z = offset_z + math.sin(theta) * depth / 2
        points.append((cx + ux * local_x + vx * local_z, cz + uz * local_x + vz * local_z))
    return points


def _add_prism(
    renderer: IsoRenderer,
    footprint: list[tuple[float, float]],
    *,
    y0: float,
    y1: float,
    side: tuple[int, int, int, int],
    top: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] | None,
    layer: int,
) -> None:
    top_points = [(x, y1, z) for x, z in footprint]
    bottom_points = [(x, y0, z) for x, z in footprint]
    renderer.add_surface(top_points, top, outline=outline, layer=layer + 2)
    for index, start in enumerate(footprint):
        end = footprint[(index + 1) % len(footprint)]
        renderer.add_surface(
            [
                (start[0], y0, start[1]),
                (end[0], y0, end[1]),
                (end[0], y1, end[1]),
                (start[0], y1, start[1]),
            ],
            side,
            outline=outline,
            layer=layer,
        )
    renderer.add_surface(bottom_points, (0, 0, 0, 0), layer=layer - 2)


def _add_low_box(
    renderer: IsoRenderer,
    transform: SourceTransform,
    obj: SemanticObject,
    *,
    height: float,
    side: tuple[int, int, int, int],
    top: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] | None = None,
    layer: int = 5,
) -> None:
    center = transform.point(obj.center_px)
    width = transform.distance(obj.width_px)
    depth = transform.distance(obj.depth_px)
    corners = _rect_points(center, width, depth, -obj.angle_deg)
    _add_prism(
        renderer,
        corners,
        y0=0.0,
        y1=height,
        side=side,
        top=top,
        outline=outline,
        layer=layer,
    )


def _add_asset_box(
    renderer: IsoRenderer,
    center: tuple[float, float],
    width: float,
    depth: float,
    angle_deg: float,
    *,
    y0: float,
    y1: float,
    side: tuple[int, int, int, int],
    top: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] | None = (112, 118, 112, 120),
    layer: int = 7,
    offset_x: float = 0.0,
    offset_z: float = 0.0,
) -> None:
    _add_prism(
        renderer,
        _rect_points(center, width, depth, angle_deg, offset_x=offset_x, offset_z=offset_z),
        y0=y0,
        y1=y1,
        side=side,
        top=top,
        outline=outline,
        layer=layer,
    )


def _add_asset_ellipse(
    renderer: IsoRenderer,
    center: tuple[float, float],
    width: float,
    depth: float,
    angle_deg: float,
    *,
    y0: float,
    y1: float,
    side: tuple[int, int, int, int],
    top: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] | None = (112, 118, 112, 120),
    layer: int = 8,
    offset_x: float = 0.0,
    offset_z: float = 0.0,
    segments: int = 14,
) -> None:
    _add_prism(
        renderer,
        _ellipse_points(
            center,
            width,
            depth,
            angle_deg,
            segments=segments,
            offset_x=offset_x,
            offset_z=offset_z,
        ),
        y0=y0,
        y1=y1,
        side=side,
        top=top,
        outline=outline,
        layer=layer,
    )


def _draw_projected_polyline(
    renderer: IsoRenderer,
    points: list[tuple[float, float]],
    *,
    y: float,
    fill: tuple[int, int, int, int],
    width: int = 1,
    closed: bool = False,
) -> None:
    def draw(draw, project) -> None:  # type: ignore[no-untyped-def]
        projected = [project((x, y, z)) for x, z in points]
        if closed and projected:
            projected.append(projected[0])
        draw.line(projected, fill=fill, width=width)

    renderer.post_draw.append(draw)


def _add_object_label(
    renderer: IsoRenderer,
    text: str,
    point: tuple[float, float],
    *,
    y: float = 0.08,
) -> None:
    def draw(draw, project) -> None:  # type: ignore[no-untyped-def]
        font = _font(15, bold=True)
        x, z = point
        sx, sy = project((x, y, z))
        width = draw.textlength(text, font=font)
        draw.text((sx - width / 2, sy - 7), text, fill=(43, 46, 43, 170), font=font)

    renderer.post_draw.append(draw)


def _add_room_label(
    renderer: IsoRenderer,
    label: str,
    detail: str,
    x: float,
    z: float,
) -> None:
    def draw(draw, project) -> None:  # type: ignore[no-untyped-def]
        label_size = 20 if len(label) <= 10 else 18
        font_label = _font(label_size)
        font_detail = _font(14)
        sx, sy = project((x, 0.035, z))
        width = draw.textlength(label, font=font_label)
        draw.text((sx - width / 2, sy - 14), label, fill=(31, 34, 32, 220), font=font_label)
        if detail:
            detail_width = draw.textlength(detail, font=font_detail)
            pad_x = 5
            box = [
                sx - detail_width / 2 - pad_x,
                sy + 11,
                sx + detail_width / 2 + pad_x,
                sy + 30,
            ]
            draw.rectangle(box, outline=(63, 68, 63, 105), width=1)
            draw.text(
                (sx - detail_width / 2, sy + 12),
                detail,
                fill=(31, 34, 32, 185),
                font=font_detail,
            )

    renderer.post_draw.append(draw)


def _add_window(
    renderer: IsoRenderer, transform: SourceTransform, opening: SemanticOpening
) -> None:
    x, z = transform.point(opening.center_px)
    length = transform.distance(opening.length_px)
    angle = math.radians(-opening.angle_deg)
    dx = math.cos(angle) * length / 2
    dz = math.sin(angle) * length / 2
    renderer.add_wall(
        x - dx,
        z - dz,
        x + dx,
        z + dz,
        thickness=0.045,
        height=1.35,
        side=(118, 151, 158, 68),
        cap=(34, 41, 43, 235),
        layer=23,
    )
    if opening.mark:
        _add_object_label(renderer, opening.mark, (x, z), y=1.46)


def _object_frame(
    transform: SourceTransform, obj: SemanticObject
) -> tuple[tuple[float, float], float, float, float]:
    center = transform.point(obj.center_px)
    width = transform.distance(obj.width_px)
    depth = transform.distance(obj.depth_px)
    angle = -obj.angle_deg
    return center, width, depth, angle


def _add_door(renderer: IsoRenderer, transform: SourceTransform, opening: SemanticOpening) -> None:
    x, z = transform.point(opening.center_px)
    length = transform.distance(opening.length_px)
    angle = math.radians(-opening.angle_deg)
    dx = math.cos(angle) * length / 2
    dz = math.sin(angle) * length / 2
    renderer.add_wall(
        x - dx,
        z - dz,
        x + dx,
        z + dz,
        thickness=0.035,
        height=1.78,
        side=(214, 219, 215, 118),
        cap=(82, 88, 84, 185),
        layer=24,
    )

    def draw(draw, project) -> None:  # type: ignore[no-untyped-def]
        anchor = np.array(project((x - dx, 0.04, z - dz)))
        leaf = np.array(project((x + dx, 0.04, z + dz)))
        draw.line([tuple(anchor), tuple(leaf)], fill=(57, 64, 60, 132), width=2)
        center = np.array([x, z], dtype=np.float64)
        hinge = np.array([x - dx, z - dz], dtype=np.float64)
        radius = float(np.linalg.norm(np.array([dx, dz]) * 2.0))
        if radius <= 0:
            return
        start_angle = math.atan2((center - hinge)[1], (center - hinge)[0])
        swing_points = []
        for index in range(11):
            theta = start_angle + math.radians(72) * index / 10
            px = hinge[0] + math.cos(theta) * radius
            pz = hinge[1] + math.sin(theta) * radius
            swing_points.append(project((float(px), 0.045, float(pz))))
        draw.line(swing_points, fill=(57, 64, 60, 82), width=1)
        draw.ellipse(
            [anchor[0] - 2.5, anchor[1] - 2.5, anchor[0] + 2.5, anchor[1] + 2.5],
            fill=(57, 64, 60, 150),
        )

    renderer.post_draw.append(draw)


def _add_toilet(renderer: IsoRenderer, transform: SourceTransform, obj: SemanticObject) -> None:
    center, width, depth, angle = _object_frame(transform, obj)
    porcelain_side = (222, 225, 220, 255)
    porcelain_top = (250, 251, 247, 255)
    outline = (104, 112, 107, 120)
    _add_asset_box(
        renderer,
        center,
        width * 0.86,
        depth * 0.35,
        angle,
        y0=0.0,
        y1=0.42,
        side=porcelain_side,
        top=porcelain_top,
        outline=outline,
        layer=8,
        offset_z=depth * 0.28,
    )
    _add_asset_ellipse(
        renderer,
        center,
        width * 0.84,
        depth * 0.66,
        angle,
        y0=0.04,
        y1=0.31,
        side=porcelain_side,
        top=porcelain_top,
        outline=outline,
        layer=9,
        offset_z=-depth * 0.12,
    )
    _add_asset_ellipse(
        renderer,
        center,
        width * 0.46,
        depth * 0.33,
        angle,
        y0=0.315,
        y1=0.322,
        side=(194, 204, 202, 230),
        top=(213, 225, 223, 230),
        outline=(95, 105, 102, 125),
        layer=10,
        offset_z=-depth * 0.13,
    )
    _add_asset_box(
        renderer,
        center,
        width * 0.46,
        depth * 0.28,
        angle,
        y0=0.0,
        y1=0.22,
        side=porcelain_side,
        top=porcelain_top,
        outline=outline,
        layer=7,
        offset_z=-depth * 0.42,
    )


def _add_bathtub(renderer: IsoRenderer, transform: SourceTransform, obj: SemanticObject) -> None:
    center, width, depth, angle = _object_frame(transform, obj)
    _add_asset_box(
        renderer,
        center,
        width,
        depth,
        angle,
        y0=0.0,
        y1=0.48,
        side=(218, 222, 217, 255),
        top=(246, 247, 243, 255),
        outline=(109, 116, 111, 145),
        layer=7,
    )
    _add_asset_box(
        renderer,
        center,
        width * 0.76,
        depth * 0.58,
        angle,
        y0=0.49,
        y1=0.5,
        side=(187, 201, 200, 220),
        top=(211, 226, 224, 225),
        outline=(95, 108, 106, 130),
        layer=10,
    )
    _add_asset_box(
        renderer,
        center,
        width * 0.12,
        depth * 0.12,
        angle,
        y0=0.5,
        y1=0.58,
        side=(132, 138, 134, 230),
        top=(165, 170, 165, 245),
        outline=(88, 94, 90, 140),
        layer=11,
        offset_x=-width * 0.33,
        offset_z=-depth * 0.2,
    )


def _add_shower(renderer: IsoRenderer, transform: SourceTransform, obj: SemanticObject) -> None:
    center, width, depth, angle = _object_frame(transform, obj)
    _add_asset_box(
        renderer,
        center,
        width,
        depth,
        angle,
        y0=0.0,
        y1=0.18,
        side=(211, 216, 211, 255),
        top=(238, 241, 237, 255),
        outline=(90, 98, 94, 145),
        layer=6,
    )
    corners = _rect_points(center, width, depth, angle)
    renderer.add_surface(
        [(x, 1.55, z) for x, z in corners],
        (126, 145, 144, 40),
        outline=(74, 83, 82, 90),
        layer=14,
    )
    for side_offset in (-width * 0.38, width * 0.38):
        panel = _rect_points(
            center,
            width * 0.04,
            depth * 0.92,
            angle,
            offset_x=side_offset,
        )
        renderer.add_surface(
            [(x, 1.2, z) for x, z in panel],
            (120, 144, 145, 62),
            outline=(72, 85, 84, 88),
            layer=15,
        )
    _draw_projected_polyline(
        renderer,
        _rect_points(center, width * 0.5, depth * 0.5, angle),
        y=0.2,
        fill=(80, 88, 84, 120),
        closed=True,
    )


def _add_cabinet(renderer: IsoRenderer, transform: SourceTransform, obj: SemanticObject) -> None:
    center, width, depth, angle = _object_frame(transform, obj)
    _add_asset_box(
        renderer,
        center,
        width,
        depth,
        angle,
        y0=0.0,
        y1=0.72,
        side=(136, 106, 74, 245),
        top=(151, 121, 84, 255),
        outline=(80, 68, 56, 140),
        layer=6,
    )
    _add_asset_box(
        renderer,
        center,
        width * 1.05,
        depth * 1.08,
        angle,
        y0=0.72,
        y1=0.8,
        side=(197, 198, 190, 245),
        top=(228, 229, 222, 255),
        outline=(103, 107, 101, 125),
        layer=9,
    )
    if max(width, depth) > 0.9:
        divisions = 4
        for index in range(1, divisions):
            offset = -width / 2 + width * index / divisions
            line = _rect_points(
                center,
                0.012,
                depth * 0.98,
                angle,
                offset_x=offset,
            )
            _draw_projected_polyline(renderer, line, y=0.81, fill=(75, 62, 49, 115), width=1)
    if obj.label:
        _add_object_label(renderer, obj.label.upper(), center, y=0.86)


def _add_appliance(renderer: IsoRenderer, transform: SourceTransform, obj: SemanticObject) -> None:
    center, width, depth, angle = _object_frame(transform, obj)
    if obj.kind == "water_heater":
        radius = min(width, depth) * 0.78
        _add_asset_ellipse(
            renderer,
            center,
            radius,
            radius,
            angle,
            y0=0.0,
            y1=1.05,
            side=(189, 194, 188, 245),
            top=(229, 232, 225, 255),
            outline=(88, 96, 91, 130),
            layer=8,
            segments=16,
        )
        _add_asset_box(
            renderer,
            center,
            radius * 0.18,
            radius * 0.18,
            angle,
            y0=1.05,
            y1=1.32,
            side=(135, 141, 136, 230),
            top=(168, 174, 169, 245),
            outline=(88, 96, 91, 120),
            layer=11,
            offset_z=radius * 0.2,
        )
        _add_object_label(renderer, "WH", center, y=1.37)
        return

    _add_asset_box(
        renderer,
        center,
        width,
        depth,
        angle,
        y0=0.0,
        y1=0.82,
        side=(190, 194, 188, 245),
        top=(227, 230, 224, 255),
        outline=(95, 101, 97, 130),
        layer=7,
    )
    _add_asset_ellipse(
        renderer,
        center,
        width * 0.58,
        depth * 0.58,
        angle,
        y0=0.835,
        y1=0.845,
        side=(118, 137, 141, 225),
        top=(143, 164, 168, 235),
        outline=(75, 84, 86, 135),
        layer=11,
    )
    if obj.label:
        _add_object_label(renderer, obj.label.upper(), center, y=0.93)


def _add_sink(renderer: IsoRenderer, transform: SourceTransform, obj: SemanticObject) -> None:
    center, width, depth, angle = _object_frame(transform, obj)
    _add_asset_box(
        renderer,
        center,
        width,
        depth,
        angle,
        y0=0.0,
        y1=0.78,
        side=(139, 109, 77, 240),
        top=(226, 227, 220, 255),
        outline=(83, 70, 57, 130),
        layer=7,
    )
    _add_asset_ellipse(
        renderer,
        center,
        width * 0.62,
        depth * 0.48,
        angle,
        y0=0.785,
        y1=0.795,
        side=(194, 207, 205, 230),
        top=(218, 232, 230, 240),
        outline=(91, 104, 102, 130),
        layer=11,
    )
    _add_asset_box(
        renderer,
        center,
        width * 0.08,
        depth * 0.15,
        angle,
        y0=0.8,
        y1=0.98,
        side=(125, 132, 128, 235),
        top=(164, 170, 165, 245),
        outline=(82, 88, 84, 120),
        layer=12,
        offset_z=-depth * 0.18,
    )


def _add_fixture_tag(
    renderer: IsoRenderer, transform: SourceTransform, obj: SemanticObject
) -> None:
    _add_low_box(
        renderer,
        transform,
        obj,
        height=0.12,
        side=(238, 240, 235, 255),
        top=(255, 255, 250, 255),
        outline=(95, 103, 98, 140),
        layer=5,
    )
    if obj.label:
        _add_object_label(renderer, obj.label, transform.point(obj.center_px), y=0.18)


def _add_ceiling_light(
    renderer: IsoRenderer, transform: SourceTransform, obj: SemanticObject
) -> None:
    center, width, depth, angle = _object_frame(transform, obj)
    _add_asset_box(
        renderer,
        center,
        width * 1.25,
        depth * 0.22,
        angle,
        y0=0.02,
        y1=0.08,
        side=(235, 236, 228, 255),
        top=(255, 255, 244, 255),
        outline=(104, 110, 105, 120),
        layer=5,
    )
    _add_asset_box(
        renderer,
        center,
        width * 0.22,
        depth * 1.25,
        angle,
        y0=0.02,
        y1=0.08,
        side=(235, 236, 228, 255),
        top=(255, 255, 244, 255),
        outline=(104, 110, 105, 120),
        layer=5,
    )
    if obj.label:
        _add_object_label(renderer, obj.label.upper(), center, y=0.14)


def _add_wall_device(
    renderer: IsoRenderer, transform: SourceTransform, obj: SemanticObject
) -> None:
    center, width, depth, angle = _object_frame(transform, obj)
    if obj.kind == "smoke_detector":
        _add_asset_ellipse(
            renderer,
            center,
            max(width, depth),
            max(width, depth),
            angle,
            y0=0.02,
            y1=0.1,
            side=(226, 229, 224, 255),
            top=(249, 250, 245, 255),
            outline=(92, 100, 96, 135),
            layer=5,
            segments=14,
        )
        label = obj.label or "SD"
    else:
        _add_asset_box(
            renderer,
            center,
            max(width, 0.18),
            max(depth, 0.1),
            angle,
            y0=0.02,
            y1=0.12,
            side=(229, 231, 225, 255),
            top=(255, 255, 250, 255),
            outline=(83, 91, 87, 135),
            layer=5,
        )
        label = obj.label or ("SW" if obj.kind == "switch" else "REC")
    _add_object_label(renderer, label.upper(), center, y=0.18)


def _opening_interval_on_wall(
    wall: SemanticWall, opening: SemanticOpening
) -> tuple[float, float] | None:
    sx, sy = wall.start_px
    ex, ey = wall.end_px
    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None
    ux = dx / length
    uy = dy / length
    ox, oy = opening.center_px
    along = (ox - sx) * ux + (oy - sy) * uy
    closest = (sx + ux * along, sy + uy * along)
    distance = math.hypot(ox - closest[0], oy - closest[1])
    if opening.kind == "door":
        tolerance = max(38.0, opening.length_px * 0.38)
        gap_half = max(34.0, opening.length_px * 0.52)
        endpoint_margin = gap_half * 0.75
    else:
        tolerance = max(22.0, opening.length_px * 0.18)
        gap_half = max(20.0, opening.length_px * 0.5)
        endpoint_margin = gap_half * 0.35
    if distance > tolerance or along < -endpoint_margin or along > length + endpoint_margin:
        return None
    return max(0.0, along - gap_half), min(length, along + gap_half)


def _point_along_wall(wall: SemanticWall, distance_px: float) -> PointPX:
    sx, sy = wall.start_px
    ex, ey = wall.end_px
    length = math.hypot(ex - sx, ey - sy)
    if length <= 1e-6:
        return wall.start_px
    ratio = distance_px / length
    return sx + (ex - sx) * ratio, sy + (ey - sy) * ratio


def _split_wall_for_openings(
    wall: SemanticWall, openings: list[SemanticOpening]
) -> tuple[list[tuple[PointPX, PointPX]], int]:
    sx, sy = wall.start_px
    ex, ey = wall.end_px
    length = math.hypot(ex - sx, ey - sy)
    if length <= 1e-6:
        return [], 0
    intervals: list[tuple[float, float]] = []
    for opening in openings:
        interval = _opening_interval_on_wall(wall, opening)
        if interval and interval[1] - interval[0] >= 8.0:
            intervals.append(interval)
    if not intervals:
        return [(wall.start_px, wall.end_px)], 0

    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 2.0:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    segments: list[tuple[PointPX, PointPX]] = []
    cursor = 0.0
    min_segment = 14.0
    for start, end in merged:
        if start - cursor >= min_segment:
            segments.append((_point_along_wall(wall, cursor), _point_along_wall(wall, start)))
        cursor = max(cursor, end)
    if length - cursor >= min_segment:
        segments.append((_point_along_wall(wall, cursor), _point_along_wall(wall, length)))
    return segments, len(merged)


def _scene_wall_render_segments(
    scene: SemanticScene,
) -> tuple[list[tuple[SemanticWall, PointPX, PointPX]], int]:
    render_segments: list[tuple[SemanticWall, PointPX, PointPX]] = []
    gap_count = 0
    for wall in scene.walls:
        segments, gaps = _split_wall_for_openings(wall, scene.openings)
        gap_count += gaps
        for start, end in segments:
            render_segments.append((wall, start, end))
    return render_segments, gap_count


def _scene_source_bounds(scene: SemanticScene) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for wall in scene.walls:
        xs.extend([wall.start_px[0], wall.end_px[0]])
        ys.extend([wall.start_px[1], wall.end_px[1]])
    for opening in scene.openings:
        xs.append(opening.center_px[0])
        ys.append(opening.center_px[1])
    for obj in scene.objects:
        half_w = obj.width_px / 2
        half_d = obj.depth_px / 2
        xs.extend([obj.center_px[0] - half_w, obj.center_px[0] + half_w])
        ys.extend([obj.center_px[1] - half_d, obj.center_px[1] + half_d])
    for label in scene.labels:
        xs.append(label.center_px[0])
        ys.append(label.center_px[1])
    for dimension in scene.dimensions:
        xs.extend([dimension.start_px[0], dimension.end_px[0]])
        ys.extend([dimension.start_px[1], dimension.end_px[1]])

    if not xs or not ys:
        return 0.0, 0.0, scene.transform.width_px, scene.transform.height_px
    pad = max(28.0, min(scene.transform.width_px, scene.transform.height_px) * 0.025)
    return (
        max(0.0, min(xs) - pad),
        max(0.0, min(ys) - pad),
        min(scene.transform.width_px, max(xs) + pad),
        min(scene.transform.height_px, max(ys) + pad),
    )


def _add_floor_texture_lines(
    renderer: IsoRenderer,
    scene: SemanticScene,
    wall_segments: list[tuple[SemanticWall, PointPX, PointPX]],
) -> None:
    transform = scene.transform

    def draw(draw, project) -> None:  # type: ignore[no-untyped-def]
        for wall, start_px, end_px in wall_segments:
            a = transform.point(start_px)
            b = transform.point(end_px)
            color = (36, 38, 36, 54) if wall.wall_type == "exterior" else (60, 64, 60, 42)
            draw.line(
                [project((a[0], 0.016, a[1])), project((b[0], 0.016, b[1]))],
                fill=color,
                width=1,
            )

    renderer.post_draw.append(draw)


def build_maricopa_source_aligned_scene() -> SemanticScene:
    """Semantic seed built from the public Maricopa County sample floor plan PDF.

    The scene is intentionally explicit: each object carries source-pixel coordinates from the
    reviewed public drawing. The production detector/OCR path fills the same schema.
    """

    transform = SourceTransform(width_px=1225, height_px=900, meters_per_px=0.016)

    walls = [
        # Garage and west bedrooms.
        SemanticWall("w_garage_w", (50, 900), (50, 525), "exterior", "garage west wall"),
        SemanticWall("w_garage_s", (50, 900), (493, 900), "exterior", "garage south wall"),
        SemanticWall("w_garage_e", (493, 900), (493, 525), "interior", "garage east wall"),
        SemanticWall("w_garage_n", (50, 525), (493, 525), "interior", "garage north wall"),
        SemanticWall("w_bed2_w", (50, 525), (50, 326), "exterior", "west bedroom wall"),
        SemanticWall("w_bed2_n", (50, 326), (300, 326), "exterior", "west bedroom north wall"),
        SemanticWall("w_bed2_e", (300, 326), (300, 525), "interior", "west bedroom east wall"),
        SemanticWall("w_upperbed_w", (288, 286), (288, 76), "exterior", "upper bedroom west wall"),
        SemanticWall("w_upperbed_n", (288, 76), (498, 76), "exterior", "upper bedroom north wall"),
        SemanticWall("w_upperbed_e", (498, 76), (498, 330), "interior", "upper bedroom east wall"),
        SemanticWall(
            "w_upperbed_s", (288, 286), (498, 286), "interior", "upper bedroom south wall"
        ),
        # Patio, breakfast, living and dining core.
        SemanticWall(
            "w_patio_n", (498, 25), (950, 25), "partition", "patio dashed/solid north edge"
        ),
        SemanticWall("w_patio_w", (498, 25), (498, 235), "exterior", "patio west edge"),
        SemanticWall(
            "w_patio_s", (498, 235), (950, 235), "exterior", "patio south wall with windows"
        ),
        SemanticWall("w_patio_e", (950, 25), (950, 235), "exterior", "patio east edge"),
        SemanticWall("w_living_e", (950, 235), (950, 585), "interior", "living east wall"),
        SemanticWall("w_living_s", (675, 735), (858, 735), "interior", "dining south wall"),
        SemanticWall(
            "w_kitchen_w", (493, 410), (493, 760), "interior", "kitchen/laundry west wall"
        ),
        SemanticWall(
            "w_kitchen_e", (668, 340), (668, 760), "interior", "kitchen east appliance wall"
        ),
        SemanticWall(
            "w_kitchen_diag_1", (590, 340), (635, 385), "interior", "angled kitchen island"
        ),
        SemanticWall(
            "w_kitchen_diag_2", (635, 385), (650, 505), "interior", "angled kitchen partition"
        ),
        SemanticWall("w_laundry_s", (493, 760), (615, 760), "interior", "laundry south wall"),
        # Right-side suite, foyer and office.
        SemanticWall("w_master_n", (950, 27), (1182, 27), "exterior", "master bedroom north wall"),
        SemanticWall("w_master_e", (1182, 27), (1182, 294), "exterior", "master bedroom east wall"),
        SemanticWall(
            "w_master_s", (950, 294), (1182, 294), "interior", "master bedroom south wall"
        ),
        SemanticWall("w_mbath_e", (1182, 294), (1182, 637), "exterior", "master bath east wall"),
        SemanticWall("w_mbath_s", (970, 637), (1182, 637), "interior", "master bath south wall"),
        SemanticWall("w_closet_w", (970, 388), (970, 637), "interior", "closet west wall"),
        SemanticWall(
            "w_shower_diag", (1035, 590), (1088, 540), "interior", "closet/shower diagonal"
        ),
        SemanticWall("w_foyer_w", (858, 585), (858, 735), "interior", "foyer west wall"),
        SemanticWall("w_foyer_e", (970, 585), (970, 735), "interior", "foyer east wall"),
        SemanticWall("w_entry_s", (858, 790), (970, 790), "exterior", "entry south wall"),
        SemanticWall("w_office_w", (970, 637), (970, 872), "interior", "office west wall"),
        SemanticWall("w_office_e", (1182, 637), (1182, 872), "exterior", "office east wall"),
        SemanticWall("w_office_s", (970, 872), (1182, 872), "exterior", "office south wall"),
    ]

    openings = [
        SemanticOpening(
            "win_bed_west", "window", (165, 326), 115, 0, "3040SH", "west bedroom window"
        ),
        SemanticOpening(
            "win_bed_upper", "window", (390, 76), 100, 0, "3040SH", "upper bedroom window"
        ),
        SemanticOpening(
            "win_patio_living_1",
            "window",
            (545, 235),
            90,
            0,
            "3040SH",
            "breakfast/patio window",
        ),
        SemanticOpening(
            "win_patio_living_2",
            "window",
            (765, 235),
            165,
            0,
            "3040SH",
            "living/patio windows",
        ),
        SemanticOpening(
            "win_master_n", "window", (1085, 27), 160, 0, "3040SH", "master north windows"
        ),
        SemanticOpening(
            "win_office_s", "window", (1080, 872), 150, 0, "3040SH", "office south window"
        ),
        SemanticOpening(
            "win_dining_s", "window", (764, 735), 135, 0, "3040SH", "dining south window"
        ),
        SemanticOpening("door_garage_w", "door", (50, 585), 82, 72, "", "garage exterior door"),
        SemanticOpening("door_bed2_hall", "door", (322, 360), 72, 34, "", "bedroom to hall"),
        SemanticOpening(
            "door_upperbed_hall", "door", (485, 314), 74, -80, "", "upper bedroom door"
        ),
        SemanticOpening(
            "door_patio_breakfast", "door", (625, 235), 85, -78, "", "patio swing door"
        ),
        SemanticOpening("door_patio_master", "door", (948, 247), 76, -45, "", "patio/living door"),
        SemanticOpening("door_master_bath", "door", (1118, 332), 85, 40, "", "master bath door"),
        SemanticOpening("door_foyer_entry", "door", (905, 680), 80, 0, "", "foyer door"),
        SemanticOpening("door_entry_out", "door", (860, 740), 78, 84, "", "entry exterior door"),
        SemanticOpening(
            "door_laundry_garage", "door", (493, 672), 76, -70, "", "garage/laundry door"
        ),
    ]

    objects = [
        SemanticObject("bath1_shower", "shower", (355, 452), 50, 115, 0, "shower"),
        SemanticObject("bath1_toilet", "toilet", (468, 452), 42, 52, 0, "wc"),
        SemanticObject("bath1_sink", "sink", (468, 512), 44, 46, 0, "sink"),
        SemanticObject("kitchen_cab_1", "cabinet_run", (558, 470), 42, 168, 0, "cabinet"),
        SemanticObject("kitchen_cab_2", "cabinet_run", (638, 498), 45, 250, 0, "range/dw"),
        SemanticObject("kitchen_island_sink", "sink", (625, 445), 52, 72, 0, "sink"),
        SemanticObject("laundry_cab", "cabinet_run", (555, 698), 120, 36, 0, "utility"),
        SemanticObject("laundry_washer", "washer_dryer", (625, 745), 44, 50, 0, "washer"),
        SemanticObject("laundry_dryer", "washer_dryer", (625, 805), 44, 50, 0, "dryer"),
        SemanticObject("laundry_wh", "water_heater", (525, 725), 52, 52, 0, "wh"),
        SemanticObject("master_toilet_1", "toilet", (1000, 315), 42, 52, 0, "wc"),
        SemanticObject("master_toilet_2", "toilet", (1172, 450), 42, 52, 90, "wc"),
        SemanticObject("master_toilet_3", "toilet", (1172, 560), 42, 52, 90, "wc"),
        SemanticObject("master_tub", "bathtub", (1120, 610), 110, 60, 0, "tub"),
        SemanticObject("master_shower", "shower", (1045, 565), 60, 95, 45, "shower"),
        SemanticObject("closet_drawers", "cabinet_run", (1040, 512), 75, 34, -45, "drawers"),
        SemanticObject("foyer_marker", "fixture_tag", (875, 585), 56, 42, 0, "arch"),
        SemanticObject("dining_marker", "fixture_tag", (845, 585), 56, 42, 0, "arch"),
    ]

    labels = [
        SemanticRoomLabel("BEDROOM", "12'-0\" x 11'-6\"", (180, 385), "BEDROOM 12'-0\" x 11'-6\""),
        SemanticRoomLabel("GARAGE", "24'-6\" x 23'-6\"", (290, 585), "GARAGE 24'-6\" x 23'-6\""),
        SemanticRoomLabel("BEDROOM", "12'-0\" x 12'-0\"", (395, 120), "BEDROOM 12'-0\" x 12'-0\""),
        SemanticRoomLabel("BATHROOM", "4'-6\" x 8'-0\"", (410, 455), "BATHROOM 4'-6\" x 8'-0\""),
        SemanticRoomLabel(
            "BREAKFAST", "NOOK 6'-6\" x 6'-0\"", (565, 255), "BREAKFAST NOOK 6'-6\" x 6'-0\""
        ),
        SemanticRoomLabel("KITCHEN", "5'-6\" x 15'-6\"", (575, 455), "KITCHEN 5'-6\" x 15'-6\""),
        SemanticRoomLabel("LAUNDRY", "4'-0\" x 9'-6\"", (565, 705), "LAUNDRY 4'-0\" x 9'-6\""),
        SemanticRoomLabel("PATIO", "27'-6\" x 12'-0\"", (710, 85), "PATIO 27'-6\" x 12'-0\""),
        SemanticRoomLabel(
            "LIVING ROOM", "16'-6\" x 21'-0\"", (800, 285), "LIVING ROOM 16'-6\" x 21'-0\""
        ),
        SemanticRoomLabel(
            "DINING ROOM", "11'-0\" x 10'-0\"", (745, 620), "DINING ROOM 11'-0\" x 10'-0\""
        ),
        SemanticRoomLabel("FOYER", "6'-0\" x 6'-0\"", (900, 620), "FOYER 6'-0\" x 6'-0\""),
        SemanticRoomLabel("ENTRY", "7'-0\" x 5'-6\"", (905, 730), "ENTRY 7'-0\" x 5'-6\""),
        SemanticRoomLabel(
            "MASTER BDRM", "13'-6\" x 15'-6\"", (1065, 75), "MASTER BDRM 13'-6\" x 15'-6\""
        ),
        SemanticRoomLabel("CLOSET", "5'-0\" x 9'-0\"", (1020, 435), "CLOSET 5'-0\" x 9'-0\""),
        SemanticRoomLabel(
            "MASTER BATH", "5'-0\" x 16'-6\"", (1115, 410), "MASTER BATH 5'-0\" x 16'-6\""
        ),
        SemanticRoomLabel("OFFICE", "12'-6\" x 12'-0\"", (1075, 675), "OFFICE 12'-6\" x 12'-0\""),
    ]

    tags = [
        SemanticIssueTag("ISSUE-01", (190, 338), "red", "right", "egress window verification"),
        SemanticIssueTag("RFI-01", (540, 360), "blue", "right", "kitchen appliance clearance"),
        SemanticIssueTag(
            "CLASH-01", (1040, 545), "yellow", "left", "shower/closet diagonal review"
        ),
        SemanticIssueTag("ISSUE-02", (1168, 610), "red", "left", "tub access verification"),
    ]

    dimensions = [
        SemanticDimension((50, 920), (493, 920), "24'-6\""),
        SemanticDimension((30, 525), (30, 900), "23'-6\""),
        SemanticDimension((498, 5), (950, 5), "27'-6\""),
        SemanticDimension((950, 7), (1182, 7), "13'-6\""),
        SemanticDimension((1188, 27), (1188, 294), "15'-6\""),
        SemanticDimension((970, 892), (1182, 892), "12'-6\""),
    ]

    return SemanticScene(
        source_pdf="data/sources/plan2field3d_house/maricopa_sample_floor_plan.pdf",
        source_page_png="data/sources/plan2field3d_house/maricopa_pages/page-1.png",
        source_crop_png="docs/plan2field3d_house_conversion/maricopa_floor_plan_crop.png",
        transform=transform,
        walls=walls,
        openings=openings,
        objects=objects,
        labels=labels,
        tags=tags,
        dimensions=dimensions,
        source_scope=(
            "Public Maricopa County floor plan sample; semantic geometry covers visible rooms, "
            "plumbing fixtures, kitchen/laundry fixtures, doors, windows, labels, and dimensions."
        ),
    )


def render_semantic_scene(scene: SemanticScene, output_png: Path) -> dict[str, object]:
    start = time.perf_counter()
    transform = scene.transform
    renderer = IsoRenderer(
        width=1780,
        height=1160,
        yaw_deg=-38,
        elevation_deg=50,
        supersample=1.55,
    )
    source_x0, source_y0, source_x1, source_y1 = _scene_source_bounds(scene)
    floor_x0, floor_z0 = transform.point((source_x0, source_y1))
    floor_x1, floor_z1 = transform.point((source_x1, source_y0))
    wall_segments, wall_opening_gaps = _scene_wall_render_segments(scene)
    renderer.add_floor(
        min(floor_x0, floor_x1),
        min(floor_z0, floor_z1),
        max(floor_x0, floor_x1),
        max(floor_z0, floor_z1),
    )
    _add_floor_grid(
        renderer,
        x0=min(floor_x0, floor_x1),
        z0=min(floor_z0, floor_z1),
        x1=max(floor_x0, floor_x1),
        z1=max(floor_z0, floor_z1),
    )
    _add_floor_texture_lines(renderer, scene, wall_segments)

    wall_style = {
        "exterior": {
            "thickness": 0.18,
            "height": 2.75,
            "side": (118, 128, 123, 90),
            "cap": (30, 34, 32, 255),
        },
        "interior": {
            "thickness": 0.12,
            "height": 2.55,
            "side": (174, 181, 175, 110),
            "cap": (77, 84, 78, 235),
        },
        "partition": {
            "thickness": 0.07,
            "height": 1.15,
            "side": (145, 154, 149, 50),
            "cap": (87, 94, 89, 155),
        },
    }
    for wall, start_px, end_px in wall_segments:
        start_x, start_z = transform.point(start_px)
        end_x, end_z = transform.point(end_px)
        style = wall_style[wall.wall_type]
        renderer.add_wall(
            start_x,
            start_z,
            end_x,
            end_z,
            thickness=style["thickness"],
            height=style["height"],
            side=style["side"],
            cap=style["cap"],
            layer=11 if wall.wall_type == "exterior" else 12,
        )

    for opening in scene.openings:
        if opening.kind == "window":
            _add_window(renderer, transform, opening)
        else:
            _add_door(renderer, transform, opening)

    for obj in scene.objects:
        if obj.kind == "toilet":
            _add_toilet(renderer, transform, obj)
        elif obj.kind == "bathtub":
            _add_bathtub(renderer, transform, obj)
        elif obj.kind == "shower":
            _add_shower(renderer, transform, obj)
        elif obj.kind == "sink":
            _add_sink(renderer, transform, obj)
        elif obj.kind == "cabinet_run":
            _add_cabinet(renderer, transform, obj)
        elif obj.kind in {"washer_dryer", "water_heater"}:
            _add_appliance(renderer, transform, obj)
        elif obj.kind == "ceiling_light":
            _add_ceiling_light(renderer, transform, obj)
        elif obj.kind in {"duplex_outlet", "smoke_detector", "switch"}:
            _add_wall_device(renderer, transform, obj)
        else:
            _add_fixture_tag(renderer, transform, obj)

    for label in scene.labels:
        x, z = transform.point(label.center_px)
        _add_room_label(renderer, label.name, label.number, x, z)

    colors = {
        "red": (206, 54, 42),
        "blue": (54, 114, 191),
        "yellow": (214, 164, 29),
    }
    for tag in scene.tags:
        x, z = transform.point(tag.center_px)
        _add_tag(renderer, tag.label, x, z, colors[tag.color], side=tag.side)

    for dimension in scene.dimensions:
        dim_center = (
            (dimension.start_px[0] + dimension.end_px[0]) / 2,
            (dimension.start_px[1] + dimension.end_px[1]) / 2,
        )
        if any(
            label.number == dimension.label
            and math.hypot(
                dim_center[0] - label.center_px[0],
                dim_center[1] - label.center_px[1],
            )
            < 92
            for label in scene.labels
        ):
            continue
        start_x, start_z = transform.point(dimension.start_px)
        end_x, end_z = transform.point(dimension.end_px)
        _add_dimension(
            renderer,
            (start_x, 0.02, start_z),
            (end_x, 0.02, end_z),
            dimension.label,
        )

    renderer.render(output_png)
    elapsed = time.perf_counter() - start
    return {
        "preview_png": str(output_png),
        "elapsed_seconds": round(elapsed, 3),
        "surface_count": len(renderer.surfaces),
        "source_scope": scene.source_scope,
        "counts": scene.to_json()["counts"],
        "render_contract": "deterministic_source_px_to_scene_m_low_poly_presentation",
        "procedural_asset_modules": sorted({obj.kind for obj in scene.objects}),
        "wall_render_segments": len(wall_segments),
        "wall_opening_gaps": wall_opening_gaps,
    }


def build_maricopa_source_aligned_artifacts(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = build_maricopa_source_aligned_scene()
    scene_json = output_dir / "maricopa_source_aligned_semantic_scene.json"
    preview_png = output_dir / "maricopa_source_aligned_plan2field3d.png"
    summary_json = output_dir / "maricopa_source_aligned_plan2field3d_summary.json"

    scene_json.write_text(json.dumps(scene.to_json(), indent=2), encoding="utf-8")
    summary = render_semantic_scene(scene, preview_png)
    summary.update(
        {
            "source_pdf": scene.source_pdf,
            "source_page_png": scene.source_page_png,
            "source_crop_png": scene.source_crop_png,
            "scene_json": str(scene_json),
            "qa": {
                "shared_transform_for_grid_walls_objects_labels": True,
                "wall_openings_cut_from_render_geometry": True,
                "generative_3d_used": False,
                "procedural_asset_generation_used": True,
                "objects_are_source_referenced": True,
                "absent_electrical_symbols_not_fabricated": True,
            },
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(
        json.dumps(
            build_maricopa_source_aligned_artifacts(Path("docs/plan2field3d_house_conversion")),
            indent=2,
        )
    )
