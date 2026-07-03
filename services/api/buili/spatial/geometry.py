from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import triangulate, unary_union

from ..config import get_settings


def _spatial_dir(project_id: str) -> Path:
    path = get_settings().storage_root / "spatial" / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_polygon_parts(geometry: Any) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry] if not geometry.is_empty and geometry.area > 1e-8 else []
    if isinstance(geometry, MultiPolygon):
        return [
            polygon for polygon in geometry.geoms if not polygon.is_empty and polygon.area > 1e-8
        ]
    return [
        polygon
        for polygon in getattr(geometry, "geoms", [])
        if isinstance(polygon, Polygon) and not polygon.is_empty and polygon.area > 1e-8
    ]


def _clean_polygon_union(polygons: list[Polygon], *, simplify_m: float = 0.01) -> Any:
    if not polygons:
        return MultiPolygon([])
    merged = unary_union(polygons).buffer(0)
    # Square-join close/open removes tiny raster/vector slivers without rounding wall corners.
    merged = merged.buffer(0.002, join_style=2).buffer(-0.002, join_style=2).buffer(0)
    if simplify_m > 0:
        merged = merged.simplify(simplify_m, preserve_topology=True).buffer(0)
    return merged


def _add_polygon_cap(
    vertices: list[tuple[float, float, float]],
    indices: list[int],
    polygon: Polygon,
    *,
    y: float,
    top: bool,
) -> None:
    for triangle in triangulate(polygon):
        if triangle.area <= 1e-8 or not polygon.covers(triangle.representative_point()):
            continue
        coords = list(triangle.exterior.coords)[:3]
        points = [(float(x), y, float(z)) for x, z in coords]
        if top:
            points = [points[0], points[2], points[1]]
        start = len(vertices)
        vertices.extend(points)
        indices.extend([start, start + 1, start + 2])


def _add_ring_sides(
    vertices: list[tuple[float, float, float]],
    indices: list[int],
    coords: Any,
    *,
    bottom: float,
    top: float,
) -> None:
    points = list(coords)
    for (x0, z0), (x1, z1) in zip(points, points[1:], strict=False):
        if abs(float(x1) - float(x0)) + abs(float(z1) - float(z0)) <= 1e-8:
            continue
        start = len(vertices)
        vertices.extend(
            [
                (float(x0), bottom, float(z0)),
                (float(x1), bottom, float(z1)),
                (float(x1), top, float(z1)),
                (float(x0), top, float(z0)),
            ]
        )
        indices.extend(
            [
                start,
                start + 1,
                start + 2,
                start,
                start + 2,
                start + 3,
            ]
        )


def _add_extruded_polygon(
    vertices: list[tuple[float, float, float]],
    indices: list[int],
    geometry: Any,
    *,
    bottom: float,
    height: float,
) -> int:
    top = bottom + height
    part_count = 0
    for polygon in _as_polygon_parts(geometry):
        _add_polygon_cap(vertices, indices, polygon, y=top, top=True)
        if bottom != top:
            _add_polygon_cap(vertices, indices, polygon, y=bottom, top=False)
            _add_ring_sides(vertices, indices, polygon.exterior.coords, bottom=bottom, top=top)
            for interior in polygon.interiors:
                _add_ring_sides(vertices, indices, interior.coords, bottom=bottom, top=top)
        part_count += 1
    return part_count


def _floor_union(graph: dict[str, Any]) -> Any:
    polygons: list[Polygon] = []
    for room in graph.get("rooms", []):
        coords = room.get("polygon") or []
        if len(coords) < 3:
            continue
        try:
            polygon = Polygon([(float(x), float(y)) for x, y in coords]).buffer(0)
        except (TypeError, ValueError):
            continue
        if polygon.area > 1e-8:
            polygons.append(polygon)
    return _clean_polygon_union(polygons, simplify_m=0.0)


def _wall_polygon(
    start_2d: list[float],
    end_2d: list[float],
    *,
    thickness: float,
) -> Polygon | None:
    sx, sy = float(start_2d[0]), float(start_2d[1])
    ex, ey = float(end_2d[0]), float(end_2d[1])
    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None
    ux = dx / length
    uy = dy / length
    px = -uy * thickness / 2
    py = ux * thickness / 2
    return Polygon(
        [
            (sx + px, sy + py),
            (ex + px, ey + py),
            (ex - px, ey - py),
            (sx - px, sy - py),
        ]
    ).buffer(0)


def _wall_union(graph: dict[str, Any], *, thickness: float) -> tuple[Any, int]:
    polygons: list[Polygon] = []
    heights: list[float] = []
    for wall in graph.get("walls", []):
        polygon = _wall_polygon(
            wall.get("from") or [0, 0],
            wall.get("to") or [0, 0],
            thickness=thickness,
        )
        if polygon is None or polygon.area <= 1e-8:
            continue
        polygons.append(polygon)
        heights.append(float(wall.get("height_m") or 2.7))
    return _clean_polygon_union(polygons, simplify_m=0.006), len(polygons)


def _add_oriented_box(
    vertices: list[tuple[float, float, float]],
    indices: list[int],
    start_2d: list[float],
    end_2d: list[float],
    *,
    thickness: float,
    height: float,
    bottom: float = 0.0,
) -> None:
    sx, sy = float(start_2d[0]), float(start_2d[1])
    ex, ey = float(end_2d[0]), float(end_2d[1])
    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return
    ux = dx / length
    uy = dy / length
    px = -uy * thickness / 2
    py = ux * thickness / 2
    top = bottom + height
    corners = [
        (sx + px, bottom, sy + py),
        (ex + px, bottom, ey + py),
        (ex - px, bottom, ey - py),
        (sx - px, bottom, sy - py),
        (sx + px, top, sy + py),
        (ex + px, top, ey + py),
        (ex - px, top, ey - py),
        (sx - px, top, sy - py),
    ]
    base = len(vertices)
    vertices.extend(corners)
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0),
    ]
    for face in faces:
        indices.extend([base + face[0], base + face[1], base + face[2]])


def _room_centers(graph: dict[str, Any]) -> dict[str, tuple[float, float]]:
    centers: dict[str, tuple[float, float]] = {}
    for room in graph.get("rooms", []):
        polygon = room.get("polygon") or []
        if not polygon:
            continue
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        centers[str(room.get("id", ""))] = (sum(xs) / len(xs), sum(ys) / len(ys))
    return centers


def _add_fixture_proxy(
    vertices: list[tuple[float, float, float]],
    indices: list[int],
    center: tuple[float, float],
    fixture_type: str,
) -> None:
    x, z = center
    if "ceiling" in fixture_type or "diffuser" in fixture_type:
        bottom = 2.65
        height = 0.06
        size = 0.28
    elif "panel" in fixture_type or "equipment" in fixture_type:
        bottom = 0.6
        height = 1.0
        size = 0.35
    else:
        bottom = 0.35
        height = 0.18
        size = 0.16
    _add_oriented_box(
        vertices,
        indices,
        [x - size / 2, z],
        [x + size / 2, z],
        thickness=size,
        height=height,
        bottom=bottom,
    )


def _grid_metadata(vertices: list[tuple[float, float, float]]) -> dict[str, Any]:
    if not vertices:
        return {
            "origin_m": [0.0, 0.0],
            "spacing_m": 1.0,
            "bounds_m": [[0.0, 0.0], [0.0, 0.0]],
        }
    array = np.array(vertices, dtype=np.float32)
    min_x = float(array[:, 0].min())
    max_x = float(array[:, 0].max())
    min_z = float(array[:, 2].min())
    max_z = float(array[:, 2].max())
    spacing = 1.0
    origin_x = math.floor(min_x / spacing) * spacing
    origin_z = math.floor(min_z / spacing) * spacing
    return {
        "origin_m": [round(origin_x, 4), round(origin_z, 4)],
        "spacing_m": spacing,
        "bounds_m": [
            [round(min_x, 4), round(min_z, 4)],
            [round(max_x, 4), round(max_z, 4)],
        ],
    }


def _pad4(data: bytes, pad_byte: bytes = b"\x00") -> bytes:
    padding = (4 - (len(data) % 4)) % 4
    return data + pad_byte * padding


def _write_glb(path: Path, vertices: list[tuple[float, float, float]], indices: list[int]) -> None:
    if not vertices:
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
        indices = [0, 1, 2]
    position_array = np.array(vertices, dtype=np.float32)
    index_array = np.array(indices, dtype=np.uint32)
    position_bytes = position_array.tobytes()
    position_padded = _pad4(position_bytes)
    index_offset = len(position_padded)
    index_bytes = index_array.tobytes()
    bin_blob = _pad4(position_padded + index_bytes)
    mins = position_array.min(axis=0).round(5).tolist()
    maxs = position_array.max(axis=0).round(5).tolist()
    gltf = {
        "asset": {"version": "2.0", "generator": "Buili Plan2Field-3D deterministic assembler"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "PlanGraph design reference"}],
        "meshes": [
            {
                "name": "PlanGraph parametric proxy mesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "mode": 4,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "Buili neutral material",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.72, 0.72, 0.68, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.85,
                },
            }
        ],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(position_bytes),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": len(index_bytes),
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
                "min": mins,
                "max": maxs,
            },
            {
                "bufferView": 1,
                "componentType": 5125,
                "count": len(indices),
                "type": "SCALAR",
            },
        ],
    }
    json_blob = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total_length = 12 + 8 + len(json_blob) + 8 + len(bin_blob)
    with path.open("wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total_length))
        fh.write(struct.pack("<I4s", len(json_blob), b"JSON"))
        fh.write(json_blob)
        fh.write(struct.pack("<I4s", len(bin_blob), b"BIN\x00"))
        fh.write(bin_blob)


def build_design_glb(
    graph: dict[str, Any], project_id: str, asset_id: str
) -> tuple[str, dict[str, Any]]:
    vertices: list[tuple[float, float, float]] = []
    indices: list[int] = []
    floor_geometry = _floor_union(graph)
    floor_parts = _add_extruded_polygon(
        vertices, indices, floor_geometry, bottom=-0.04, height=0.04
    )
    wall_geometry, source_wall_count = _wall_union(graph, thickness=0.12)
    wall_height = max(
        [float(wall.get("height_m") or 2.7) for wall in graph.get("walls", [])] or [2.7]
    )
    wall_parts = _add_extruded_polygon(
        vertices, indices, wall_geometry, bottom=0.0, height=wall_height
    )
    centers = _room_centers(graph)
    fixture_counts: dict[str, int] = {}
    for fixture in graph.get("fixtures", []):
        room_id = str(fixture.get("room_id", ""))
        count = fixture_counts.get(room_id, 0)
        fixture_counts[room_id] = count + 1
        center_m = fixture.get("center_m")
        if isinstance(center_m, list | tuple) and len(center_m) >= 2:
            center = (float(center_m[0]), float(center_m[1]))
            offset = 0.0
        else:
            center = centers.get(room_id, (0.5, 0.5))
            offset = ((count % 4) - 1.5) * 0.35
        _add_fixture_proxy(
            vertices,
            indices,
            (center[0] + offset, center[1] + math.floor(count / 4) * 0.25),
            str(fixture.get("type") or "fixture"),
        )

    out_dir = _spatial_dir(project_id)
    filename = f"{asset_id}_design.glb"
    path = out_dir / filename
    _write_glb(path, vertices, indices)
    uri = f"spatial/{project_id}/{filename}"
    metadata = {
        "format": "glb",
        "assembly": "deterministic_plangraph_union_geometry",
        "rooms": len(graph.get("rooms", [])),
        "walls": len(graph.get("walls", [])),
        "openings": len(graph.get("openings", [])),
        "fixtures": len(graph.get("fixtures", [])),
        "vertex_count": len(vertices),
        "triangle_count": len(indices) // 3,
        "floor_polygon_parts": floor_parts,
        "wall_source_segments": source_wall_count,
        "wall_polygon_parts": wall_parts,
        "wall_union_enabled": True,
        "grid": _grid_metadata(vertices),
        "source_required_for_strong_evidence": True,
    }
    return uri, metadata
