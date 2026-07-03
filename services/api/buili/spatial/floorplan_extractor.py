from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import numpy as np

from ..models import Document
from ..storage import object_path

try:  # OpenCV is only needed for raster fallback extraction.
    import cv2
except ImportError:  # pragma: no cover - exercised in slim deployments without OpenCV
    cv2 = None


@dataclass(frozen=True)
class AxisSegment:
    orientation: str
    fixed_px: float
    start_px: float
    end_px: float
    thickness_px: float


def _page_rgb(path: Path, page_no: int, zoom: float) -> np.ndarray:
    with fitz.open(path) as pdf:
        page = pdf[max(0, min(page_no - 1, len(pdf) - 1))]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    return image[:, :, :3].copy()


def _dark_mask(rgb: np.ndarray, *, y_window: tuple[float, float]) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required for raster floor-plan extraction")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    y0 = int(rgb.shape[0] * y_window[0])
    y1 = int(rgb.shape[0] * y_window[1])
    trimmed = np.zeros_like(mask)
    trimmed[y0:y1, :] = mask[y0:y1, :]
    return trimmed


def _component_segments(
    mask: np.ndarray,
    *,
    orientation: str,
    min_length_px: int,
) -> list[AxisSegment]:
    if cv2 is None:
        return []
    kernel = (
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, min_length_px // 2), 3))
        if orientation == "h"
        else cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(9, min_length_px // 2)))
    )
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, close_kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
    segments: list[AxisSegment] = []
    for index in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[index]]
        if area <= 0:
            continue
        if orientation == "h":
            if width < min_length_px or height > max(34, width // 3):
                continue
            segments.append(
                AxisSegment(
                    orientation="h",
                    fixed_px=y + height / 2,
                    start_px=float(x),
                    end_px=float(x + width),
                    thickness_px=float(height),
                )
            )
        else:
            if height < min_length_px or width > max(34, height // 3):
                continue
            segments.append(
                AxisSegment(
                    orientation="v",
                    fixed_px=x + width / 2,
                    start_px=float(y),
                    end_px=float(y + height),
                    thickness_px=float(width),
                )
            )
    return segments


def _cluster_axis_values(values: list[float], tolerance_px: float) -> dict[float, float]:
    if not values:
        return {}
    sorted_values = sorted(values)
    groups: list[list[float]] = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if abs(value - sum(groups[-1]) / len(groups[-1])) <= tolerance_px:
            groups[-1].append(value)
        else:
            groups.append([value])
    mapping: dict[float, float] = {}
    for group in groups:
        snapped = sum(group) / len(group)
        for value in group:
            mapping[value] = snapped
    return mapping


def snap_and_merge_segments(
    segments: list[AxisSegment],
    *,
    snap_tolerance_px: float = 12.0,
    merge_gap_px: float = 18.0,
    min_length_px: float = 40.0,
) -> list[AxisSegment]:
    snapped: list[AxisSegment] = []
    for orientation in ("h", "v"):
        grouped = [segment for segment in segments if segment.orientation == orientation]
        fixed_map = _cluster_axis_values(
            [segment.fixed_px for segment in grouped], snap_tolerance_px
        )
        by_axis: dict[float, list[AxisSegment]] = {}
        for segment in grouped:
            fixed = fixed_map.get(segment.fixed_px, segment.fixed_px)
            by_axis.setdefault(fixed, []).append(segment)
        for fixed, axis_segments in by_axis.items():
            intervals = sorted(
                (
                    min(segment.start_px, segment.end_px),
                    max(segment.start_px, segment.end_px),
                    segment.thickness_px,
                )
                for segment in axis_segments
            )
            if not intervals:
                continue
            start, end, thickness = intervals[0]
            for next_start, next_end, next_thickness in intervals[1:]:
                if next_start <= end + merge_gap_px:
                    end = max(end, next_end)
                    thickness = max(thickness, next_thickness)
                else:
                    if end - start >= min_length_px:
                        snapped.append(
                            AxisSegment(orientation, fixed, start, end, thickness)
                        )
                    start, end, thickness = next_start, next_end, next_thickness
            if end - start >= min_length_px:
                snapped.append(AxisSegment(orientation, fixed, start, end, thickness))
    return snapped


def _segments_from_image(rgb: np.ndarray) -> tuple[list[AxisSegment], dict[str, Any]]:
    mask = _dark_mask(rgb, y_window=(0.16, 0.9))
    min_length = max(42, min(rgb.shape[:2]) // 30)
    raw_segments = [
        *_component_segments(mask, orientation="h", min_length_px=min_length),
        *_component_segments(mask, orientation="v", min_length_px=min_length),
    ]
    merged = snap_and_merge_segments(
        raw_segments,
        snap_tolerance_px=max(8.0, min(rgb.shape[:2]) * 0.006),
        merge_gap_px=max(12.0, min(rgb.shape[:2]) * 0.012),
        min_length_px=min_length,
    )
    metadata = {
        "image_size_px": [int(rgb.shape[1]), int(rgb.shape[0])],
        "raw_axis_segments": len(raw_segments),
        "merged_axis_segments": len(merged),
        "min_length_px": min_length,
        "y_window": [0.16, 0.9],
    }
    return merged, metadata


def _segment_to_wall(
    segment: AxisSegment,
    *,
    center_x: float,
    center_y: float,
    meters_per_px: float,
    index: int,
) -> dict[str, Any]:
    if segment.orientation == "h":
        start = [
            (segment.start_px - center_x) * meters_per_px,
            (segment.fixed_px - center_y) * meters_per_px,
        ]
        end = [
            (segment.end_px - center_x) * meters_per_px,
            (segment.fixed_px - center_y) * meters_per_px,
        ]
    else:
        start = [
            (segment.fixed_px - center_x) * meters_per_px,
            (segment.start_px - center_y) * meters_per_px,
        ]
        end = [
            (segment.fixed_px - center_x) * meters_per_px,
            (segment.end_px - center_y) * meters_per_px,
        ]
    return {
        "id": f"img_wall_{index:03d}",
        "room_id": "extracted_floorplan",
        "from": [round(start[0], 4), round(start[1], 4)],
        "to": [round(end[0], 4), round(end[1], 4)],
        "height_m": 2.7,
        "source": "raster_axis_line_snap_merge",
        "thickness_px": round(segment.thickness_px, 2),
    }


def _floor_polygon_for_segments(
    segments: list[AxisSegment],
    *,
    center_x: float,
    center_y: float,
    meters_per_px: float,
) -> list[list[float]]:
    if not segments:
        return [[0, 0], [4, 0], [4, 3], [0, 3]]
    xs: list[float] = []
    ys: list[float] = []
    for segment in segments:
        if segment.orientation == "h":
            xs.extend([segment.start_px, segment.end_px])
            ys.append(segment.fixed_px)
        else:
            xs.append(segment.fixed_px)
            ys.extend([segment.start_px, segment.end_px])
    pad_px = 18.0
    min_x = (min(xs) - pad_px - center_x) * meters_per_px
    max_x = (max(xs) + pad_px - center_x) * meters_per_px
    min_y = (min(ys) - pad_px - center_y) * meters_per_px
    max_y = (max(ys) + pad_px - center_y) * meters_per_px
    return [
        [round(min_x, 4), round(min_y, 4)],
        [round(max_x, 4), round(min_y, 4)],
        [round(max_x, 4), round(max_y, 4)],
        [round(min_x, 4), round(max_y, 4)],
    ]


def extract_floorplan_payload_from_pdf(
    doc: Document,
    *,
    project_id: str,
    sheet_id: str,
    page_no: int,
    scale: dict[str, Any],
) -> dict[str, Any] | None:
    path = object_path(doc.r2_key)
    if not path.exists() or path.suffix.lower() != ".pdf" or cv2 is None:
        return None
    rgb = _page_rgb(path, page_no=page_no, zoom=2.5)
    segments, extraction_metadata = _segments_from_image(rgb)
    if len(segments) < 6:
        return None
    meters_per_px = 1.0 / float(scale.get("px_per_meter") or 100.0)
    center_x = rgb.shape[1] / 2.0
    center_y = rgb.shape[0] / 2.0
    walls = [
        _segment_to_wall(
            segment,
            center_x=center_x,
            center_y=center_y,
            meters_per_px=meters_per_px,
            index=index,
        )
        for index, segment in enumerate(segments, start=1)
    ]
    floor_polygon = _floor_polygon_for_segments(
        segments,
        center_x=center_x,
        center_y=center_y,
        meters_per_px=meters_per_px,
    )
    extraction = {
        **extraction_metadata,
        "method": "pdf_raster_axis_line_snap_merge",
        "source_doc_id": doc.doc_id,
        "source_filename": doc.filename,
        "source_required_for_strong_evidence": True,
        "meters_per_px": round(meters_per_px, 6),
        "grid_source": "same_model_coordinate_frame",
    }
    return {
        "project_id": project_id,
        "sheet_id": sheet_id,
        "scale": scale,
        "rooms": [
            {
                "id": "extracted_floorplan",
                "name": "Extracted Floor Plan",
                "polygon": floor_polygon,
            }
        ],
        "walls": walls,
        "openings": [],
        "fixtures": [],
        "sources": [
            {
                "citation_chunk_id": "",
                "doc_id": doc.doc_id,
                "sheet_id": sheet_id,
                "bbox": floor_polygon,
                "source_type": "floorplan_axis_segments",
                "source_strength": "display_review",
            }
        ],
        "extraction": extraction,
    }
