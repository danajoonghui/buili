from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image, ImageDraw

from .floorplan_extractor import AxisSegment, _page_rgb, _segments_from_image
from .micro_vlm import detect_micro_vlm_plan_elements
from .semantic_scene import (
    SemanticDimension,
    SemanticObject,
    SemanticOpening,
    SemanticRoomLabel,
    SemanticScene,
    SemanticWall,
    SourceTransform,
    render_semantic_scene,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None


ROOM_ALIASES = {
    "BATH": "BATHROOM",
    "BATHROOM": "BATHROOM",
    "BEDROOM": "BEDROOM",
    "BREAKFAST": "BREAKFAST",
    "CLOSET": "CLOSET",
    "DINING": "DINING ROOM",
    "DINING ROOM": "DINING ROOM",
    "ENTRY": "ENTRY",
    "ENTRE": "ENTRY",
    "FOYER": "FOYER",
    "GARAGE": "GARAGE",
    "KITCHEN": "KITCHEN",
    "LAUNDRY": "LAUNDRY",
    "LIVING": "LIVING ROOM",
    "LIVING ROOM": "LIVING ROOM",
    "MASTER BATH": "MASTER BATH",
    "MASTER BDRM": "MASTER BDRM",
    "MASTER BEDROOM": "MASTER BDRM",
    "NOOK": "NOOK",
    "OFFICE": "OFFICE",
    "PATIO": "PATIO",
}

DIMENSION_RE = re.compile(
    r"\d{1,2}\s*['′]?\s*[- ]?\s*\d{0,2}\s*[\"”]?\s*[xX*]\s*"
    r"\d{1,3}\s*['′]?\s*[- ]?\s*\d{0,2}\s*[\"”]?"
)

_EASYOCR_READER: Any | None = None


@dataclass(frozen=True)
class TextItem:
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float
    source: str

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2, (y0 + y1) / 2


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("|", "I")).strip()


def _upper_key(text: str) -> str:
    cleaned = _clean_text(text).upper()
    cleaned = cleaned.replace("BDRM", "BDRM")
    cleaned = cleaned.replace("BATHI", "BATH")
    cleaned = cleaned.replace("ROOMM", "ROOM")
    return re.sub(r"[^A-Z0-9'\" xX*.-]+", " ", cleaned).strip()


def _axis_segment_bounds(
    segments: list[AxisSegment], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    if not segments:
        return 0, 0, image_width, image_height
    xs: list[float] = []
    ys: list[float] = []
    for segment in segments:
        if segment.orientation == "h":
            xs.extend([segment.start_px, segment.end_px])
            ys.append(segment.fixed_px)
        else:
            xs.append(segment.fixed_px)
            ys.extend([segment.start_px, segment.end_px])
    pad = max(24, int(min(image_width, image_height) * 0.015))
    return (
        max(0, int(min(xs) - pad)),
        max(0, int(min(ys) - pad)),
        min(image_width, int(max(xs) + pad)),
        min(image_height, int(max(ys) + pad)),
    )


def _segment_length(segment: AxisSegment) -> float:
    return abs(segment.end_px - segment.start_px)


def _filter_floorplan_segments(
    segments: list[AxisSegment], image_width: int, image_height: int
) -> list[AxisSegment]:
    filtered: list[AxisSegment] = []
    for segment in segments:
        length = _segment_length(segment)
        if segment.orientation == "h":
            if length > image_width * 0.88:
                continue
            if length > image_width * 0.72 and segment.fixed_px > image_height * 0.74:
                continue
            if segment.fixed_px < image_height * 0.12 or segment.fixed_px > image_height * 0.9:
                continue
        else:
            if length > image_height * 0.88:
                continue
            if length > image_height * 0.72 and segment.fixed_px < image_width * 0.12:
                continue
            if segment.fixed_px < image_width * 0.02 or segment.fixed_px > image_width * 0.98:
                continue
        filtered.append(segment)
    return filtered if len(filtered) >= 6 else segments


def _crop_segments(
    segments: list[AxisSegment], bbox: tuple[int, int, int, int]
) -> list[AxisSegment]:
    x0, y0, x1, y1 = bbox
    cropped: list[AxisSegment] = []
    for segment in segments:
        if segment.orientation == "h":
            if not y0 <= segment.fixed_px <= y1:
                continue
            start = max(segment.start_px, x0) - x0
            end = min(segment.end_px, x1) - x0
            fixed = segment.fixed_px - y0
        else:
            if not x0 <= segment.fixed_px <= x1:
                continue
            start = max(segment.start_px, y0) - y0
            end = min(segment.end_px, y1) - y0
            fixed = segment.fixed_px - x0
        if end - start >= 16:
            cropped.append(
                AxisSegment(segment.orientation, fixed, start, end, segment.thickness_px)
            )
    return cropped


def _snap_segments_to_dark_evidence(
    crop_png: Path, segments: list[AxisSegment], *, max_offset_px: int = 6
) -> list[AxisSegment]:
    _dark, distance = _scene_dark_distance(crop_png)
    if distance is None:
        return segments
    snapped: list[AxisSegment] = []
    for segment in segments:
        best_segment = segment
        best_score: tuple[float, float] | None = None
        for offset in range(-max_offset_px, max_offset_px + 1):
            fixed = segment.fixed_px + offset
            if segment.orientation == "h":
                points = _sample_line_points(
                    (segment.start_px, fixed), (segment.end_px, fixed), step_px=8.0
                )
            else:
                points = _sample_line_points(
                    (fixed, segment.start_px), (fixed, segment.end_px), step_px=8.0
                )
            values = _distance_samples(distance, points)
            metrics = _window_values(values)
            score = (metrics["p95_px"], metrics["mean_px"])
            if best_score is None or score < best_score:
                best_score = score
                best_segment = AxisSegment(
                    segment.orientation,
                    fixed,
                    segment.start_px,
                    segment.end_px,
                    segment.thickness_px,
                )
        snapped.append(best_segment)
    return snapped


def _embedded_pdf_text_items(
    pdf_path: Path,
    *,
    page_no: int,
    zoom: float,
    crop_bbox: tuple[int, int, int, int],
) -> list[TextItem]:
    x0, y0, x1, y1 = crop_bbox
    with fitz.open(pdf_path) as pdf:
        page = pdf[max(0, min(page_no - 1, len(pdf) - 1))]
        grouped: dict[tuple[int, int], list[tuple[float, float, float, float, str]]] = {}
        for word in page.get_text("words"):
            wx0, wy0, wx1, wy1, text, block, line, *_ = word
            px0, py0, px1, py1 = wx0 * zoom, wy0 * zoom, wx1 * zoom, wy1 * zoom
            if px1 < x0 or px0 > x1 or py1 < y0 or py0 > y1:
                continue
            grouped.setdefault((int(block), int(line)), []).append(
                (px0 - x0, py0 - y0, px1 - x0, py1 - y0, str(text))
            )
    items: list[TextItem] = []
    for words in grouped.values():
        ordered = sorted(words, key=lambda item: item[0])
        text = _clean_text(" ".join(word[-1] for word in ordered))
        if not text:
            continue
        items.append(
            TextItem(
                text=text,
                bbox=(
                    min(word[0] for word in ordered),
                    min(word[1] for word in ordered),
                    max(word[2] for word in ordered),
                    max(word[3] for word in ordered),
                ),
                confidence=1.0,
                source="pdf_text",
            )
        )
    return items


def _easyocr_reader() -> Any | None:
    global _EASYOCR_READER
    if easyocr is None:
        return None
    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _EASYOCR_READER


def _easyocr_text_items_from_input(
    image_input: Any, *, source: str
) -> tuple[list[TextItem], dict[str, Any]]:
    start = time.perf_counter()
    reader = _easyocr_reader()
    init_seconds = time.perf_counter() - start
    if reader is None:
        return [], {"engine": "none", "available": False, "reason": "easyocr_not_installed"}

    start = time.perf_counter()
    results = reader.readtext(
        image_input,
        detail=1,
        paragraph=False,
        decoder="greedy",
        batch_size=8,
    )
    items: list[TextItem] = []
    for box, text, confidence in results:
        if float(confidence) < 0.12:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        items.append(
            TextItem(
                text=_clean_text(str(text)),
                bbox=(min(xs), min(ys), max(xs), max(ys)),
                confidence=float(confidence),
                source=source,
            )
        )
    return items, {
        "engine": "easyocr",
        "available": True,
        "reader_init_seconds": round(init_seconds, 4),
        "ocr_seconds": round(time.perf_counter() - start, 4),
        "raw_results": len(results),
        "kept_results": len(items),
    }


def _ocr_text_items(crop_png: Path) -> tuple[list[TextItem], dict[str, Any]]:
    return _easyocr_text_items_from_input(str(crop_png), source="easyocr")


def _scaled_ocr_text_items(
    crop_png: Path, *, scale: float = 1.4
) -> tuple[list[TextItem], dict[str, Any]]:
    image = Image.open(crop_png).convert("RGB")
    scaled = image.resize(
        (int(image.width * scale), int(image.height * scale)),
        Image.Resampling.BICUBIC,
    )
    items, metadata = _easyocr_text_items_from_input(
        np.asarray(scaled),
        source=f"easyocr_scaled_{scale:g}x",
    )
    mapped = [
        TextItem(
            text=item.text,
            bbox=tuple(value / scale for value in item.bbox),  # type: ignore[arg-type]
            confidence=item.confidence,
            source=item.source,
        )
        for item in items
    ]
    metadata["scale"] = scale
    return mapped, metadata


def _dedupe_text_items(text_items: list[TextItem]) -> list[TextItem]:
    ordered = sorted(text_items, key=lambda item: (-item.confidence, len(item.text)))
    kept: list[TextItem] = []
    for item in ordered:
        key = _upper_key(item.text)
        if not key:
            continue
        cx, cy = item.center
        if any(
            _upper_key(existing.text) == key
            and math.hypot(cx - existing.center[0], cy - existing.center[1]) < 32
            for existing in kept
        ):
            continue
        kept.append(item)
    return sorted(kept, key=lambda item: (item.bbox[1], item.bbox[0]))


def _is_room_label(text: str) -> str | None:
    key = _upper_key(text)
    if DIMENSION_RE.search(key):
        return None
    for alias, canonical in sorted(ROOM_ALIASES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"\b{re.escape(alias)}\b", key):
            return canonical
    return None


def _dimension_text(text: str) -> str:
    key = _upper_key(text)
    match = DIMENSION_RE.search(key)
    if not match:
        return ""
    value = match.group(0)
    value = value.replace("*", "x").replace("X", "x").replace("′", "'").replace("”", '"')
    sides = re.split(r"\s*x\s*", value, maxsplit=1)
    if len(sides) != 2:
        return ""
    left = _normalize_dimension_side(sides[0])
    right = _normalize_dimension_side(sides[1])
    if not left or not right:
        return ""
    return f"{left} x {right}"


def _normalize_dimension_side(text: str) -> str:
    cleaned = text.strip().replace("′", "'").replace("”", '"').replace(" ", "").replace("--", "-")
    if not cleaned:
        return ""
    explicit = re.match(r"^(?P<feet>\d{1,2})'?-?(?P<inch>\d{0,2})\"?$", cleaned)
    if explicit and ("'" in cleaned or "-" in cleaned):
        feet = int(explicit.group("feet"))
        inch_text = explicit.group("inch") or "0"
        inches = int(inch_text)
        if 0 < feet <= 80 and 0 <= inches <= 11:
            return f"{feet}'-{inches}\""
        return ""
    digits = "".join(re.findall(r"\d", cleaned))
    if len(digits) == 2:
        feet = int(digits)
        if 0 < feet <= 35:
            return f"{feet}'-0\""
    if len(digits) == 3:
        feet = int(digits[:2])
        inches = int(digits[2])
        if 0 < feet <= 80 and 0 <= inches <= 11:
            return f"{feet}'-{inches}\""
    if len(digits) == 4:
        feet = int(digits[:2])
        inches = int(digits[2:])
        if 0 < feet <= 80 and 0 <= inches <= 11:
            return f"{feet}'-{inches}\""
    return ""


def _nearby_dimension(label: TextItem, text_items: list[TextItem]) -> str:
    lx, ly = label.center
    best: tuple[float, str] | None = None
    for item in text_items:
        dim = _dimension_text(item.text)
        if not dim:
            continue
        ix, iy = item.center
        if abs(ix - lx) > 90 or not -8 <= iy - ly <= 48:
            continue
        score = abs(ix - lx) + abs(iy - ly) * 1.5
        if best is None or score < best[0]:
            best = (score, dim)
    return best[1] if best else ""


def _room_labels_from_text(text_items: list[TextItem]) -> list[SemanticRoomLabel]:
    labels: list[SemanticRoomLabel] = []
    for item in text_items:
        room_name = _is_room_label(item.text)
        if not room_name:
            continue
        if room_name == "BREAKFAST":
            cx, cy = item.center
            if any(
                _upper_key(other.text) == "NOOK"
                and abs(other.center[0] - cx) < 60
                and 0 <= other.center[1] - cy <= 40
                for other in text_items
            ):
                room_name = "BREAKFAST NOOK"
        detail = _dimension_text(item.text) or _nearby_dimension(item, text_items)
        labels.append(
            SemanticRoomLabel(
                room_name,
                detail,
                item.center,
                item.text,
            )
        )

    deduped: list[SemanticRoomLabel] = []
    for label in labels:
        if label.name == "NOOK" and any(
            existing.name == "BREAKFAST NOOK"
            and math.hypot(
                existing.center_px[0] - label.center_px[0],
                existing.center_px[1] - label.center_px[1],
            )
            < 72
            for existing in labels
        ):
            continue
        if any(
            existing.name == label.name
            and math.hypot(
                existing.center_px[0] - label.center_px[0],
                existing.center_px[1] - label.center_px[1],
            )
            < 34
            for existing in deduped
        ):
            continue
        deduped.append(label)
    return deduped


def _walls_from_segments(
    segments: list[AxisSegment],
    *,
    crop_width: int,
    crop_height: int,
) -> list[SemanticWall]:
    if not segments:
        return []
    xs: list[float] = []
    ys: list[float] = []
    for segment in segments:
        if segment.orientation == "h":
            xs.extend([segment.start_px, segment.end_px])
            ys.append(segment.fixed_px)
        else:
            xs.append(segment.fixed_px)
            ys.extend([segment.start_px, segment.end_px])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    edge_tol = max(24.0, min(crop_width, crop_height) * 0.035)
    walls: list[SemanticWall] = []
    for index, segment in enumerate(segments, start=1):
        if segment.orientation == "h":
            start = (segment.start_px, segment.fixed_px)
            end = (segment.end_px, segment.fixed_px)
            near_edge = (
                abs(segment.fixed_px - min_y) <= edge_tol
                or abs(segment.fixed_px - max_y) <= edge_tol
            )
        else:
            start = (segment.fixed_px, segment.start_px)
            end = (segment.fixed_px, segment.end_px)
            near_edge = (
                abs(segment.fixed_px - min_x) <= edge_tol
                or abs(segment.fixed_px - max_x) <= edge_tol
            )
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        wall_type = "exterior" if near_edge and length > edge_tol * 1.8 else "interior"
        walls.append(
            SemanticWall(
                id=f"auto_wall_{index:03d}",
                start_px=start,
                end_px=end,
                wall_type=wall_type,
                source_note="auto_cv_axis_segment",
            )
        )
    return walls


def _openings_from_wall_gaps(
    segments: list[AxisSegment],
    *,
    crop_width: int,
    crop_height: int,
) -> list[SemanticOpening]:
    openings: list[SemanticOpening] = []
    groups: dict[tuple[str, int], list[AxisSegment]] = {}
    for segment in segments:
        key = (segment.orientation, int(round(segment.fixed_px / 8.0) * 8))
        groups.setdefault(key, []).append(segment)

    index = 1
    for (orientation, fixed), group in groups.items():
        intervals = sorted(
            (min(segment.start_px, segment.end_px), max(segment.start_px, segment.end_px))
            for segment in group
        )
        if len(intervals) < 2:
            continue
        for (_, prev_end), (next_start, _) in zip(intervals, intervals[1:], strict=False):
            gap = next_start - prev_end
            if gap < 30 or gap > 150:
                continue
            if orientation == "h":
                center = ((prev_end + next_start) / 2, float(fixed))
                angle = 0.0
            else:
                center = (float(fixed), (prev_end + next_start) / 2)
                angle = 90.0
            openings.append(
                SemanticOpening(
                    id=f"auto_door_{index:03d}",
                    kind="door",
                    center_px=center,
                    length_px=gap,
                    angle_deg=angle,
                    mark="",
                    source_note="auto_collinear_wall_gap",
                )
            )
            index += 1
    return _dedupe_openings(openings)


def _nearest_wall_projection(
    point: tuple[float, float], walls: list[SemanticWall]
) -> tuple[SemanticWall | None, tuple[float, float], float]:
    px, py = point
    best_wall: SemanticWall | None = None
    best_point = point
    best_distance = float("inf")
    for wall in walls:
        sx, sy = wall.start_px
        ex, ey = wall.end_px
        dx = ex - sx
        dy = ey - sy
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-6:
            continue
        ratio = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
        wx = sx + dx * ratio
        wy = sy + dy * ratio
        distance = math.hypot(px - wx, py - wy)
        if distance < best_distance:
            best_wall = wall
            best_point = (wx, wy)
            best_distance = distance
    return best_wall, best_point, best_distance


def _windows_from_colored_markers(
    crop_png: Path, walls: list[SemanticWall]
) -> list[SemanticOpening]:
    if cv2 is None:
        return []
    image = cv2.imread(str(crop_png), cv2.IMREAD_COLOR)
    if image is None:
        return []
    # This catches red/pink window size markers common in public review PDFs without OCR cost.
    bgr = image.astype(np.float32)
    blue, green, red = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    mask = (
        (red > 120) & (red > green * 1.3) & (red > blue * 1.3) & (green < 140) & (blue < 140)
    ).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask * 255, 8)
    components: list[tuple[float, float, float, float, float, float]] = []
    for index in range(1, count):
        x, y, width, height, area = [float(value) for value in stats[index]]
        if area < 6 or width < 4 or height < 2 or width > 380 or height > 18:
            continue
        region = mask[int(y) : int(y + height), int(x) : int(x + width)]
        col_has = np.count_nonzero(region, axis=0) > 0
        runs: list[tuple[int, int]] = []
        start: int | None = None
        last_seen: int | None = None
        for offset, has_pixel in enumerate(col_has):
            if has_pixel:
                if start is None:
                    start = offset
                last_seen = offset
            elif start is not None and last_seen is not None and offset - last_seen > 18:
                runs.append((start, last_seen + 1))
                start = None
                last_seen = None
        if start is not None and last_seen is not None:
            runs.append((start, last_seen + 1))
        for run_start, run_end in runs or [(0, int(width))]:
            run_region = region[:, run_start:run_end]
            run_area = int(np.count_nonzero(run_region))
            run_width = run_end - run_start
            if run_area < 6 or run_width < 4:
                continue
            ys, xs = np.nonzero(run_region)
            components.append(
                (
                    x + run_start,
                    y + float(ys.min()),
                    x + run_end,
                    y + float(ys.max()) + 1,
                    x + run_start + float(xs.mean()),
                    y + float(ys.mean()),
                )
            )
    if not components:
        return []

    rows: list[list[tuple[float, float, float, float, float, float]]] = []
    for component in sorted(components, key=lambda item: item[5]):
        for row in rows:
            row_y = sum(item[5] for item in row) / len(row)
            if abs(component[5] - row_y) <= 10:
                row.append(component)
                break
        else:
            rows.append([component])

    grouped: list[list[tuple[float, float, float, float, float, float]]] = []
    for row in rows:
        for component in sorted(row, key=lambda item: item[0]):
            if not grouped:
                grouped.append([component])
                continue
            current = grouped[-1]
            current_y = sum(item[5] for item in current) / len(current)
            current_x1 = max(item[2] for item in current)
            if abs(component[5] - current_y) <= 10 and component[0] - current_x1 <= 36:
                current.append(component)
            else:
                grouped.append([component])

    windows: list[SemanticOpening] = []
    for index, group in enumerate(grouped, start=1):
        x0 = min(item[0] for item in group)
        y0 = min(item[1] for item in group)
        x1 = max(item[2] for item in group)
        y1 = max(item[3] for item in group)
        center = ((x0 + x1) / 2, (y0 + y1) / 2)
        wall, projected, distance = _nearest_wall_projection(center, walls)
        if wall is None or distance > 48:
            continue
        angle = (
            0.0
            if abs(wall.end_px[0] - wall.start_px[0]) >= abs(wall.end_px[1] - wall.start_px[1])
            else 90.0
        )
        marker_span = x1 - x0 if angle == 0.0 else y1 - y0
        length = max(76.0, min(190.0, marker_span * 1.55))
        windows.append(
            SemanticOpening(
                id=f"auto_window_marker_{index:03d}",
                kind="window",
                center_px=projected,
                length_px=length,
                angle_deg=angle,
                mark="",
                source_note=f"auto_colored_window_marker:distance_px={distance:.1f}",
            )
        )
    return _dedupe_openings(windows)


def _dedupe_openings(openings: list[SemanticOpening]) -> list[SemanticOpening]:
    deduped: list[SemanticOpening] = []
    for opening in openings:
        if any(
            math.hypot(
                opening.center_px[0] - existing.center_px[0],
                opening.center_px[1] - existing.center_px[1],
            )
            < max(22.0, min(opening.length_px, existing.length_px) * 0.45)
            for existing in deduped
        ):
            continue
        deduped.append(opening)
    return deduped


def _objects_from_text(text_items: list[TextItem]) -> list[SemanticObject]:
    objects: list[SemanticObject] = []
    index = 1
    for item in text_items:
        key = _upper_key(item.text)
        kind = ""
        label = ""
        if "WASHER" in key or key == "W":
            kind, label = "washer_dryer", "washer"
        elif "DRYER" in key or key == "D":
            kind, label = "washer_dryer", "dryer"
        elif re.search(r"\bWH\b|WATER HEATER", key):
            kind, label = "water_heater", "wh"
        elif "SINK" in key:
            kind, label = "sink", "sink"
        elif any(token in key for token in ("RANGE", "DW", "D W", "FREEZER", "REFRIG")):
            kind, label = "cabinet_run", key.lower()
        elif "SMOKE" in key or re.search(r"\bSD\b", key):
            kind, label = "smoke_detector", "sd"
        elif "CEILING" in key and "LIGHT" in key:
            kind, label = "ceiling_light", "light"
        elif "OUTLET" in key or "RECEPTACLE" in key:
            kind, label = "duplex_outlet", "rec"
        elif re.fullmatch(r"S|SW", key):
            kind, label = "switch", "sw"
        if not kind:
            continue
        width = max(34.0, item.bbox[2] - item.bbox[0] + 14)
        depth = max(34.0, item.bbox[3] - item.bbox[1] + 22)
        objects.append(
            SemanticObject(
                id=f"auto_object_{index:03d}",
                kind=kind,  # type: ignore[arg-type]
                center_px=item.center,
                width_px=width,
                depth_px=depth,
                angle_deg=0.0,
                label=label,
                source_note=f"auto_ocr:{item.text}",
            )
        )
        index += 1
    return objects


def _dedupe_objects(
    objects: list[SemanticObject], labels: list[SemanticRoomLabel]
) -> list[SemanticObject]:
    if not objects:
        return []
    bath_centers = [label.center_px for label in labels if "BATH" in label.name]
    laundry_centers = [label.center_px for label in labels if label.name in {"LAUNDRY", "UTILITY"}]

    def nearest_distance(obj: SemanticObject, centers: list[tuple[float, float]]) -> float:
        if not centers:
            return float("inf")
        ox, oy = obj.center_px
        return min(math.hypot(ox - cx, oy - cy) for cx, cy in centers)

    def rank(obj: SemanticObject) -> tuple[int, float, float]:
        if obj.source_note.startswith("micro_vlm_verified") or obj.source_note.startswith(
            "micro_vlm_corrected"
        ):
            source_rank = 0
        elif obj.source_note.startswith("auto_ocr"):
            source_rank = 1
        else:
            source_rank = 2
        if obj.kind in {"toilet", "sink", "bathtub", "shower"}:
            context = nearest_distance(obj, bath_centers)
        elif obj.kind in {"washer_dryer", "water_heater"}:
            context = nearest_distance(obj, laundry_centers)
        else:
            context = 0.0
        area = obj.width_px * obj.depth_px
        return source_rank, context, -area

    ordered = sorted(objects, key=rank)
    kept: list[SemanticObject] = []
    limits = {
        "water_heater": max(1, len(laundry_centers)),
        "washer_dryer": max(2, len(laundry_centers) * 2),
        "toilet": max(1, len(bath_centers) * 2),
    }
    counts: dict[str, int] = {}
    for obj in ordered:
        minimum_distance = 56.0
        if obj.kind in {"water_heater", "toilet", "washer_dryer"}:
            minimum_distance = 72.0
        if any(
            existing.kind == obj.kind
            and math.hypot(
                existing.center_px[0] - obj.center_px[0],
                existing.center_px[1] - obj.center_px[1],
            )
            < minimum_distance
            for existing in kept
        ):
            continue
        if counts.get(obj.kind, 0) >= limits.get(obj.kind, 80):
            continue
        kept.append(obj)
        counts[obj.kind] = counts.get(obj.kind, 0) + 1
    return sorted(kept, key=lambda item: item.id)


def _bbox_overlap_ratio(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    ix0 = max(lx0, rx0)
    iy0 = max(ly0, ry0)
    ix1 = min(lx1, rx1)
    iy1 = min(ly1, ry1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    left_area = max((lx1 - lx0) * (ly1 - ly0), 1.0)
    return intersection / left_area


def _nearest_room_label(
    center: tuple[float, float], labels: list[SemanticRoomLabel]
) -> tuple[SemanticRoomLabel | None, float]:
    if not labels:
        return None, float("inf")
    cx, cy = center
    nearest = min(
        labels,
        key=lambda label: math.hypot(cx - label.center_px[0], cy - label.center_px[1]),
    )
    return nearest, math.hypot(cx - nearest.center_px[0], cy - nearest.center_px[1])


def _component_objects_from_image(
    crop_png: Path,
    labels: list[SemanticRoomLabel],
    text_items: list[TextItem],
    existing_objects: list[SemanticObject],
) -> list[SemanticObject]:
    if cv2 is None:
        return []
    image = cv2.imread(str(crop_png), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    _, dark = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY_INV)
    horizontal = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
    )
    vertical = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 40))
    )
    wall_like = cv2.bitwise_or(horizontal, vertical)
    detail = cv2.bitwise_and(dark, cv2.bitwise_not(wall_like))
    detail = cv2.morphologyEx(
        detail, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(detail, 8)
    text_boxes = [item.bbox for item in text_items]
    objects: list[SemanticObject] = []
    class_limits = {
        "bathtub": 2,
        "cabinet_run": 5,
        "fixture_tag": 3,
        "shower": 3,
        "sink": 4,
        "toilet": 4,
        "washer_dryer": 2,
        "water_heater": 1,
    }
    class_counts: dict[str, int] = {}
    for index in range(1, count):
        x, y, width, height, area = [float(value) for value in stats[index]]
        if area < 65 or area > 1800:
            continue
        if width < 14 or height < 14 or width > 140 or height > 140:
            continue
        bbox = (x, y, x + width, y + height)
        density = area / max(width * height, 1.0)
        if density > 0.42:
            continue
        if any(_bbox_overlap_ratio(bbox, text_box) > 0.12 for text_box in text_boxes):
            continue
        center = (float(centroids[index][0]), float(centroids[index][1]))
        nearest, distance = _nearest_room_label(center, labels)
        if nearest is None or distance > 175:
            continue
        ratio = max(width, height) / max(min(width, height), 1.0)
        room_name = nearest.name
        kind = ""
        label = ""
        if "BATH" in room_name or room_name == "CLOSET":
            if max(width, height) >= 86 and ratio >= 1.45:
                kind, label = "bathtub", "tub"
            elif max(width, height) >= 62:
                kind, label = "shower", "shower"
            elif class_counts.get("toilet", 0) <= class_counts.get("sink", 0):
                kind, label = "toilet", "wc"
            else:
                kind, label = "sink", "sink"
        elif room_name == "KITCHEN":
            if max(width, height) >= 58:
                kind, label = "cabinet_run", "cabinet"
        elif room_name == "LAUNDRY":
            if max(width, height) <= 58 and ratio <= 1.45:
                kind, label = "washer_dryer", "laundry"
            elif max(width, height) >= 58:
                kind, label = "cabinet_run", "utility"
        elif room_name in {"DINING ROOM", "FOYER", "ENTRY"} and 1.2 <= ratio <= 2.6:
            kind, label = "fixture_tag", room_name.split()[0].lower()
        if not kind or class_counts.get(kind, 0) >= class_limits.get(kind, 2):
            continue
        objects.append(
            SemanticObject(
                id=f"auto_component_{index:03d}",
                kind=kind,  # type: ignore[arg-type]
                center_px=center,
                width_px=max(width, 28.0),
                depth_px=max(height, 28.0),
                angle_deg=90.0 if height > width * 1.35 else 0.0,
                label=label,
                source_note=(
                    "auto_cv_component_after_text_and_wall_removal:"
                    f"room={room_name},density={density:.2f}"
                ),
            )
        )
        class_counts[kind] = class_counts.get(kind, 0) + 1
    return objects


def _near_existing_object(
    center: tuple[float, float], objects: list[SemanticObject], *, min_distance_px: float
) -> bool:
    cx, cy = center
    return any(
        math.hypot(cx - obj.center_px[0], cy - obj.center_px[1])
        < max(min_distance_px, min(obj.width_px, obj.depth_px) * 0.85)
        for obj in objects
    )


def _near_opening(
    center: tuple[float, float], openings: list[SemanticOpening], *, min_distance_px: float
) -> bool:
    cx, cy = center
    return any(
        math.hypot(cx - opening.center_px[0], cy - opening.center_px[1])
        < max(min_distance_px, opening.length_px * 0.32)
        for opening in openings
    )


def _detail_objects_from_image(
    crop_png: Path,
    labels: list[SemanticRoomLabel],
    text_items: list[TextItem],
    existing_objects: list[SemanticObject],
    openings: list[SemanticOpening],
) -> list[SemanticObject]:
    if cv2 is None:
        return []
    image = cv2.imread(str(crop_png), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    _, dark = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY_INV)
    horizontal = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
    )
    vertical = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 40))
    )
    wall_like = cv2.bitwise_or(horizontal, vertical)
    detail = cv2.bitwise_and(dark, cv2.bitwise_not(wall_like))
    detail = cv2.morphologyEx(
        detail, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(detail, 8)
    text_boxes = [
        (item.bbox[0] - 4, item.bbox[1] - 4, item.bbox[2] + 4, item.bbox[3] + 4)
        for item in text_items
    ]
    objects: list[SemanticObject] = []
    room_counts: dict[str, int] = {}
    for index in range(1, count):
        x, y, width, height, area = [float(value) for value in stats[index]]
        if area < 55 or area > 950:
            continue
        if width < 8 or height < 8 or width > 112 or height > 112:
            continue
        bbox = (x, y, x + width, y + height)
        density = area / max(width * height, 1.0)
        ratio = max(width, height) / max(min(width, height), 1.0)
        if density > 0.48 or ratio > 6.2:
            continue
        if height <= 12 and ratio > 3.6:
            continue
        if density > 0.36 and min(width, height) <= 14:
            continue
        if any(_bbox_overlap_ratio(bbox, text_box) > 0.08 for text_box in text_boxes):
            continue
        center = (float(centroids[index][0]), float(centroids[index][1]))
        nearest, distance = _nearest_room_label(center, labels)
        if nearest is None or distance > 220:
            continue
        if (
            abs(center[0] - nearest.center_px[0]) < 105
            and abs(center[1] - nearest.center_px[1]) < 34
        ):
            continue
        if _near_existing_object(center, existing_objects, min_distance_px=44):
            continue
        if _near_opening(center, openings, min_distance_px=72):
            continue
        room_name = nearest.name
        detail_rooms = {
            "BATHROOM",
            "CLOSET",
            "DINING ROOM",
            "ENTRY",
            "FOYER",
            "KITCHEN",
            "LAUNDRY",
            "MASTER BATH",
        }
        if room_name not in detail_rooms:
            continue
        if room_name in {"PATIO", "GARAGE"} and area < 130:
            continue
        if room_counts.get(room_name, 0) >= 4:
            continue
        objects.append(
            SemanticObject(
                id=f"auto_detail_{index:03d}",
                kind="fixture_tag",
                center_px=center,
                width_px=max(width, 24.0),
                depth_px=max(height, 24.0),
                angle_deg=90.0 if height > width * 1.35 else 0.0,
                label="",
                source_note=(
                    "auto_cv_detail_component:"
                    f"room={room_name},area={area:.0f},density={density:.2f}"
                ),
            )
        )
        room_counts[room_name] = room_counts.get(room_name, 0) + 1
        if len(objects) >= 18:
            break
    return objects


def _circle_objects_from_image(
    crop_png: Path,
    labels: list[SemanticRoomLabel],
    text_objects: list[SemanticObject],
) -> list[SemanticObject]:
    if cv2 is None:
        return []
    image = cv2.imread(str(crop_png), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    blurred = cv2.medianBlur(image, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=32,
        param1=80,
        param2=18,
        minRadius=8,
        maxRadius=28,
    )
    if circles is None:
        return []
    bath_centers = [label.center_px for label in labels if "BATH" in label.name]
    laundry_centers = [label.center_px for label in labels if label.name == "LAUNDRY"]
    existing = [obj.center_px for obj in text_objects]
    objects: list[SemanticObject] = []
    for idx, circle in enumerate(np.round(circles[0, :]).astype(int), start=1):
        x, y, radius = [float(value) for value in circle]
        if any(math.hypot(x - ex, y - ey) < 28 for ex, ey in existing):
            continue
        nearest_bath = (
            min(math.hypot(x - bx, y - by) for bx, by in bath_centers)
            if bath_centers
            else float("inf")
        )
        nearest_laundry = (
            min(math.hypot(x - lx, y - ly) for lx, ly in laundry_centers)
            if laundry_centers
            else float("inf")
        )
        if nearest_bath <= 125:
            kind = "toilet"
            label = "wc"
        elif nearest_laundry <= 95 and radius >= 13:
            kind = "water_heater"
            label = "wh"
        else:
            continue
        objects.append(
            SemanticObject(
                id=f"auto_circle_{idx:03d}",
                kind=kind,  # type: ignore[arg-type]
                center_px=(x, y),
                width_px=max(28.0, radius * 2.2),
                depth_px=max(32.0, radius * 2.6),
                angle_deg=0.0,
                label=label,
                source_note="auto_hough_circle_fixture_candidate",
            )
        )
    return objects


def _dimensions_from_text(
    text_items: list[TextItem], labels: list[SemanticRoomLabel]
) -> list[SemanticDimension]:
    dimensions: list[SemanticDimension] = []
    for item in text_items:
        dim = _dimension_text(item.text)
        if not dim:
            continue
        x0, y0, x1, y1 = item.bbox
        pad = max(10.0, min(42.0, (x1 - x0) * 0.16))
        dimensions.append(SemanticDimension((x0 - pad, y1 + 8), (x1 + pad, y1 + 8), dim))

    deduped: list[SemanticDimension] = []
    for dimension in sorted(
        dimensions,
        key=lambda item: (
            item.label,
            round((item.start_px[0] + item.end_px[0]) / 48),
            round((item.start_px[1] + item.end_px[1]) / 36),
        ),
    ):
        cx = (dimension.start_px[0] + dimension.end_px[0]) / 2
        cy = (dimension.start_px[1] + dimension.end_px[1]) / 2
        if any(
            existing.label == dimension.label
            and math.hypot(
                cx - (existing.start_px[0] + existing.end_px[0]) / 2,
                cy - (existing.start_px[1] + existing.end_px[1]) / 2,
            )
            < 36
            for existing in deduped
        ):
            continue
        deduped.append(dimension)
    return sorted(deduped, key=lambda item: (item.start_px[1], item.start_px[0]))[:18]


def _scene_dark_distance(crop_png: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    if cv2 is None:
        return None, None
    image = cv2.imread(str(crop_png), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None
    _, dark = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY_INV)
    distance = cv2.distanceTransform(cv2.bitwise_not(dark), cv2.DIST_L2, 3)
    return dark, distance


def _sample_line_points(
    start: tuple[float, float], end: tuple[float, float], *, step_px: float = 8.0
) -> list[tuple[int, int]]:
    sx, sy = start
    ex, ey = end
    length = max(math.hypot(ex - sx, ey - sy), 1.0)
    count = max(2, int(length / step_px) + 1)
    points: list[tuple[int, int]] = []
    for index in range(count):
        ratio = index / max(count - 1, 1)
        points.append((int(round(sx + (ex - sx) * ratio)), int(round(sy + (ey - sy) * ratio))))
    return points


def _distance_samples(distance: np.ndarray, points: list[tuple[int, int]]) -> list[float]:
    height, width = distance.shape[:2]
    values: list[float] = []
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            values.append(float(distance[y, x]))
    return values


def _window_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_px": 0.0, "p95_px": 0.0, "max_px": 0.0}
    array = np.array(values, dtype=np.float32)
    return {
        "mean_px": round(float(array.mean()), 3),
        "p95_px": round(float(np.percentile(array, 95)), 3),
        "max_px": round(float(array.max()), 3),
    }


def _object_dark_fraction(dark: np.ndarray, obj: SemanticObject) -> float:
    height, width = dark.shape[:2]
    x0 = max(0, int(round(obj.center_px[0] - obj.width_px / 2)))
    y0 = max(0, int(round(obj.center_px[1] - obj.depth_px / 2)))
    x1 = min(width, int(round(obj.center_px[0] + obj.width_px / 2)))
    y1 = min(height, int(round(obj.center_px[1] + obj.depth_px / 2)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    region = dark[y0:y1, x0:x1]
    return round(float(np.count_nonzero(region) / max(region.size, 1)), 4)


def evaluate_scene_alignment(
    scene: SemanticScene, crop_png: Path, overlay_png: Path | None = None
) -> dict[str, Any]:
    dark, distance = _scene_dark_distance(crop_png)
    if dark is None or distance is None:
        return {"available": False, "reason": "opencv_unavailable_or_crop_missing"}

    wall_values: list[float] = []
    wall_failures: list[dict[str, Any]] = []
    for wall in scene.walls:
        values = _distance_samples(distance, _sample_line_points(wall.start_px, wall.end_px))
        wall_values.extend(values)
        metrics = _window_values(values)
        if metrics["p95_px"] > 2.5:
            wall_failures.append({"id": wall.id, **metrics})

    object_rows: list[dict[str, Any]] = []
    weak_objects: list[dict[str, Any]] = []
    for obj in scene.objects:
        dark_fraction = _object_dark_fraction(dark, obj)
        row = {
            "id": obj.id,
            "kind": obj.kind,
            "center_px": [round(obj.center_px[0], 2), round(obj.center_px[1], 2)],
            "dark_fraction": dark_fraction,
            "source_note": obj.source_note,
        }
        object_rows.append(row)
        if dark_fraction < 0.025:
            weak_objects.append(row)

    opening_rows: list[dict[str, Any]] = []
    height, width = dark.shape[:2]
    for opening in scene.openings:
        cx, cy = int(round(opening.center_px[0])), int(round(opening.center_px[1]))
        radius = max(6, min(18, int(round(opening.length_px / 6))))
        x0, y0 = max(0, cx - radius), max(0, cy - radius)
        x1, y1 = min(width, cx + radius), min(height, cy + radius)
        region = dark[y0:y1, x0:x1]
        dark_fraction = float(np.count_nonzero(region) / max(region.size, 1)) if region.size else 0
        opening_rows.append(
            {
                "id": opening.id,
                "kind": opening.kind,
                "center_px": [round(opening.center_px[0], 2), round(opening.center_px[1], 2)],
                "center_dark_fraction": round(dark_fraction, 4),
            }
        )

    if overlay_png is not None:
        _write_alignment_overlay(scene, crop_png, overlay_png, weak_objects, wall_failures)

    wall_metrics = _window_values(wall_values)
    return {
        "available": True,
        "coordinate_transform_error_px": 0.0,
        "wall_distance_to_source_dark_px": wall_metrics,
        "wall_failure_count": len(wall_failures),
        "wall_failures": wall_failures[:12],
        "object_evidence": object_rows,
        "weak_object_count": len(weak_objects),
        "weak_objects": weak_objects[:12],
        "opening_clearance": opening_rows,
        "quality_gate_passed": (
            wall_metrics["p95_px"] <= 2.5
            and len(wall_failures) <= max(1, int(len(scene.walls) * 0.12))
            and len(weak_objects) == 0
        ),
        "overlay_png": str(overlay_png) if overlay_png is not None else "",
    }


def _write_alignment_overlay(
    scene: SemanticScene,
    crop_png: Path,
    overlay_png: Path,
    weak_objects: list[dict[str, Any]],
    wall_failures: list[dict[str, Any]],
) -> None:
    image = Image.open(crop_png).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    weak_ids = {item["id"] for item in weak_objects}
    weak_wall_ids = {item["id"] for item in wall_failures}
    for wall in scene.walls:
        color = (224, 72, 56, 210) if wall.id in weak_wall_ids else (47, 126, 78, 190)
        draw.line([wall.start_px, wall.end_px], fill=color, width=3)
    for opening in scene.openings:
        cx, cy = opening.center_px
        radius = max(5, min(14, opening.length_px / 8))
        color = (54, 114, 191, 210) if opening.kind == "door" else (28, 145, 160, 210)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=color, width=3)
    for obj in scene.objects:
        x0 = obj.center_px[0] - obj.width_px / 2
        y0 = obj.center_px[1] - obj.depth_px / 2
        x1 = obj.center_px[0] + obj.width_px / 2
        y1 = obj.center_px[1] + obj.depth_px / 2
        color = (224, 72, 56, 230) if obj.id in weak_ids else (214, 164, 29, 225)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0, max(0, y0 - 14)), obj.kind, fill=color)
    Image.alpha_composite(image, overlay).convert("RGB").save(overlay_png)


def build_semantic_scene_from_pdf(
    pdf_path: Path,
    *,
    output_dir: Path,
    page_no: int = 1,
    zoom: float = 2.5,
    use_ocr: bool = True,
) -> tuple[SemanticScene, dict[str, Any]]:
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    full_page_png = output_dir / "auto_page_1.png"
    crop_png = output_dir / "auto_plan_crop.png"

    stage_start = time.perf_counter()
    rgb = _page_rgb(pdf_path, page_no=page_no, zoom=zoom)
    Image.fromarray(rgb).save(full_page_png)
    page_render_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    segments, wall_metadata = _segments_from_image(rgb)
    floorplan_segments = _filter_floorplan_segments(segments, rgb.shape[1], rgb.shape[0])
    crop_bbox = _axis_segment_bounds(floorplan_segments, rgb.shape[1], rgb.shape[0])
    cropped_rgb = rgb[crop_bbox[1] : crop_bbox[3], crop_bbox[0] : crop_bbox[2]]
    Image.fromarray(cropped_rgb).save(crop_png)
    cropped_segments = _crop_segments(floorplan_segments, crop_bbox)
    cropped_segments = _snap_segments_to_dark_evidence(crop_png, cropped_segments)
    wall_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    text_items = _embedded_pdf_text_items(pdf_path, page_no=page_no, zoom=zoom, crop_bbox=crop_bbox)
    embedded_count = len(text_items)
    ocr_metadata: dict[str, Any] = {"engine": "not_used"}
    scaled_ocr_metadata: dict[str, Any] = {"engine": "not_used"}
    if use_ocr and len(text_items) < 8:
        ocr_items, ocr_metadata = _ocr_text_items(crop_png)
        text_items.extend(ocr_items)
        scaled_items, scaled_ocr_metadata = _scaled_ocr_text_items(crop_png, scale=1.4)
        text_items.extend(scaled_items)
        text_items = _dedupe_text_items(text_items)
    text_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    crop_height, crop_width = cropped_rgb.shape[:2]
    labels = _room_labels_from_text(text_items)
    text_objects = _objects_from_text(text_items)
    circle_objects = _circle_objects_from_image(crop_png, labels, text_objects)
    component_objects = _component_objects_from_image(
        crop_png, labels, text_items, [*text_objects, *circle_objects]
    )
    walls = _walls_from_segments(cropped_segments, crop_width=crop_width, crop_height=crop_height)
    gap_openings = _openings_from_wall_gaps(
        cropped_segments, crop_width=crop_width, crop_height=crop_height
    )
    marker_windows = _windows_from_colored_markers(crop_png, walls)
    openings = _dedupe_openings([*gap_openings, *marker_windows])
    micro_vlm_objects, micro_vlm_openings, micro_vlm_metadata = detect_micro_vlm_plan_elements(
        crop_png,
        labels=labels,
        existing_objects=[*text_objects, *circle_objects, *component_objects],
        existing_openings=openings,
    )
    openings = _dedupe_openings([*micro_vlm_openings, *openings])
    detail_objects = _detail_objects_from_image(
        crop_png,
        labels,
        text_items,
        [*text_objects, *circle_objects, *component_objects, *micro_vlm_objects],
        openings,
    )
    objects = _dedupe_objects(
        [
            *micro_vlm_objects,
            *text_objects,
            *circle_objects,
            *component_objects,
            *detail_objects,
        ],
        labels,
    )
    scene = SemanticScene(
        source_pdf=str(pdf_path),
        source_page_png=str(full_page_png),
        source_crop_png=str(crop_png),
        transform=SourceTransform(
            width_px=float(crop_width),
            height_px=float(crop_height),
            meters_per_px=0.016,
        ),
        walls=walls,
        openings=openings,
        objects=objects,
        labels=labels,
        tags=[],
        dimensions=_dimensions_from_text(text_items, labels),
        source_scope="automatic_pdf_semantic_scene_extraction",
    )
    compile_seconds = time.perf_counter() - stage_start

    metadata = {
        "method": "automatic_pdf_to_semantic_scene_v1",
        "total_scene_build_seconds": round(time.perf_counter() - start, 4),
        "stages": {
            "pdf_page_render_seconds": round(page_render_seconds, 4),
            "wall_segments_and_crop_seconds": round(wall_seconds, 4),
            "text_ocr_seconds": round(text_seconds, 4),
            "semantic_compile_seconds": round(compile_seconds, 4),
        },
        "page_image_size_px": [int(rgb.shape[1]), int(rgb.shape[0])],
        "crop_bbox_px": list(crop_bbox),
        "crop_size_px": [int(crop_width), int(crop_height)],
        "wall_metadata": wall_metadata,
        "filtered_wall_segments": len(floorplan_segments),
        "embedded_text_items": embedded_count,
        "text_items": len(text_items),
        "ocr": ocr_metadata,
        "scaled_ocr": scaled_ocr_metadata,
        "micro_vlm": micro_vlm_metadata,
        "counts": scene.to_json()["counts"],
        "quality_gates": {
            "manual_semantic_seed_used": False,
            "micro_vlm_primary_parser_enabled": bool(micro_vlm_metadata.get("enabled")),
            "object_candidates_before_dedupe": (
                len(text_objects)
                + len(circle_objects)
                + len(component_objects)
                + len(micro_vlm_objects)
                + len(detail_objects)
            ),
            "ocr_text_object_candidates": len(text_objects),
            "circle_object_candidates": len(circle_objects),
            "component_object_candidates": len(component_objects),
            "micro_vlm_object_candidates": len(micro_vlm_objects),
            "micro_vlm_opening_candidates": len(micro_vlm_openings),
            "detail_object_candidates": len(detail_objects),
            "object_candidates_after_dedupe": len(objects),
            "gap_door_candidates": len(gap_openings),
            "colored_window_candidates": len(marker_windows),
            "room_label_count": len(labels),
            "wall_count": len(scene.walls),
            "automatic_scene_ready": len(scene.walls) >= 6 and len(labels) >= 1,
        },
    }
    return scene, metadata


def _scene_bounds_px(scene: SemanticScene) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for wall in scene.walls:
        xs.extend([wall.start_px[0], wall.end_px[0]])
        ys.extend([wall.start_px[1], wall.end_px[1]])
    if not xs or not ys:
        return 0.0, 0.0, scene.transform.width_px, scene.transform.height_px
    pad = max(24.0, min(scene.transform.width_px, scene.transform.height_px) * 0.02)
    return (
        max(0.0, min(xs) - pad),
        max(0.0, min(ys) - pad),
        min(scene.transform.width_px, max(xs) + pad),
        min(scene.transform.height_px, max(ys) + pad),
    )


def _point_m(scene: SemanticScene, point_px: tuple[float, float]) -> list[float]:
    x, z = scene.transform.point(point_px)
    return [round(x, 4), round(z, 4)]


def _nearest_wall_id(scene: SemanticScene, point_px: tuple[float, float]) -> str:
    best_id = scene.walls[0].id if scene.walls else ""
    best_distance = float("inf")
    px, py = point_px
    for wall in scene.walls:
        sx, sy = wall.start_px
        ex, ey = wall.end_px
        dx = ex - sx
        dy = ey - sy
        denom = dx * dx + dy * dy
        if denom <= 1e-6:
            continue
        t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))
        wx = sx + dx * t
        wy = sy + dy * t
        distance = math.hypot(px - wx, py - wy)
        if distance < best_distance:
            best_distance = distance
            best_id = wall.id
    return best_id


def semantic_scene_to_plan_graph_payload(
    scene: SemanticScene,
    *,
    project_id: str,
    sheet_id: str,
    scale: dict[str, Any],
    source_doc_id: str = "",
    source_filename: str = "",
) -> dict[str, Any]:
    x0, y0, x1, y1 = _scene_bounds_px(scene)
    floor_polygon = [
        _point_m(scene, (x0, y1)),
        _point_m(scene, (x1, y1)),
        _point_m(scene, (x1, y0)),
        _point_m(scene, (x0, y0)),
    ]
    room_id = "auto_floorplan"
    walls = [
        {
            "id": wall.id,
            "room_id": room_id,
            "from": _point_m(scene, wall.start_px),
            "to": _point_m(scene, wall.end_px),
            "height_m": 2.75 if wall.wall_type == "exterior" else 2.55,
            "source": wall.source_note or "automatic_semantic_wall",
            "wall_type": wall.wall_type,
        }
        for wall in scene.walls
    ]
    openings = [
        {
            "type": opening.kind,
            "wall_id": _nearest_wall_id(scene, opening.center_px),
            "center_m": _point_m(scene, opening.center_px),
            "width_m": round(scene.transform.distance(opening.length_px), 3),
            "source": opening.source_note or "automatic_semantic_opening",
        }
        for opening in scene.openings
    ]
    fixtures = [
        {
            "type": obj.kind,
            "room_id": room_id,
            "wall_id": _nearest_wall_id(scene, obj.center_px),
            "required_count": 1,
            "observed_count": 0,
            "bbox": [
                round(obj.center_px[0] - obj.width_px / 2, 3),
                round(obj.center_px[1] - obj.depth_px / 2, 3),
                round(obj.center_px[0] + obj.width_px / 2, 3),
                round(obj.center_px[1] + obj.depth_px / 2, 3),
            ],
            "center_m": _point_m(scene, obj.center_px),
            "source_entity_id": obj.id,
            "source": obj.source_note or "automatic_semantic_object",
        }
        for obj in scene.objects
    ]
    sources = [
        {
            "citation_chunk_id": "",
            "doc_id": source_doc_id,
            "sheet_id": sheet_id,
            "bbox": floor_polygon,
            "source_type": "automatic_semantic_scene",
            "source_strength": "display_review",
        }
    ]
    for label in scene.labels:
        sources.append(
            {
                "citation_chunk_id": "",
                "doc_id": source_doc_id,
                "sheet_id": sheet_id,
                "bbox": [*_point_m(scene, label.center_px), 0.0, 0.0],
                "source_type": "room_label",
                "source_strength": "display_review",
                "text": f"{label.name} {label.number}".strip(),
            }
        )
    return {
        "project_id": project_id,
        "sheet_id": sheet_id,
        "scale": scale,
        "rooms": [
            {
                "id": room_id,
                "name": "Automatically extracted floor plan",
                "polygon": floor_polygon,
            }
        ],
        "walls": walls,
        "openings": openings,
        "fixtures": fixtures,
        "sources": sources,
        "extraction": {
            "method": "automatic_pdf_semantic_scene_v1",
            "source_doc_id": source_doc_id,
            "source_filename": source_filename,
            "source_pdf": scene.source_pdf,
            "source_page_png": scene.source_page_png,
            "source_crop_png": scene.source_crop_png,
            "room_labels": [label.__dict__ for label in scene.labels],
            "dimensions": [dimension.__dict__ for dimension in scene.dimensions],
            "counts": scene.to_json()["counts"],
            "source_required_for_strong_evidence": True,
            "grid_source": "shared_source_pixel_transform",
        },
    }


def build_auto_plan2field3d_artifacts(
    pdf_path: Path,
    output_dir: Path,
    *,
    page_no: int = 1,
    use_ocr: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene, scene_metadata = build_semantic_scene_from_pdf(
        pdf_path,
        output_dir=output_dir,
        page_no=page_no,
        use_ocr=use_ocr,
    )
    scene_json = output_dir / "auto_semantic_scene.json"
    preview_png = output_dir / "auto_plan2field3d.png"
    alignment_overlay_png = output_dir / "auto_alignment_overlay.png"
    summary_json = output_dir / "auto_plan2field3d_summary.json"
    scene_json.write_text(json.dumps(scene.to_json(), indent=2), encoding="utf-8")

    alignment_qa = evaluate_scene_alignment(
        scene, Path(scene.source_crop_png), alignment_overlay_png
    )
    render_start = time.perf_counter()
    render_summary = render_semantic_scene(scene, preview_png)
    render_seconds = time.perf_counter() - render_start

    summary = {
        "input_pdf": str(pdf_path),
        "scene_json": str(scene_json),
        "preview_png": str(preview_png),
        "alignment_overlay_png": str(alignment_overlay_png),
        "scene_build": scene_metadata,
        "alignment_qa": alignment_qa,
        "render": render_summary,
        "total_seconds": round(scene_metadata["total_scene_build_seconds"] + render_seconds, 4),
        "qa": {
            "manual_semantic_seed_used": False,
            "ocr_used": scene_metadata["ocr"].get("engine") == "easyocr",
            "wall_openings_cut_from_render_geometry": True,
            "procedural_asset_generation_used": True,
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
