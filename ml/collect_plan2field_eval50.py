from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import requests
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CUBICASA_RECORD_API = "https://zenodo.org/api/records/2613548"
CUBICASA_ZIP_URL = f"{CUBICASA_RECORD_API}/files/cubicasa5k.zip/content"


Matrix = tuple[float, float, float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class CubicasaSource:
    dataset: str = "CubiCasa5K"
    zenodo_record: str = CUBICASA_RECORD_API
    zip_url: str = CUBICASA_ZIP_URL
    paper: str = "https://arxiv.org/abs/1904.01920"
    license_note: str = "Use according to CubiCasa5K / Zenodo dataset terms."


def _identity() -> Matrix:
    return 1.0, 0.0, 0.0, 1.0, 0.0, 0.0


def _compose(parent: Matrix, child: Matrix) -> Matrix:
    pa, pb, pc, pd, pe, pf = parent
    ca, cb, cc, cd, ce, cf = child
    return (
        pa * ca + pc * cb,
        pb * ca + pd * cb,
        pa * cc + pc * cd,
        pb * cc + pd * cd,
        pa * ce + pc * cf + pe,
        pb * ce + pd * cf + pf,
    )


def _apply(matrix: Matrix, point: Point) -> Point:
    a, b, c, d, e, f = matrix
    x, y = point
    return a * x + c * y + e, b * x + d * y + f


def _parse_transform(transform: str | None) -> Matrix:
    if not transform:
        return _identity()
    current = _identity()
    for name, raw_values in re.findall(r"([a-zA-Z]+)\(([^)]*)\)", transform):
        values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", raw_values)]
        local = _identity()
        if name == "matrix" and len(values) >= 6:
            local = tuple(values[:6])  # type: ignore[assignment]
        elif name == "translate":
            tx = values[0] if values else 0.0
            ty = values[1] if len(values) > 1 else 0.0
            local = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == "scale":
            sx = values[0] if values else 1.0
            sy = values[1] if len(values) > 1 else sx
            local = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "rotate" and values:
            angle = math.radians(values[0])
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            rotate = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
            if len(values) >= 3:
                cx, cy = values[1], values[2]
                local = _compose(
                    _compose((1.0, 0.0, 0.0, 1.0, cx, cy), rotate),
                    (1.0, 0.0, 0.0, 1.0, -cx, -cy),
                )
            else:
                local = rotate
        current = _compose(current, local)
    return current


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _class_tokens(element: ET.Element) -> list[str]:
    return (element.get("class") or "").replace("-", " ").split()


def _class_text(element: ET.Element) -> str:
    return element.get("class") or ""


def _parse_points(value: str | None) -> list[Point]:
    if not value:
        return []
    numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value)]
    return [(numbers[index], numbers[index + 1]) for index in range(0, len(numbers) - 1, 2)]


def _direct_polygon(element: ET.Element, matrix: Matrix) -> list[Point]:
    for child in list(element):
        if _tag_name(child) == "polygon":
            return [_apply(matrix, point) for point in _parse_points(child.get("points"))]
    return []


def _first_descendant_polygon(element: ET.Element, matrix: Matrix) -> list[Point]:
    stack: list[tuple[ET.Element, Matrix]] = [(element, matrix)]
    while stack:
        current, current_matrix = stack.pop(0)
        if _tag_name(current) == "polygon":
            return [_apply(current_matrix, point) for point in _parse_points(current.get("points"))]
        for child in list(current):
            stack.append((child, _compose(current_matrix, _parse_transform(child.get("transform")))))
    return []


def _boundary_polygon(element: ET.Element, matrix: Matrix) -> list[Point]:
    stack: list[tuple[ET.Element, Matrix]] = [(element, matrix)]
    fallback: list[Point] = []
    while stack:
        current, current_matrix = stack.pop(0)
        if _tag_name(current) == "polygon":
            points = [_apply(current_matrix, point) for point in _parse_points(current.get("points"))]
            if not fallback:
                fallback = points
            continue
        if "BoundaryPolygon" in _class_text(current):
            polygon = _first_descendant_polygon(current, current_matrix)
            if polygon:
                return polygon
        for child in list(current):
            stack.append((child, _compose(current_matrix, _parse_transform(child.get("transform")))))
    return fallback


def _bbox(points: list[Point]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _principal_segment(points: list[Point]) -> tuple[float, float, float, float, float, float]:
    if len(points) < 2:
        x0, y0, x1, y1 = _bbox(points)
        return x0, y0, x1, y1, 0.0, math.hypot(x1 - x0, y1 - y0)
    arr = np_array(points)
    mean_x = float(arr[:, 0].mean())
    mean_y = float(arr[:, 1].mean())
    centered = arr - np.array([[mean_x, mean_y]], dtype=np.float32)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    projections = centered @ axis
    start = np.array([mean_x, mean_y], dtype=np.float32) + axis * float(projections.min())
    end = np.array([mean_x, mean_y], dtype=np.float32) + axis * float(projections.max())
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    length = float(projections.max() - projections.min())
    return float(start[0]), float(start[1]), float(end[0]), float(end[1]), angle, length


def np_array(points: list[Point]) -> Any:
    import numpy as np

    return np.array(points, dtype=np.float32)


def _normalize_object_kind(class_text: str) -> str:
    text = class_text.lower()
    if "toilet" in text:
        return "toilet"
    if "shower" in text:
        return "shower"
    if "bathtub" in text or "bath" in text:
        return "bathtub"
    if "doublesink" in text or "sink" in text:
        return "sink"
    if "saunastove" in text or "waterheater" in text:
        return "water_heater"
    if "electricalappliance" in text or "appliance" in text:
        return "cabinet_run"
    if "closet" in text or "cabinet" in text or "housing" in text:
        return "cabinet_run"
    if "column" in text:
        return "column"
    return "fixture"


def parse_cubicasa_svg(svg_path: Path) -> dict[str, Any]:
    root = ET.parse(svg_path).getroot()
    width = float(root.get("width", "0") or 0)
    height = float(root.get("height", "0") or 0)
    gt: dict[str, Any] = {
        "source_svg": str(svg_path),
        "image_size_px": [width, height],
        "walls": [],
        "openings": [],
        "objects": [],
    }

    counters = {"wall": 0, "opening": 0, "object": 0}

    def walk(element: ET.Element, parent_matrix: Matrix) -> None:
        local = _compose(parent_matrix, _parse_transform(element.get("transform")))
        class_text = _class_text(element)
        tokens = set(_class_tokens(element))
        tag_id = element.get("id") or ""
        is_wall = "Wall" in tokens and tag_id == "Wall"
        is_opening = tag_id in {"Door", "Window"} or (
            "Door" in tokens or "Window" in tokens
        )
        is_object = (
            ("FixedFurniture" in tokens and "FixedFurnitureSet" not in tokens)
            or tag_id == "Column"
            or "Column" in tokens
        )

        if is_wall:
            points = _direct_polygon(element, local)
            if len(points) >= 2:
                counters["wall"] += 1
                x0, y0, x1, y1, angle, length = _principal_segment(points)
                gt["walls"].append(
                    {
                        "id": f"gt_wall_{counters['wall']:04d}",
                        "kind": "wall",
                        "wall_type": "exterior" if "External" in tokens else "interior",
                        "segment": [x0, y0, x1, y1],
                        "bbox": list(_bbox(points)),
                        "angle_deg": angle,
                        "length_px": length,
                        "source_class": class_text,
                    }
                )

        if is_opening:
            points = _direct_polygon(element, local)
            if len(points) >= 2:
                counters["opening"] += 1
                x0, y0, x1, y1, angle, length = _principal_segment(points)
                kind = "door" if tag_id == "Door" or "Door" in tokens else "window"
                gt["openings"].append(
                    {
                        "id": f"gt_opening_{counters['opening']:04d}",
                        "kind": kind,
                        "bbox": list(_bbox(points)),
                        "segment": [x0, y0, x1, y1],
                        "angle_deg": angle,
                        "length_px": length,
                        "source_class": class_text,
                    }
                )

        if is_object:
            points = _boundary_polygon(element, local)
            if len(points) >= 3:
                x0, y0, x1, y1 = _bbox(points)
                area = max(0.0, (x1 - x0) * (y1 - y0))
                if area >= 16.0:
                    counters["object"] += 1
                    _sx, _sy, _ex, _ey, angle, length = _principal_segment(points)
                    gt["objects"].append(
                        {
                            "id": f"gt_object_{counters['object']:04d}",
                            "kind": _normalize_object_kind(class_text),
                            "bbox": [x0, y0, x1, y1],
                            "angle_deg": angle,
                            "length_px": length,
                            "source_class": class_text,
                        }
                    )

        for child in list(element):
            walk(child, local)

    walk(root, _identity())
    gt["counts"] = {
        "walls": len(gt["walls"]),
        "openings": len(gt["openings"]),
        "objects": len(gt["objects"]),
    }
    return gt


def _scale_bbox(bbox: list[float], sx: float, sy: float) -> list[float]:
    return [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]


def _scale_segment(segment: list[float], sx: float, sy: float) -> list[float]:
    return [segment[0] * sx, segment[1] * sy, segment[2] * sx, segment[3] * sy]


def align_ground_truth_to_image(gt: dict[str, Any], image_path: Path) -> dict[str, Any]:
    image = Image.open(image_path)
    image_width, image_height = image.size
    svg_width, svg_height = [float(value) for value in gt.get("image_size_px", [image_width, image_height])]
    if svg_width <= 0 or svg_height <= 0:
        return gt
    sx = image_width / svg_width
    sy = image_height / svg_height
    if abs(sx - 1.0) < 1e-4 and abs(sy - 1.0) < 1e-4:
        gt["coordinate_frame"] = "cubicasa_svg_px_already_matching_png"
        return gt
    for row in gt.get("walls", []):
        row["bbox"] = _scale_bbox(row["bbox"], sx, sy)
        row["segment"] = _scale_segment(row["segment"], sx, sy)
        row["length_px"] = math.hypot(
            row["segment"][2] - row["segment"][0],
            row["segment"][3] - row["segment"][1],
        )
    for collection in ("openings", "objects"):
        for row in gt.get(collection, []):
            row["bbox"] = _scale_bbox(row["bbox"], sx, sy)
            if "segment" in row:
                row["segment"] = _scale_segment(row["segment"], sx, sy)
                row["length_px"] = math.hypot(
                    row["segment"][2] - row["segment"][0],
                    row["segment"][3] - row["segment"][1],
                )
            elif "length_px" in row:
                row["length_px"] = float(row["length_px"]) * max(sx, sy)
    gt["image_size_px"] = [float(image_width), float(image_height)]
    gt["coordinate_frame"] = "F1_scaled_png_px"
    gt["coordinate_scale_from_svg"] = {"sx": sx, "sy": sy}
    return gt


def _remote_zip() -> Any:
    try:
        from remotezip import RemoteZip
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("remotezip is required: python -m pip install remotezip") from exc
    return RemoteZip(CUBICASA_ZIP_URL)


def _select_samples(rz: Any, split: str, count: int) -> list[str]:
    split_text = rz.read(f"cubicasa5k/{split}.txt").decode("utf-8")
    prefixes = [line.strip().strip("/") for line in split_text.splitlines() if line.strip()]
    selected: list[str] = []
    names = set(rz.namelist())
    for prefix in prefixes:
        base = f"cubicasa5k/{prefix}"
        if not base.endswith("/"):
            base += "/"
        if f"{base}F1_scaled.png" in names and f"{base}model.svg" in names:
            selected.append(base)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(f"Only found {len(selected)} usable CubiCasa samples for {split}")
    return selected


def collect_eval50(output_dir: Path, *, count: int = 50, split: str = "test") -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    start = time.perf_counter()

    response = requests.get(CUBICASA_RECORD_API, timeout=30)
    response.raise_for_status()
    record = response.json()
    source = asdict(CubicasaSource())

    rows: list[dict[str, Any]] = []
    with _remote_zip() as rz:
        selected = _select_samples(rz, split, count)
        for index, base in enumerate(selected, start=1):
            sample_id = base.rstrip("/").split("/")[-1]
            sample_dir = samples_dir / f"{index:03d}_{sample_id}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            image_path = sample_dir / "F1_scaled.png"
            svg_path = sample_dir / "model.svg"
            gt_path = sample_dir / "ground_truth.json"
            if not image_path.exists():
                image_path.write_bytes(rz.read(f"{base}F1_scaled.png"))
            if not svg_path.exists():
                svg_path.write_bytes(rz.read(f"{base}model.svg"))
            ground_truth = align_ground_truth_to_image(parse_cubicasa_svg(svg_path), image_path)
            gt_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
            rows.append(
                {
                    "sample_index": index,
                    "sample_id": sample_id,
                    "dataset": "CubiCasa5K",
                    "split": split,
                    "remote_prefix": base,
                    "image_path": str(image_path),
                    "svg_path": str(svg_path),
                    "ground_truth_path": str(gt_path),
                    "counts": ground_truth["counts"],
                    "source": source,
                }
            )

    manifest_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary = {
        "dataset": "CubiCasa5K",
        "split": split,
        "samples": len(rows),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "download_mode": "remotezip_http_range_subset",
        "seconds": round(time.perf_counter() - start, 4),
        "source": {
            **source,
            "zenodo_title": record.get("metadata", {}).get("title"),
            "zenodo_publication_date": record.get("metadata", {}).get("publication_date"),
        },
        "aggregate_counts": {
            "walls": sum(row["counts"]["walls"] for row in rows),
            "openings": sum(row["counts"]["openings"] for row in rows),
            "objects": sum(row["counts"]["objects"] for row in rows),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval/plan2field_cubicasa50"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()
    print(json.dumps(collect_eval50(args.output_dir, count=args.count, split=args.split), indent=2))


if __name__ == "__main__":
    main()
