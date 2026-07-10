from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, max(0, int(q * len(ordered)) - 1))])


def analyze_manifest(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    rows = _load_manifest(manifest_path)
    classes: collections.Counter[str] = collections.Counter()
    geometry: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
    duplicate_boxes = 0
    per_sample_counts: list[int] = []
    for row in rows:
        gt = json.loads(Path(row["ground_truth_path"]).read_text(encoding="utf-8"))
        seen: set[tuple[str, tuple[float, ...]]] = set()
        sample_total = 0
        for group in ("walls", "openings", "objects"):
            for item in gt.get(group, []):
                kind = "wall" if group == "walls" else str(item.get("kind", "unknown"))
                bbox = item.get("bbox")
                classes[kind] += 1
                sample_total += 1
                if bbox:
                    width = float(bbox[2] - bbox[0])
                    height = float(bbox[3] - bbox[1])
                    min_side = min(width, height)
                    max_side = max(width, height)
                    aspect = max_side / max(min_side, 1.0)
                    area = width * height
                    geometry[kind].append(
                        {
                            "width": width,
                            "height": height,
                            "min_side": min_side,
                            "aspect": aspect,
                            "area": area,
                        }
                    )
                    key = (kind, tuple(round(float(value), 1) for value in bbox))
                    if key in seen:
                        duplicate_boxes += 1
                    seen.add(key)
        per_sample_counts.append(sample_total)

    class_rows = []
    for kind, count in classes.most_common():
        values = geometry.get(kind, [])
        class_rows.append(
            {
                "class": kind,
                "count": count,
                "median_area": round(statistics.median([v["area"] for v in values]), 3)
                if values
                else 0.0,
                "median_min_side": round(statistics.median([v["min_side"] for v in values]), 3)
                if values
                else 0.0,
                "median_aspect": round(statistics.median([v["aspect"] for v in values]), 3)
                if values
                else 0.0,
                "p95_aspect": round(_percentile([v["aspect"] for v in values], 0.95), 3),
            }
        )
    report = {
        "manifest_path": str(manifest_path),
        "samples": len(rows),
        "instances": sum(classes.values()),
        "duplicate_boxes": duplicate_boxes,
        "instances_per_sample": {
            "median": round(statistics.median(per_sample_counts), 3) if per_sample_counts else 0,
            "p95": round(_percentile([float(v) for v in per_sample_counts], 0.95), 3),
        },
        "classes": class_rows,
        "method_implications": [
            "Walls and doors are thin high-aspect-ratio structures; bbox-only detection is expected to underperform on IoU.",
            "Wall extraction should use vector/morphology snapping rather than a generic object detector.",
            "Object/opening detector should be trained separately from walls to reduce class imbalance and localization conflict.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/plan2field_data_quality.json"))
    args = parser.parse_args()
    print(json.dumps(analyze_manifest(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
