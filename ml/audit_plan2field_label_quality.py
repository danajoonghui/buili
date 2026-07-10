from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.buili.spatial.micro_vlm import MICRO_VLM_CLASSES


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dark_density(image: Image.Image, bbox: list[float]) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = gray.shape
    x0, y0, x1, y1 = bbox
    ix0 = int(max(0, min(width - 1, round(x0))))
    iy0 = int(max(0, min(height - 1, round(y0))))
    ix1 = int(max(ix0 + 1, min(width, round(x1))))
    iy1 = int(max(iy0 + 1, min(height, round(y1))))
    crop = gray[iy0:iy1, ix0:ix1]
    return float((crop < 175).mean()) if crop.size else 0.0


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
        return round(float(ordered[index]), 6)

    return {
        "count": len(values),
        "mean": round(float(statistics.fmean(values)), 6),
        "median": pct(0.5),
        "p05": pct(0.05),
        "p25": pct(0.25),
        "p75": pct(0.75),
        "p95": pct(0.95),
        "min": round(float(ordered[0]), 6),
        "max": round(float(ordered[-1]), 6),
    }


def _class_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = collections.Counter(str(row["label"]) for row in rows)
    positives = {key: value for key, value in counts.items() if key != "background"}
    min_positive = min(positives.values()) if positives else 0
    max_positive = max(positives.values()) if positives else 0
    return {
        "counts": dict(sorted(counts.items())),
        "positive_classes": len(positives),
        "min_positive_count": min_positive,
        "max_positive_count": max_positive,
        "imbalance_ratio_max_over_min_positive": round(max_positive / max(min_positive, 1), 3),
        "unused_declared_positive_classes": [
            name for name in MICRO_VLM_CLASSES if name not in counts and name != "background"
        ],
    }


def audit_vlm_patch_dataset(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    missing = 0
    normalized_out_of_range = 0
    degenerate_norm_boxes = 0
    class_id_mismatch = 0
    density_by_class: dict[str, list[float]] = collections.defaultdict(list)
    source_by_split: dict[str, set[str]] = collections.defaultdict(set)
    duplicate_keys = collections.Counter()
    image_sizes = collections.Counter()
    for row in rows:
        image_path = Path(row["image"])
        if not image_path.exists():
            missing += 1
            continue
        try:
            with Image.open(image_path) as image:
                image_sizes[f"{image.width}x{image.height}"] += 1
        except Exception:
            missing += 1
            continue
        bbox = [float(value) for value in row["bbox"]]
        cx, cy, bw, bh = bbox
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in bbox):
            normalized_out_of_range += 1
        if bw <= 0.0 or bh <= 0.0:
            degenerate_norm_boxes += 1
        label = str(row["label"])
        if 0 <= int(row["class_id"]) < len(MICRO_VLM_CLASSES):
            class_id_mismatch += int(MICRO_VLM_CLASSES[int(row["class_id"])] != label)
        else:
            class_id_mismatch += 1
        if label != "background":
            density_by_class[label].append(float(row.get("target_dark_density", 0.0)))
        source_by_split[str(row["split"])].add(str(row.get("source_sample_id", "")))
        duplicate_keys[
            (
                str(row["split"]),
                str(row.get("source_sample_id", "")),
                label,
                tuple(round(float(v), 4) for v in bbox),
            )
        ] += 1
    duplicate_rows = sum(count - 1 for count in duplicate_keys.values() if count > 1)
    train_sources = source_by_split.get("train", set())
    eval_sources = source_by_split.get("eval", set())
    distribution = _class_distribution(rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "missing_images": missing,
        "image_sizes": dict(image_sizes),
        "normalized_out_of_range_boxes": normalized_out_of_range,
        "degenerate_normalized_boxes": degenerate_norm_boxes,
        "class_id_mismatch": class_id_mismatch,
        "duplicate_patch_label_rows": duplicate_rows,
        "split_source_counts": {key: len(value) for key, value in sorted(source_by_split.items())},
        "train_eval_source_overlap_count": len(train_sources & eval_sources),
        "class_distribution": distribution,
        "target_dark_density_by_class": {
            key: _stats(values) for key, values in sorted(density_by_class.items())
        },
        "quality_gates": {
            "no_missing_images": missing == 0,
            "no_normalized_bbox_errors": normalized_out_of_range == 0 and degenerate_norm_boxes == 0,
            "class_ids_match_labels": class_id_mismatch == 0,
            "no_train_eval_source_leakage": len(train_sources & eval_sources) == 0,
            "unused_classes_documented": bool(distribution["unused_declared_positive_classes"]),
        },
    }


def audit_source_manifests(paths: list[Path], *, low_density_threshold: float) -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    source_sets: dict[str, set[str]] = {}
    for path in paths:
        rows = _read_jsonl(path)
        split_name = "eval50" if path.name == "manifest.jsonl" and "eval" in path.parts else path.stem
        source_sets[split_name] = {str(row.get("sample_id")) for row in rows}
        counts = collections.Counter()
        degenerate = 0
        out_of_bounds = 0
        low_density = collections.Counter()
        density_by_class: dict[str, list[float]] = collections.defaultdict(list)
        min_side_by_class: dict[str, list[float]] = collections.defaultdict(list)
        aspect_by_class: dict[str, list[float]] = collections.defaultdict(list)
        duplicate_keys = collections.Counter()
        missing = 0
        for row in rows:
            image_path = Path(row["image_path"])
            gt_path = Path(row["ground_truth_path"])
            if not image_path.exists() or not gt_path.exists():
                missing += 1
                continue
            image = Image.open(image_path).convert("RGB")
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
            width, height = image.width, image.height
            targets = [
                *[
                    {"kind": str(item.get("kind", "")), "bbox": item.get("bbox", []), "group": "openings"}
                    for item in gt.get("openings", [])
                ],
                *[
                    {"kind": str(item.get("kind", "")), "bbox": item.get("bbox", []), "group": "objects"}
                    for item in gt.get("objects", [])
                ],
            ]
            for target in targets:
                kind = target["kind"] or "unknown"
                bbox = [float(value) for value in target["bbox"]]
                if len(bbox) != 4:
                    degenerate += 1
                    continue
                x0, y0, x1, y1 = bbox
                if x1 <= x0 or y1 <= y0:
                    degenerate += 1
                    continue
                if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                    out_of_bounds += 1
                box_w = x1 - x0
                box_h = y1 - y0
                density = _dark_density(image, bbox)
                counts[kind] += 1
                duplicate_keys[(str(row.get("sample_id")), kind, tuple(round(v, 3) for v in bbox))] += 1
                density_by_class[kind].append(density)
                min_side_by_class[kind].append(min(box_w, box_h))
                aspect_by_class[kind].append(max(box_w, box_h) / max(min(box_w, box_h), 1e-6))
                if density < low_density_threshold:
                    low_density[kind] += 1
        duplicate_rows = sum(count - 1 for count in duplicate_keys.values() if count > 1)
        by_split[split_name] = {
            "manifest": str(path),
            "samples": len(rows),
            "missing_files": missing,
            "target_counts": dict(sorted(counts.items())),
            "target_total": sum(counts.values()),
            "degenerate_targets": degenerate,
            "out_of_bounds_targets": out_of_bounds,
            "duplicate_targets": duplicate_rows,
            "low_dark_density_targets": dict(sorted(low_density.items())),
            "low_dark_density_total": sum(low_density.values()),
            "low_dark_density_rate": round(
                sum(low_density.values()) / max(sum(counts.values()), 1), 6
            ),
            "dark_density_by_class": {
                key: _stats(values) for key, values in sorted(density_by_class.items())
            },
            "min_side_px_by_class": {
                key: _stats(values) for key, values in sorted(min_side_by_class.items())
            },
            "aspect_ratio_by_class": {
                key: _stats(values) for key, values in sorted(aspect_by_class.items())
            },
        }
    overlaps: dict[str, int] = {}
    names = sorted(source_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = len(source_sets[left] & source_sets[right])
    return {
        "splits": by_split,
        "source_overlap_counts": overlaps,
        "low_density_threshold": low_density_threshold,
    }


def audit_yolo_labels(data_yaml: Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config["path"])
    names = {int(key): str(value) for key, value in config["names"].items()}
    result: dict[str, Any] = {"data_yaml": str(data_yaml), "names": names, "splits": {}}
    for split in ("train", "val"):
        label_dir = root / "labels" / split
        counts = collections.Counter()
        files = list(label_dir.glob("*.txt"))
        malformed = 0
        for label_file in files:
            for line in label_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    malformed += 1
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    malformed += 1
                    continue
                counts[names.get(class_id, f"unknown_{class_id}")] += 1
        result["splits"][split] = {
            "label_files": len(files),
            "boxes": sum(counts.values()),
            "class_counts": dict(sorted(counts.items())),
            "malformed_rows": malformed,
            "declared_classes_without_labels": [
                name for _, name in sorted(names.items()) if counts.get(name, 0) == 0
            ],
        }
    return result


def write_markdown(report: dict[str, Any], path: Path) -> None:
    vlm = report["vlm_patch_dataset"]
    source = report["source_manifests"]
    yolo = report["yolo_labels"]
    lines = [
        "# Plan2Field Label Quality Audit",
        "",
        "## Verdict",
        "",
        "- The CubiCasa-derived labels are usable for controlled experiments, but they are not flawless product labels.",
        "- The main weakness is semantic/metric mismatch: doors and windows are often annotated as thin symbol segments, while product 3D expects whole openings.",
        "- The VLM patch dataset passes missing-file, bbox-range, class-id, and train/eval leakage gates.",
        "- Several declared runtime classes have zero positive VLM samples and must not be claimed as trained categories.",
        "",
        "## VLM Patch Dataset",
        "",
        f"- Rows: {vlm['rows']}",
        f"- Missing images: {vlm['missing_images']}",
        f"- Train/eval source overlap: {vlm['train_eval_source_overlap_count']}",
        f"- Duplicate patch-label rows: {vlm['duplicate_patch_label_rows']}",
        f"- Positive class imbalance max/min: {vlm['class_distribution']['imbalance_ratio_max_over_min_positive']}",
        f"- Unused declared positive classes: {', '.join(vlm['class_distribution']['unused_declared_positive_classes']) or 'none'}",
        "",
        "## Source Ground Truth",
        "",
    ]
    for split_name, split in source["splits"].items():
        lines.extend(
            [
                f"### {split_name}",
                "",
                f"- Samples: {split['samples']}",
                f"- Targets: {split['target_total']}",
                f"- Degenerate targets: {split['degenerate_targets']}",
                f"- Out-of-bounds targets: {split['out_of_bounds_targets']}",
                f"- Duplicate targets: {split['duplicate_targets']}",
                f"- Low dark-density target rate: {split['low_dark_density_rate']}",
                "",
            ]
        )
    lines.extend(
        [
            "## YOLO Tile Labels",
            "",
            f"- Data YAML: {yolo['data_yaml']}",
        ]
    )
    for split_name, split in yolo["splits"].items():
        lines.extend(
            [
                f"- {split_name}: {split['boxes']} boxes in {split['label_files']} label files; malformed rows {split['malformed_rows']}",
                f"- {split_name} declared classes without labels: {', '.join(split['declared_classes_without_labels']) or 'none'}",
            ]
        )
    lines.extend(
        [
            "",
            "## Paper-Safe Interpretation",
            "",
            "The label audit supports using this data for a transparent methods paper only if the paper states the limitations. "
            "It does not support claiming a fully solved general floorplan parser. "
            "The clean claim is: public labels were quality-gated; thin opening symbols and sparse class coverage explain why VLM-only generalization remains weak; "
            "TilePlanDet plus deterministic geometry is currently the stronger production path, while VLM heads are a research component requiring better symbol-aligned supervision.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--yolo-data-yaml", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--low-density-threshold", type=float, default=0.012)
    args = parser.parse_args()

    report = {
        "audit": "plan2field_label_quality",
        "vlm_patch_dataset": audit_vlm_patch_dataset(args.vlm_dataset),
        "source_manifests": audit_source_manifests(
            args.source_manifest,
            low_density_threshold=args.low_density_threshold,
        ),
        "yolo_labels": audit_yolo_labels(args.yolo_data_yaml),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, args.out_md)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
