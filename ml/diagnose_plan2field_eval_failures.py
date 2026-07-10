from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluate_plan2field3d_eval50 import deterministic_image_baseline, tiled_yolo_vectorsnap
from services.api.buili.spatial.eval_metrics import (
    iou_bbox,
    match_bbox_items,
    match_wall_segments,
)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _dark_density(image: Image.Image, bbox: list[float]) -> float:
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0 = max(0, min(image.width - 1, x0))
    x1 = max(0, min(image.width, x1))
    y0 = max(0, min(image.height - 1, y0))
    y1 = max(0, min(image.height, y1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = image.crop((x0, y0, x1, y1)).convert("L")
    pixels = crop.tobytes()
    return sum(px < 210 for px in pixels) / max(len(pixels), 1)


def _area(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _best_iou_by_gt(preds: list[dict[str, Any]], gts: list[dict[str, Any]], *, class_aware: bool) -> list[float]:
    values: list[float] = []
    for gt in gts:
        best = 0.0
        for pred in preds:
            if class_aware and str(pred.get("kind")) != str(gt.get("kind")):
                continue
            best = max(best, iou_bbox(tuple(pred["bbox"]), tuple(gt["bbox"])))
        values.append(best)
    return values


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "mean": round(float(statistics.fmean(values)), 4),
        "p50": round(float(statistics.median(values)), 4),
        "p75": round(float(sorted(values)[int(0.75 * (len(values) - 1))]), 4),
        "p90": round(float(sorted(values)[int(0.90 * (len(values) - 1))]), 4),
    }


def diagnose(
    manifest_path: Path,
    output_dir: Path,
    *,
    weights: Path,
    confidence: float,
    stride: int,
    max_samples: int | None,
) -> dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_manifest(manifest_path)
    if max_samples:
        rows = rows[:max_samples]

    gt_class_counts: dict[str, Counter[str]] = {"objects": Counter(), "openings": Counter()}
    gt_area: dict[str, list[float]] = defaultdict(list)
    gt_dark: dict[str, list[float]] = defaultdict(list)
    pred_class_counts: dict[str, Counter[str]] = {"objects": Counter(), "openings": Counter()}
    strict_rows: list[dict[str, Any]] = []
    relaxed: dict[str, list[float]] = defaultdict(list)
    best_ious: dict[str, list[float]] = defaultdict(list)
    wall_f1: list[float] = []
    wall_coverage: list[float] = []
    pred_counts: Counter[str] = Counter()
    gt_counts: Counter[str] = Counter()
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    for row in rows:
        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        gt = json.loads(Path(row["ground_truth_path"]).read_text(encoding="utf-8"))
        deterministic_payload, _ = deterministic_image_baseline(image_path, output_dir)
        payload, metadata = tiled_yolo_vectorsnap(
            image_path,
            weights=str(weights),
            confidence=confidence,
            stride=stride,
            wall_payload=deterministic_payload.get("walls", []),
            method_name="diagnostic_tileplandet",
        )
        if len(list(overlay_dir.glob("*.png"))) < 6:
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            for group, color in (("objects", "lime"), ("openings", "cyan")):
                for item in gt.get(group, []):
                    draw.rectangle(item["bbox"], outline=color, width=3)
                    draw.text((item["bbox"][0], item["bbox"][1]), f"G:{item.get('kind')}", fill=color)
            for group, color in (("objects", "red"), ("openings", "blue")):
                for item in payload.get(group, []):
                    draw.rectangle(item["bbox"], outline=color, width=2)
                    draw.text((item["bbox"][0], item["bbox"][1]), f"P:{item.get('kind')}", fill=color)
            overlay.save(overlay_dir / f"{int(row['sample_index']):03d}_{row['sample_id']}_gt_pred.png")

        for group in ("objects", "openings"):
            for item in gt.get(group, []):
                gt_class_counts[group][str(item.get("kind"))] += 1
                gt_area[group].append(_area(item["bbox"]))
                gt_dark[group].append(_dark_density(image, item["bbox"]))
            for item in payload.get(group, []):
                pred_class_counts[group][str(item.get("kind"))] += 1

            pred = payload.get(group, [])
            truth = gt.get(group, [])
            gt_counts[group] += len(truth)
            pred_counts[group] += len(pred)
            strict_threshold = 0.50 if group == "objects" else 0.35
            strict = match_bbox_items(pred, truth, iou_threshold=strict_threshold, class_aware=True)
            strict_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "group": group,
                    "strict_f1": strict["f1"],
                    "strict_tp": strict["true_positive"],
                    "strict_pred": strict["count_pred"],
                    "strict_gt": strict["count_gt"],
                    "seconds": metadata.get("seconds", 0.0),
                }
            )
            for iou_threshold in (0.10, 0.20, 0.35, 0.50):
                for class_aware in (True, False):
                    key = f"{group}_iou{iou_threshold:.2f}_{'class' if class_aware else 'agnostic'}"
                    m = match_bbox_items(
                        pred,
                        truth,
                        iou_threshold=iou_threshold,
                        class_aware=class_aware,
                    )
                    relaxed[key].append(float(m["f1"]))
            best_ious[f"{group}_class"].extend(_best_iou_by_gt(pred, truth, class_aware=True))
            best_ious[f"{group}_agnostic"].extend(_best_iou_by_gt(pred, truth, class_aware=False))

        walls = match_wall_segments(payload.get("walls", []), gt.get("walls", []))
        wall_f1.append(float(walls["f1"]))
        wall_coverage.append(float(walls["gt_coverage_at_threshold"]))

    result = {
        "manifest_path": str(manifest_path),
        "weights": str(weights),
        "confidence": confidence,
        "stride": stride,
        "samples": len(rows),
        "gpu_policy": "CUDA_VISIBLE_DEVICES=7",
        "counts": {
            "gt": dict(gt_counts),
            "pred": dict(pred_counts),
            "gt_classes": {group: dict(counts) for group, counts in gt_class_counts.items()},
            "pred_classes": {group: dict(counts) for group, counts in pred_class_counts.items()},
        },
        "gt_area_px2": {group: _summary(values) for group, values in gt_area.items()},
        "gt_dark_density": {group: _summary(values) for group, values in gt_dark.items()},
        "strict_mean_f1": {
            "objects": round(
                statistics.fmean(row["strict_f1"] for row in strict_rows if row["group"] == "objects"), 4
            ),
            "openings": round(
                statistics.fmean(row["strict_f1"] for row in strict_rows if row["group"] == "openings"), 4
            ),
            "walls": round(statistics.fmean(wall_f1), 4) if wall_f1 else 0.0,
        },
        "wall_gt_coverage_at_12px": round(statistics.fmean(wall_coverage), 4) if wall_coverage else 0.0,
        "relaxed_mean_f1": {
            key: round(statistics.fmean(values), 4) if values else 0.0
            for key, values in sorted(relaxed.items())
        },
        "best_iou_by_gt": {key: _summary(values) for key, values in sorted(best_ious.items())},
        "strict_rows_path": str(output_dir / "strict_rows.jsonl"),
        "overlay_dir": str(overlay_dir),
    }
    (output_dir / "diagnosis_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "strict_rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in strict_rows) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/eval/plan2field_cubicasa50/manifest.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/plan2field_eval_failure_diagnosis"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("data/artifacts/tileplandet_bg_snapshots/yolo11s_tile768_visible85_bg_epoch17_best_snapshot.pt"),
    )
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            diagnose(
                args.manifest,
                args.output_dir,
                weights=args.weights,
                confidence=args.confidence,
                stride=args.stride,
                max_samples=args.max_samples or None,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
