from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluate_plan2field3d_eval50 import (  # noqa: E402
    _nms_rows,
    deterministic_image_baseline,
    tiled_yolo_vectorsnap,
)
from services.api.buili.spatial.eval_metrics import evaluate_plan_elements  # noqa: E402


OBJECT_KINDS = ("bathtub", "cabinet_run", "column", "fixture", "shower", "sink", "toilet", "water_heater")
OPENING_KINDS = ("door", "window")
THRESHOLD_GRID = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25)
DARK_GRID = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _dark_density(image: np.ndarray, bbox: list[float]) -> float:
    x0, y0, x1, y1 = [int(round(value)) for value in bbox]
    height, width = image.shape[:2]
    x0 = max(0, min(width - 1, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = image[y0:y1, x0:x1, :]
    gray = crop.mean(axis=2)
    return float((gray < 220).mean())


def _with_features(rows: list[dict[str, Any]], image: np.ndarray) -> list[dict[str, Any]]:
    featured: list[dict[str, Any]] = []
    for row in rows:
        bbox = [float(value) for value in row["bbox"]]
        x0, y0, x1, y1 = bbox
        width = max(x1 - x0, 1.0)
        height = max(y1 - y0, 1.0)
        featured.append(
            {
                **row,
                "bbox": bbox,
                "score": float(row.get("score", 0.0)),
                "dark_density": _dark_density(image, bbox),
                "area_px": width * height,
                "aspect_ratio": max(width, height) / max(min(width, height), 1.0),
            }
        )
    return featured


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    score_thresholds: dict[str, float],
    dark_thresholds: dict[str, float],
    nms_iou: float,
) -> list[dict[str, Any]]:
    kept = [
        row
        for row in rows
        if row.get("score", 0.0) >= score_thresholds.get(str(row.get("kind")), 0.03)
        and row.get("dark_density", 0.0) >= dark_thresholds.get(str(row.get("kind")), 0.0)
    ]
    return _nms_rows(kept, iou_threshold=nms_iou)


def _apply_config(cached: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "walls": cached["deterministic_walls"],
        "objects": _filter_rows(
            cached["objects"],
            score_thresholds=config["object_score_thresholds"],
            dark_thresholds=config["object_dark_thresholds"],
            nms_iou=config["object_nms_iou"],
        ),
        "openings": _filter_rows(
            cached["openings"],
            score_thresholds=config["opening_score_thresholds"],
            dark_thresholds=config["opening_dark_thresholds"],
            nms_iou=config["opening_nms_iou"],
        ),
    }


def _evaluate_cached(cached_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    metrics_rows: list[dict[str, Any]] = []
    seconds_rows: list[float] = []
    for cached in cached_rows:
        start = time.perf_counter()
        payload = _apply_config(cached, config)
        seconds_rows.append(time.perf_counter() - start + cached["base_seconds"])
        metrics_rows.append(evaluate_plan_elements(payload, cached["ground_truth"]))

    summary: dict[str, Any] = {}
    for group in ("objects", "openings", "walls"):
        summary[group] = {
            "mean_precision": round(float(np.mean([row[group]["precision"] for row in metrics_rows])), 4),
            "mean_recall": round(float(np.mean([row[group]["recall"] for row in metrics_rows])), 4),
            "mean_f1": round(float(np.mean([row[group]["f1"] for row in metrics_rows])), 4),
            "true_positive_sum": int(sum(row[group].get("true_positive", 0) for row in metrics_rows)),
        }
    summary["mean_seconds"] = round(float(np.mean(seconds_rows)), 4)
    summary["objective"] = round(summary["objects"]["mean_f1"] + summary["openings"]["mean_f1"], 4)
    return summary


def _default_config() -> dict[str, Any]:
    return {
        "object_score_thresholds": {kind: 0.03 for kind in OBJECT_KINDS},
        "opening_score_thresholds": {kind: 0.03 for kind in OPENING_KINDS},
        "object_dark_thresholds": {kind: 0.0 for kind in OBJECT_KINDS},
        "opening_dark_thresholds": {kind: 0.0 for kind in OPENING_KINDS},
        "object_nms_iou": 0.35,
        "opening_nms_iou": 0.35,
    }


def _coordinate_tune(cached_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _default_config()
    best = _evaluate_cached(cached_rows, config)
    for _ in range(3):
        for group, key, kinds in (
            ("objects", "object_score_thresholds", OBJECT_KINDS),
            ("openings", "opening_score_thresholds", OPENING_KINDS),
        ):
            for kind in kinds:
                local_best = best
                local_value = config[key][kind]
                for value in THRESHOLD_GRID:
                    candidate = json.loads(json.dumps(config))
                    candidate[key][kind] = value
                    summary = _evaluate_cached(cached_rows, candidate)
                    if (summary["objective"], summary[group]["mean_f1"]) > (
                        local_best["objective"],
                        local_best[group]["mean_f1"],
                    ):
                        local_best = summary
                        local_value = value
                config[key][kind] = local_value
                best = local_best
        for group, key, kinds in (
            ("objects", "object_dark_thresholds", OBJECT_KINDS),
            ("openings", "opening_dark_thresholds", OPENING_KINDS),
        ):
            for kind in kinds:
                local_best = best
                local_value = config[key][kind]
                for value in DARK_GRID:
                    candidate = json.loads(json.dumps(config))
                    candidate[key][kind] = value
                    summary = _evaluate_cached(cached_rows, candidate)
                    if (summary["objective"], summary[group]["mean_f1"]) > (
                        local_best["objective"],
                        local_best[group]["mean_f1"],
                    ):
                        local_best = summary
                        local_value = value
                config[key][kind] = local_value
                best = local_best
        for group, key, values in (
            ("objects", "object_nms_iou", (0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.55)),
            ("openings", "opening_nms_iou", (0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.55)),
        ):
            local_best = best
            local_value = config[key]
            for value in values:
                candidate = json.loads(json.dumps(config))
                candidate[key] = value
                summary = _evaluate_cached(cached_rows, candidate)
                if (summary["objective"], summary[group]["mean_f1"]) > (
                    local_best["objective"],
                    local_best[group]["mean_f1"],
                ):
                    local_best = summary
                    local_value = value
            config[key] = local_value
            best = local_best
    return config, best


def _build_cache(
    manifest: Path,
    *,
    weights: str,
    confidence: float,
    stride: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    cache_path = output_dir / f"{manifest.stem}_tile_cache_conf{str(confidence).replace('.', '_')}.jsonl"
    if cache_path.exists():
        return [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line]

    cached_rows: list[dict[str, Any]] = []
    for sample in _load_manifest(manifest):
        image_path = Path(sample["image_path"])
        ground_truth = json.loads(Path(sample["ground_truth_path"]).read_text(encoding="utf-8"))
        deterministic_payload, deterministic_metadata = deterministic_image_baseline(image_path, output_dir)
        tile_payload, tile_metadata = tiled_yolo_vectorsnap(
            image_path,
            weights=weights,
            confidence=confidence,
            stride=stride,
            wall_payload=deterministic_payload.get("walls", []),
        )
        image = np.asarray(Image.open(image_path).convert("RGB"))
        cached_rows.append(
            {
                "sample_id": sample["sample_id"],
                "image_path": str(image_path),
                "ground_truth": ground_truth,
                "deterministic_walls": deterministic_payload.get("walls", []),
                "objects": _with_features(tile_payload.get("objects", []), image),
                "openings": _with_features(tile_payload.get("openings", []), image),
                "base_seconds": float(tile_metadata.get("seconds", 0.0))
                + float(deterministic_metadata.get("seconds", 0.0)) * 0.0,
            }
        )

    cache_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in cached_rows) + "\n",
        encoding="utf-8",
    )
    return cached_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, default=Path("data/train/plan2field_cubicasa_train500_splits/val50.jsonl"))
    parser.add_argument("--eval-manifest", type=Path, default=Path("data/eval/plan2field_cubicasa50/manifest.jsonl"))
    parser.add_argument("--weights", default="data/artifacts/tileplandet/tileplandet_yolo11s_visible85_cadsafe_best.pt")
    parser.add_argument("--confidence", type=float, default=0.03)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/tileplandet_calibrated_postprocess"))
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _build_cache(
        args.train_manifest,
        weights=args.weights,
        confidence=args.confidence,
        stride=args.stride,
        output_dir=args.output_dir,
    )
    eval_cache = _build_cache(
        args.eval_manifest,
        weights=args.weights,
        confidence=args.confidence,
        stride=args.stride,
        output_dir=args.output_dir,
    )

    base_config = _default_config()
    base_val = _evaluate_cached(train_cache, base_config)
    config, tuned_val = _coordinate_tune(train_cache)
    tuned_eval = _evaluate_cached(eval_cache, config)

    result = {
        "method": "TilePlanDet-Calibrated: val50-tuned score/dark-density/NMS postprocess over frozen detector predictions",
        "gpu_policy": "CUDA_VISIBLE_DEVICES=7 for detector cache generation",
        "train_manifest": str(args.train_manifest),
        "eval_manifest": str(args.eval_manifest),
        "weights": args.weights,
        "base_detector_confidence": args.confidence,
        "stride": args.stride,
        "selection_rule": "Coordinate search maximizes object mean F1 + opening mean F1 on val50 only.",
        "base_val": base_val,
        "tuned_val": tuned_val,
        "tuned_eval": tuned_eval,
        "config": config,
    }
    (args.output_dir / "calibration_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "calibration_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
