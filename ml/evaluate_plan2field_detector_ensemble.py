from __future__ import annotations

import argparse
import itertools
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
from ml.tune_tileplandet_calibrated_postprocess import _with_features  # noqa: E402
from services.api.buili.spatial.eval_metrics import evaluate_plan_elements  # noqa: E402


DEFAULT_MODELS = [
    {
        "name": "bg_epoch17",
        "weights": "data/artifacts/tileplandet_bg_snapshots/yolo11s_tile768_visible85_bg_epoch17_best_snapshot.pt",
        "confidence": 0.03,
    },
    {
        "name": "visible85_cadsafe",
        "weights": "data/artifacts/tileplandet/tileplandet_yolo11s_visible85_cadsafe_best.pt",
        "confidence": 0.03,
    },
    {
        "name": "clean_visible75",
        "weights": "runs/detect/data/artifacts/tileplandet/yolo11s_clean_visible75_dark08_bg/weights/best.pt",
        "confidence": 0.03,
    },
]

THRESHOLDS = (0.08, 0.10, 0.15)
DARK_THRESHOLDS = (0.0,)
NMS_IOU = (0.35, 0.45)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _load_models(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_MODELS
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("--models-json must contain a list of model specs")
    return rows


def _cache_path(output_dir: Path, manifest: Path, models: list[dict[str, Any]], stride: int) -> Path:
    suffix = "_".join(_safe_name(model["name"]) for model in models)
    return output_dir / f"{manifest.stem}_ensemble_cache_{suffix}_s{stride}.jsonl"


def _build_cache(
    manifest: Path,
    *,
    output_dir: Path,
    models: list[dict[str, Any]],
    stride: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(output_dir, manifest, models, stride)
    if cache_file.exists():
        return _load_jsonl(cache_file)

    rows: list[dict[str, Any]] = []
    for sample in _load_jsonl(manifest):
        image_path = Path(sample["image_path"])
        ground_truth = json.loads(Path(sample["ground_truth_path"]).read_text(encoding="utf-8"))
        deterministic_payload, deterministic_metadata = deterministic_image_baseline(image_path, output_dir)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        per_model: dict[str, Any] = {}
        seconds = float(deterministic_metadata.get("seconds", 0.0)) * 0.0
        for spec in models:
            payload, metadata = tiled_yolo_vectorsnap(
                image_path,
                weights=str(spec["weights"]),
                confidence=float(spec.get("confidence", 0.03)),
                stride=stride,
                wall_payload=deterministic_payload.get("walls", []),
                method_name=f"ensemble_source_{spec['name']}",
            )
            seconds += float(metadata.get("seconds", 0.0))
            per_model[str(spec["name"])] = {
                "objects": _with_features(payload.get("objects", []), image),
                "openings": _with_features(payload.get("openings", []), image),
                "metadata": metadata,
            }
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "image_path": str(image_path),
                "ground_truth": ground_truth,
                "deterministic_walls": deterministic_payload.get("walls", []),
                "sources": per_model,
                "base_seconds": seconds,
            }
        )
    cache_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return rows


def _filter_group(
    rows: list[dict[str, Any]],
    *,
    score_threshold: float,
    dark_threshold: float,
    nms_iou: float,
    source: str,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if float(row.get("score", 0.0)) < score_threshold:
            continue
        if float(row.get("dark_density", 0.0)) < dark_threshold:
            continue
        kept.append(
            {
                **row,
                "id": f"{source}_{row.get('id', index)}",
                "ensemble_source": source,
            }
        )
    return _nms_rows(kept, iou_threshold=nms_iou)


def _payload(cached: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    openings: list[dict[str, Any]] = []
    for source in config["sources"]:
        source_payload = cached["sources"][source]
        objects.extend(
            _filter_group(
                source_payload["objects"],
                score_threshold=config["object_thresholds"][source],
                dark_threshold=config["object_dark_threshold"],
                nms_iou=config["within_source_object_nms"],
                source=source,
            )
        )
        openings.extend(
            _filter_group(
                source_payload["openings"],
                score_threshold=config["opening_thresholds"][source],
                dark_threshold=config["opening_dark_threshold"],
                nms_iou=config["within_source_opening_nms"],
                source=source,
            )
        )
    return {
        "walls": cached["deterministic_walls"],
        "objects": _nms_rows(objects, iou_threshold=config["cross_source_object_nms"]),
        "openings": _nms_rows(openings, iou_threshold=config["cross_source_opening_nms"]),
    }


def _summarize(cached_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    metrics_rows = []
    seconds = []
    for cached in cached_rows:
        start = time.perf_counter()
        payload = _payload(cached, config)
        seconds.append(cached["base_seconds"] + (time.perf_counter() - start))
        metrics_rows.append(evaluate_plan_elements(payload, cached["ground_truth"]))
    summary: dict[str, Any] = {}
    for group in ("objects", "openings", "walls"):
        summary[group] = {
            "mean_precision": round(float(np.mean([row[group]["precision"] for row in metrics_rows])), 4),
            "mean_recall": round(float(np.mean([row[group]["recall"] for row in metrics_rows])), 4),
            "mean_f1": round(float(np.mean([row[group]["f1"] for row in metrics_rows])), 4),
            "true_positive_sum": int(sum(row[group].get("true_positive", 0) for row in metrics_rows)),
        }
    summary["mean_seconds"] = round(float(np.mean(seconds)), 4)
    summary["objective"] = round(summary["objects"]["mean_f1"] + summary["openings"]["mean_f1"], 4)
    return summary


def _candidate_configs(source_names: list[str]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    source_subsets = []
    for size in range(1, len(source_names) + 1):
        source_subsets.extend(itertools.combinations(source_names, size))

    compact_settings = (
        (0.08, 0.10, 0.35),
        (0.10, 0.10, 0.35),
        (0.15, 0.15, 0.35),
        (0.10, 0.15, 0.45),
    )
    for subset in source_subsets:
        for object_threshold, opening_threshold, nms_iou in compact_settings:
            configs.append(
                {
                    "sources": list(subset),
                    "object_thresholds": {source: object_threshold for source in subset},
                    "opening_thresholds": {source: opening_threshold for source in subset},
                    "object_dark_threshold": 0.0,
                    "opening_dark_threshold": 0.0,
                    "within_source_object_nms": nms_iou,
                    "within_source_opening_nms": nms_iou,
                    "cross_source_object_nms": nms_iou,
                    "cross_source_opening_nms": nms_iou,
                }
            )
    return configs


def _tune(train_cache: list[dict[str, Any]], source_names: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    best_config: dict[str, Any] | None = None
    best_summary: dict[str, Any] | None = None
    for config in _candidate_configs(source_names):
        summary = _summarize(train_cache, config)
        ranking = (
            summary["objective"],
            summary["objects"]["mean_f1"],
            summary["openings"]["mean_f1"],
            -summary["mean_seconds"],
        )
        if best_summary is None or ranking > (
            best_summary["objective"],
            best_summary["objects"]["mean_f1"],
            best_summary["openings"]["mean_f1"],
            -best_summary["mean_seconds"],
        ):
            best_config = config
            best_summary = summary
    assert best_config is not None and best_summary is not None
    return best_config, best_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, default=Path("data/train/plan2field_cubicasa_train500_splits/val50.jsonl"))
    parser.add_argument("--eval-manifest", type=Path, default=Path("data/eval/plan2field_cubicasa50/manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/tileplandet_ensemble_eval50"))
    parser.add_argument("--models-json", type=Path)
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
    models = _load_models(args.models_json)
    source_names = [str(model["name"]) for model in models]
    train_cache = _build_cache(args.train_manifest, output_dir=args.output_dir, models=models, stride=args.stride)
    eval_cache = _build_cache(args.eval_manifest, output_dir=args.output_dir, models=models, stride=args.stride)
    best_config, train_summary = _tune(train_cache, source_names)
    eval_summary = _summarize(eval_cache, best_config)
    result = {
        "method": "tileplandet_detector_ensemble",
        "gpu_policy": "CUDA_VISIBLE_DEVICES=7 for detector cache generation",
        "train_manifest": str(args.train_manifest),
        "eval_manifest": str(args.eval_manifest),
        "models": models,
        "best_config_selected_on_train_manifest": best_config,
        "train_summary": train_summary,
        "eval_summary": eval_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ensemble_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
