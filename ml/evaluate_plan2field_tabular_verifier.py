from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.build_plan2field_proposal_verifier_dataset import (  # noqa: E402
    OPENING_CLASSES,
    PROPOSAL_CLASSES,
    _clamp_box,
    _dark_density,
)
from ml.evaluate_plan2field3d_eval50 import (  # noqa: E402
    _aggregate,
    deterministic_image_baseline,
    tiled_yolo_vectorsnap,
)
from services.api.buili.gpu import force_gpu_7  # noqa: E402
from services.api.buili.spatial.eval_metrics import evaluate_plan_elements  # noqa: E402


force_gpu_7()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _feature_vector(row: dict[str, Any]) -> list[float]:
    bbox = [float(value) for value in row["bbox"]]
    x0, y0, x1, y1 = bbox
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    score = max(min(float(row.get("score", 0.0)), 1.0 - 1e-6), 1e-6)
    class_id = int(row.get("class_id", PROPOSAL_CLASSES.index(str(row.get("kind", "door")))))
    one_hot = [1.0 if class_id == index else 0.0 for index in range(len(PROPOSAL_CLASSES))]
    return [
        score,
        math.log(score / (1.0 - score)),
        float(row.get("area_fraction", 0.0)),
        math.log(max(float(row.get("area_fraction", 0.0)), 1e-8)),
        float(row.get("aspect_log", math.log(width / height))),
        abs(float(row.get("aspect_log", math.log(width / height)))),
        float(row.get("dark_density", 0.0)),
        math.log(width),
        math.log(height),
        1.0 if str(row.get("kind", "")) in OPENING_CLASSES else 0.0,
        *one_hot,
    ]


def _rows_to_xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([_feature_vector(row) for row in rows], dtype=np.float32)
    y = np.asarray([1 if row.get("keep") else 0 for row in rows], dtype=np.int64)
    weights = np.ones_like(y, dtype=np.float32)
    positives = max(int(y.sum()), 1)
    negatives = max(int((1 - y).sum()), 1)
    weights[y == 1] = len(y) / (2.0 * positives)
    weights[y == 0] = len(y) / (2.0 * negatives)
    return x, y, weights


def _proposal_f1(rows: list[dict[str, Any]], scores: np.ndarray, *, threshold: float, group: str) -> float:
    tp = fp = fn = 0
    for row, score in zip(rows, scores, strict=False):
        row_group = "openings" if str(row.get("kind")) in OPENING_CLASSES else "objects"
        if row_group != group:
            continue
        pred = float(score) >= threshold
        keep = bool(row.get("keep"))
        if pred and keep:
            tp += 1
        elif pred and not keep:
            fp += 1
        elif not pred and keep:
            fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def _train(dataset_dir: Path, model_path: Path) -> dict[str, Any]:
    train_rows = _read_jsonl(dataset_dir / "train.jsonl")
    val_rows = _read_jsonl(dataset_dir / "val.jsonl")
    x_train, y_train, weights = _rows_to_xy(train_rows)
    x_val, y_val, _ = _rows_to_xy(val_rows)

    start = time.perf_counter()
    model = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.055,
        max_leaf_nodes=17,
        l2_regularization=0.03,
        random_state=20260704,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=18,
    )
    model.fit(x_train, y_train, sample_weight=weights)
    train_seconds = time.perf_counter() - start
    val_scores = model.predict_proba(x_val)[:, 1]

    thresholds = [index / 100 for index in range(5, 96, 5)]
    best_object = max(
        ({"threshold": threshold, "proposal_f1": _proposal_f1(val_rows, val_scores, threshold=threshold, group="objects")} for threshold in thresholds),
        key=lambda item: item["proposal_f1"],
    )
    best_opening = max(
        ({"threshold": threshold, "proposal_f1": _proposal_f1(val_rows, val_scores, threshold=threshold, group="openings")} for threshold in thresholds),
        key=lambda item: item["proposal_f1"],
    )
    summary = {
        "method": "Plan2Field tabular hard-negative verifier",
        "dataset_dir": str(dataset_dir),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_positive_ratio": round(float(y_train.mean()), 6),
        "val_positive_ratio": round(float(y_val.mean()), 6),
        "feature_dim": int(x_train.shape[1]),
        "classes": PROPOSAL_CLASSES,
        "thresholds": {
            "objects": best_object,
            "openings": best_opening,
        },
        "train_seconds": round(train_seconds, 4),
        "model": {
            "family": "sklearn.HistGradientBoostingClassifier",
            "max_iter": 220,
            "max_leaf_nodes": 17,
            "learning_rate": 0.055,
            "l2_regularization": 0.03,
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "summary": summary}, model_path)
    (model_path.parent / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _load_or_train(dataset_dir: Path, model_path: Path) -> tuple[Any, dict[str, Any]]:
    if not model_path.exists():
        _train(dataset_dir, model_path)
    payload = joblib.load(model_path)
    return payload["model"], payload["summary"]


def _enrich_candidate(row: dict[str, Any], image: Image.Image) -> dict[str, Any] | None:
    kind = str(row.get("kind", ""))
    if kind not in PROPOSAL_CLASSES:
        return None
    crop_box = _clamp_box(row["bbox"], width=image.width, height=image.height, pad_ratio=0.65, min_size=96)
    if crop_box is None:
        return None
    x0, y0, x1, y1 = [float(value) for value in row["bbox"]]
    width_px = max(x1 - x0, 1.0)
    height_px = max(y1 - y0, 1.0)
    return {
        **row,
        "class_id": PROPOSAL_CLASSES.index(kind),
        "target_group": "openings" if kind in OPENING_CLASSES else "objects",
        "crop_box": list(crop_box),
        "area_fraction": float((width_px * height_px) / max(image.width * image.height, 1)),
        "aspect_log": float(np.log(width_px / height_px)),
        "dark_density": _dark_density(image, crop_box),
    }


def _score_candidates(image_path: Path, rows: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    enriched = [item for row in rows if (item := _enrich_candidate(row, image)) is not None]
    if not enriched:
        return []
    x = np.asarray([_feature_vector(row) for row in enriched], dtype=np.float32)
    scores = model.predict_proba(x)[:, 1]
    return [{**row, "verifier_score": float(score)} for row, score in zip(enriched, scores, strict=False)]


def _payload_at_threshold(cached: dict[str, Any], *, object_threshold: float, opening_threshold: float) -> dict[str, Any]:
    return {
        "walls": cached["walls"],
        "objects": [
            row for row in cached["objects"] if float(row.get("verifier_score", 0.0)) >= object_threshold
        ],
        "openings": [
            row for row in cached["openings"] if float(row.get("verifier_score", 0.0)) >= opening_threshold
        ],
    }


def _build_cached_rows(
    *,
    manifest: Path,
    weights: str,
    confidence: float,
    stride: int,
    model: Any,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(_read_jsonl(manifest)):
        image_path = Path(sample["image_path"])
        ground_truth = json.loads(Path(sample["ground_truth_path"]).read_text(encoding="utf-8"))
        deterministic_payload, deterministic_metadata = deterministic_image_baseline(image_path, output_dir)
        tile_payload, tile_metadata = tiled_yolo_vectorsnap(
            image_path,
            weights=weights,
            confidence=confidence,
            stride=stride,
            wall_payload=deterministic_payload.get("walls", []),
            method_name="tileplandet_for_tabular_verifier",
        )
        start = time.perf_counter()
        scored = _score_candidates(
            image_path,
            [*tile_payload.get("objects", []), *tile_payload.get("openings", [])],
            model,
        )
        verifier_seconds = time.perf_counter() - start
        rows.append(
            {
                "sample_id": str(sample["sample_id"]),
                "sample_index": sample_index,
                "ground_truth": ground_truth,
                "walls": deterministic_payload.get("walls", []),
                "objects": [row for row in scored if row.get("target_group") == "objects"],
                "openings": [row for row in scored if row.get("target_group") == "openings"],
                "tile_payload": tile_payload,
                "tile_metadata": tile_metadata,
                "deterministic_metadata": deterministic_metadata,
                "verifier_seconds": verifier_seconds,
            }
        )
    return rows


def _mean_f1(cached_rows: list[dict[str, Any]], *, object_threshold: float, opening_threshold: float) -> dict[str, float]:
    metrics = [
        evaluate_plan_elements(
            _payload_at_threshold(cached, object_threshold=object_threshold, opening_threshold=opening_threshold),
            cached["ground_truth"],
        )
        for cached in cached_rows
    ]
    return {
        "object_f1": round(float(np.mean([row["objects"]["f1"] for row in metrics])), 4),
        "opening_f1": round(float(np.mean([row["openings"]["f1"] for row in metrics])), 4),
        "wall_f1": round(float(np.mean([row["walls"]["f1"] for row in metrics])), 4),
    }


def _tune_final_thresholds(cached_rows: list[dict[str, Any]], initial: dict[str, Any]) -> dict[str, Any]:
    grid = [index / 100 for index in range(5, 96, 5)]
    best_object = {"threshold": float(initial["thresholds"]["objects"]["threshold"]), "f1": -1.0}
    best_opening = {"threshold": float(initial["thresholds"]["openings"]["threshold"]), "f1": -1.0}
    for threshold in grid:
        f1 = _mean_f1(cached_rows, object_threshold=threshold, opening_threshold=0.0)["object_f1"]
        if f1 > best_object["f1"]:
            best_object = {"threshold": threshold, "f1": f1}
        f1 = _mean_f1(cached_rows, object_threshold=0.0, opening_threshold=threshold)["opening_f1"]
        if f1 > best_opening["f1"]:
            best_opening = {"threshold": threshold, "f1": f1}
    combined = _mean_f1(
        cached_rows,
        object_threshold=float(best_object["threshold"]),
        opening_threshold=float(best_opening["threshold"]),
    )
    return {"objects": best_object, "openings": best_opening, "combined": combined}


def _result_rows(
    cached_rows: list[dict[str, Any]],
    *,
    object_threshold: float,
    opening_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cached in cached_rows:
        base_payload = cached["tile_payload"]
        verified_payload = _payload_at_threshold(
            cached,
            object_threshold=object_threshold,
            opening_threshold=opening_threshold,
        )
        for variant, payload, seconds in (
            ("tileplandet_vectorsnap", base_payload, float(cached["tile_metadata"].get("seconds", 0.0))),
            (
                "tileplandet_tabular_verifier",
                verified_payload,
                float(cached["tile_metadata"].get("seconds", 0.0)) + float(cached["verifier_seconds"]),
            ),
        ):
            rows.append(
                {
                    "sample_id": cached["sample_id"],
                    "sample_index": cached["sample_index"],
                    "variant": variant,
                    "metrics": evaluate_plan_elements(payload, cached["ground_truth"]),
                    "metadata": {
                        **cached["tile_metadata"],
                        "seconds": round(seconds, 4),
                    },
                    "counts": {
                        "pred": {key: len(payload.get(key, [])) for key in ("walls", "openings", "objects")},
                        "gt": {key: len(cached["ground_truth"].get(key, [])) for key in ("walls", "openings", "objects")},
                    },
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tune-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--confidence", type=float, default=0.03)
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, training_summary = _load_or_train(args.dataset_dir, args.model_path)

    tune_rows = _build_cached_rows(
        manifest=args.tune_manifest,
        weights=args.weights,
        confidence=args.confidence,
        stride=args.stride,
        model=model,
        output_dir=args.output_dir,
    )
    tuned = _tune_final_thresholds(tune_rows, training_summary)

    eval_rows = _build_cached_rows(
        manifest=args.eval_manifest,
        weights=args.weights,
        confidence=args.confidence,
        stride=args.stride,
        model=model,
        output_dir=args.output_dir,
    )
    result_rows = _result_rows(
        eval_rows,
        object_threshold=float(tuned["objects"]["threshold"]),
        opening_threshold=float(tuned["openings"]["threshold"]),
    )
    detail_path = args.output_dir / "eval_results.jsonl"
    detail_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in result_rows) + "\n",
        encoding="utf-8",
    )
    summary = {
        "method": "Plan2Field tabular hard-negative verifier",
        "training_summary": training_summary,
        "threshold_tuning": tuned,
        "tile_confidence": args.confidence,
        "tile_stride": args.stride,
        "weights": args.weights,
        "model_path": str(args.model_path),
        "detail_path": str(detail_path),
        "aggregate": {
            variant: _aggregate(result_rows, variant)
            for variant in ("tileplandet_vectorsnap", "tileplandet_tabular_verifier")
        },
    }
    summary_path = args.output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
