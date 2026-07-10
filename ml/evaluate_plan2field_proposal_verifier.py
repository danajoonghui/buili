from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.buili.gpu import assert_gpu_7, force_gpu_7, gpu_policy

force_gpu_7()

from ml.build_plan2field_proposal_verifier_dataset import (
    OPENING_CLASSES,
    PROPOSAL_CLASSES,
    _clamp_box,
    _dark_density,
)
from ml.evaluate_plan2field3d_eval50 import (
    _aggregate,
    deterministic_image_baseline,
    tiled_yolo_vectorsnap,
)
from ml.train_plan2field_proposal_verifier import ProposalVerifier, _feature_vector
from services.api.buili.spatial.eval_metrics import evaluate_plan_elements
from services.api.buili.spatial.vlm_primary import _clip_tensor


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_verifier(path: Path) -> tuple[ProposalVerifier, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    model = ProposalVerifier(
        encoder_id=str(config["encoder_id"]),
        feature_dim=int(config["feature_dim"]),
        hidden=int(config["hidden"]),
        freeze_vision=True,
    )
    model.load_state_dict(checkpoint["head_state"], strict=False)
    model.to("cuda:0")
    model.eval()
    return model, checkpoint.get("summary", {})


def _enrich_candidate(row: dict[str, Any], image: Image.Image) -> dict[str, Any] | None:
    kind = str(row.get("kind", ""))
    if kind not in PROPOSAL_CLASSES:
        return None
    crop_box = _clamp_box(
        row["bbox"],
        width=image.width,
        height=image.height,
        pad_ratio=0.65,
        min_size=96,
    )
    if crop_box is None:
        return None
    x0, y0, x1, y1 = [float(value) for value in row["bbox"]]
    width_px = max(x1 - x0, 1.0)
    height_px = max(y1 - y0, 1.0)
    enriched = {
        **row,
        "class_id": PROPOSAL_CLASSES.index(kind),
        "target_group": "openings" if kind in OPENING_CLASSES else "objects",
        "crop_box": list(crop_box),
        "area_fraction": float((width_px * height_px) / max(image.width * image.height, 1)),
        "aspect_log": float(np.log(width_px / height_px)),
        "dark_density": _dark_density(image, crop_box),
    }
    return enriched


def _score_candidates(
    *,
    image_path: Path,
    candidates: list[dict[str, Any]],
    model: ProposalVerifier,
    image_size: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    enriched = [item for row in candidates if (item := _enrich_candidate(row, image)) is not None]
    if not enriched:
        return [], round(time.perf_counter() - start, 4)
    scores: list[float] = []
    with torch.no_grad():
        for offset in range(0, len(enriched), batch_size):
            batch = enriched[offset : offset + batch_size]
            pixels = torch.stack(
                [
                    _clip_tensor(image.crop(tuple(row["crop_box"])), image_size)
                    for row in batch
                ]
            ).to("cuda:0")
            features = torch.stack(
                [_feature_vector(row, class_count=len(PROPOSAL_CLASSES)) for row in batch]
            ).to("cuda:0")
            output = model(pixels, features)
            scores.extend(
                float(value)
                for value in torch.sigmoid(output["keep_logit"]).detach().cpu().numpy()
            )
    scored: list[dict[str, Any]] = []
    for row, score in zip(enriched, scores, strict=False):
        out = {**row, "verifier_score": score}
        scored.append(out)
    return scored, round(time.perf_counter() - start, 4)


def _payload_at_threshold(
    cached: dict[str, Any],
    *,
    object_threshold: float,
    opening_threshold: float,
) -> dict[str, Any]:
    objects = [
        row
        for row in cached["objects"]
        if float(row.get("verifier_score", 0.0)) >= object_threshold
    ]
    openings = [
        row
        for row in cached["openings"]
        if float(row.get("verifier_score", 0.0)) >= opening_threshold
    ]
    return {"walls": cached["walls"], "objects": objects, "openings": openings}


def _mean_group_f1(
    cached_rows: list[dict[str, Any]],
    *,
    object_threshold: float,
    opening_threshold: float,
) -> dict[str, float]:
    metrics = []
    for cached in cached_rows:
        payload = _payload_at_threshold(
            cached,
            object_threshold=object_threshold,
            opening_threshold=opening_threshold,
        )
        metrics.append(evaluate_plan_elements(payload, cached["ground_truth"]))
    return {
        "object_f1": round(float(np.mean([row["objects"]["f1"] for row in metrics])), 4),
        "opening_f1": round(float(np.mean([row["openings"]["f1"] for row in metrics])), 4),
        "wall_f1": round(float(np.mean([row["walls"]["f1"] for row in metrics])), 4),
    }


def _tune_thresholds(cached_rows: list[dict[str, Any]]) -> dict[str, Any]:
    thresholds = [index / 100 for index in range(5, 96, 5)]
    best_object = {"threshold": 0.5, "f1": -1.0}
    best_opening = {"threshold": 0.5, "f1": -1.0}
    for threshold in thresholds:
        group = _mean_group_f1(cached_rows, object_threshold=threshold, opening_threshold=0.0)
        if group["object_f1"] > best_object["f1"]:
            best_object = {"threshold": threshold, "f1": group["object_f1"]}
        group = _mean_group_f1(cached_rows, object_threshold=0.0, opening_threshold=threshold)
        if group["opening_f1"] > best_opening["f1"]:
            best_opening = {"threshold": threshold, "f1": group["opening_f1"]}
    combined = _mean_group_f1(
        cached_rows,
        object_threshold=float(best_object["threshold"]),
        opening_threshold=float(best_opening["threshold"]),
    )
    return {
        "object_threshold": float(best_object["threshold"]),
        "opening_threshold": float(best_opening["threshold"]),
        "val_object_f1": float(best_object["f1"]),
        "val_opening_f1": float(best_opening["f1"]),
        "combined": combined,
    }


def _build_cached_rows(
    *,
    manifest_path: Path,
    output_dir: Path,
    weights: str,
    verifier: ProposalVerifier,
    tile_confidence: float,
    tile_stride: int,
    image_size: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows = _load_manifest(manifest_path)
    cached_rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(rows):
        image_path = Path(sample["image_path"])
        ground_truth = json.loads(Path(sample["ground_truth_path"]).read_text(encoding="utf-8"))
        deterministic_payload, deterministic_metadata = deterministic_image_baseline(image_path, output_dir)
        tile_payload, tile_metadata = tiled_yolo_vectorsnap(
            image_path,
            weights=weights,
            confidence=tile_confidence,
            stride=tile_stride,
            wall_payload=deterministic_payload.get("walls", []),
            method_name="tileplandet_for_proposal_verifier",
        )
        scored, verifier_seconds = _score_candidates(
            image_path=image_path,
            candidates=[*tile_payload.get("objects", []), *tile_payload.get("openings", [])],
            model=verifier,
            image_size=image_size,
            batch_size=batch_size,
        )
        cached_rows.append(
            {
                "sample_id": str(sample["sample_id"]),
                "sample_index": sample_index,
                "ground_truth": ground_truth,
                "walls": deterministic_payload.get("walls", []),
                "objects": [row for row in scored if row.get("target_group") == "objects"],
                "openings": [row for row in scored if row.get("target_group") == "openings"],
                "tile_payload": tile_payload,
                "deterministic_metadata": deterministic_metadata,
                "tile_metadata": tile_metadata,
                "verifier_seconds": verifier_seconds,
            }
        )
    return cached_rows


def _result_rows(
    cached_rows: list[dict[str, Any]],
    *,
    object_threshold: float,
    opening_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cached in cached_rows:
        base_payload = cached["tile_payload"]
        base_metrics = evaluate_plan_elements(base_payload, cached["ground_truth"])
        rows.append(
            {
                "sample_id": cached["sample_id"],
                "sample_index": cached["sample_index"],
                "variant": "tileplandet_vectorsnap",
                "metrics": base_metrics,
                "metadata": cached["tile_metadata"],
                "counts": {
                    "pred": {key: len(base_payload.get(key, [])) for key in ("walls", "openings", "objects")},
                    "gt": {key: len(cached["ground_truth"].get(key, [])) for key in ("walls", "openings", "objects")},
                },
            }
        )
        payload = _payload_at_threshold(
            cached,
            object_threshold=object_threshold,
            opening_threshold=opening_threshold,
        )
        metrics = evaluate_plan_elements(payload, cached["ground_truth"])
        total_seconds = (
            float(cached["deterministic_metadata"].get("seconds", 0.0))
            + float(cached["tile_metadata"].get("seconds", 0.0))
            + float(cached["verifier_seconds"])
        )
        rows.append(
            {
                "sample_id": cached["sample_id"],
                "sample_index": cached["sample_index"],
                "variant": "tileplandet_clip_verifier",
                "metrics": metrics,
                "metadata": {
                    "method": "tileplandet_clip_proposal_verifier",
                    "seconds": round(total_seconds, 4),
                    "detector_seconds": cached["tile_metadata"].get("seconds", 0.0),
                    "verifier_seconds": cached["verifier_seconds"],
                    "object_threshold": object_threshold,
                    "opening_threshold": opening_threshold,
                    "counts": {
                        "walls": len(payload["walls"]),
                        "objects": len(payload["objects"]),
                        "openings": len(payload["openings"]),
                    },
                },
                "counts": {
                    "pred": {key: len(payload.get(key, [])) for key in ("walls", "openings", "objects")},
                    "gt": {key: len(cached["ground_truth"].get(key, [])) for key in ("walls", "openings", "objects")},
                },
            }
        )
    return rows


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    assert_gpu_7()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to run verifier outside GPU 7.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verifier, verifier_summary = _load_verifier(args.verifier)
    start = time.perf_counter()
    tune = None
    object_threshold = args.object_threshold
    opening_threshold = args.opening_threshold
    if args.tune_manifest:
        tune_rows = _build_cached_rows(
            manifest_path=args.tune_manifest,
            output_dir=args.output_dir / "tune_cache",
            weights=args.weights,
            verifier=verifier,
            tile_confidence=args.tile_confidence,
            tile_stride=args.tile_stride,
            image_size=args.image_size,
            batch_size=args.batch_size,
        )
        tune = _tune_thresholds(tune_rows)
        object_threshold = float(tune["object_threshold"])
        opening_threshold = float(tune["opening_threshold"])
        (args.output_dir / "threshold_tuning.json").write_text(
            json.dumps(tune, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    cached_rows = _build_cached_rows(
        manifest_path=args.manifest,
        output_dir=args.output_dir / "eval_cache",
        weights=args.weights,
        verifier=verifier,
        tile_confidence=args.tile_confidence,
        tile_stride=args.tile_stride,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    result_rows = _result_rows(
        cached_rows,
        object_threshold=object_threshold,
        opening_threshold=opening_threshold,
    )
    detail_path = args.output_dir / "eval_results.jsonl"
    detail_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in result_rows) + "\n",
        encoding="utf-8",
    )
    variants = sorted({row["variant"] for row in result_rows})
    summary = {
        "manifest_path": str(args.manifest),
        "samples": len(cached_rows),
        "detail_path": str(detail_path),
        "seconds": round(time.perf_counter() - start, 4),
        "gpu_policy": gpu_policy(),
        "method": "TilePlanDet proposals filtered by frozen-CLIP hard-negative verifier",
        "weights": args.weights,
        "tile_confidence": args.tile_confidence,
        "tile_stride": args.tile_stride,
        "verifier": str(args.verifier),
        "verifier_training": verifier_summary,
        "thresholds": {
            "object": object_threshold,
            "opening": opening_threshold,
            "tuned_on": str(args.tune_manifest) if args.tune_manifest else None,
            "tuning": tune,
        },
        "aggregate": [_aggregate(result_rows, variant) for variant in variants],
    }
    (args.output_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tune-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--tile-confidence", type=float, default=0.03)
    parser.add_argument("--tile-stride", type=int, default=512)
    parser.add_argument("--object-threshold", type=float, default=0.5)
    parser.add_argument("--opening-threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
