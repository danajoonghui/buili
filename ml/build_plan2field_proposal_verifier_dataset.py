from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
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

from services.api.buili.gpu import force_gpu_7, gpu_policy

force_gpu_7()

from ml.evaluate_plan2field3d_eval50 import tiled_yolo_vectorsnap
from services.api.buili.spatial.eval_metrics import iou_bbox


PROPOSAL_CLASSES = [
    "door",
    "window",
    "bathtub",
    "cabinet_run",
    "column",
    "fixture",
    "shower",
    "sink",
    "toilet",
    "water_heater",
]
OPENING_CLASSES = {"door", "window"}


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _clamp_box(
    bbox: list[float],
    *,
    width: int,
    height: int,
    pad_ratio: float,
    min_size: int,
) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    if x1 <= x0 or y1 <= y0:
        return None
    pad = max((x1 - x0), (y1 - y0), float(min_size)) * pad_ratio
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    side_w = max(x1 - x0 + 2 * pad, float(min_size))
    side_h = max(y1 - y0 + 2 * pad, float(min_size))
    ix0 = max(0, int(math.floor(cx - side_w / 2)))
    iy0 = max(0, int(math.floor(cy - side_h / 2)))
    ix1 = min(width, int(math.ceil(cx + side_w / 2)))
    iy1 = min(height, int(math.ceil(cy + side_h / 2)))
    if ix1 - ix0 < 4 or iy1 - iy0 < 4:
        return None
    return ix0, iy0, ix1, iy1


def _dark_density(image: Image.Image, bbox: tuple[int, int, int, int]) -> float:
    crop = image.crop(bbox).convert("L")
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr < 150))


def _match_proposal(
    proposal: dict[str, Any],
    gt: dict[str, Any],
) -> dict[str, Any]:
    kind = str(proposal.get("kind", ""))
    group = "openings" if kind in OPENING_CLASSES else "objects"
    threshold = 0.35 if group == "openings" else 0.50
    best_iou = 0.0
    best_id = ""
    for target in gt.get(group, []):
        if str(target.get("kind")) != kind:
            continue
        score = iou_bbox(tuple(proposal["bbox"]), tuple(target["bbox"]))
        if score > best_iou:
            best_iou = float(score)
            best_id = str(target.get("id", ""))
    return {
        "target_group": group,
        "target_id": best_id,
        "max_iou": round(best_iou, 6),
        "keep": bool(best_iou >= threshold),
        "iou_threshold": threshold,
    }


def _write_split(
    *,
    rows: list[dict[str, Any]],
    split: str,
    output_dir: Path,
    weights: str,
    confidence: float,
    stride: int,
    pad_ratio: float,
    min_crop_size: int,
    max_negatives_per_image: int,
) -> dict[str, Any]:
    crop_dir = output_dir / "images" / split
    crop_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split}.jsonl"
    class_to_id = {name: index for index, name in enumerate(PROPOSAL_CLASSES)}
    written = 0
    positives = 0
    negatives = 0
    raw_proposals = 0
    start = time.perf_counter()

    with out_path.open("w", encoding="utf-8") as fh:
        for sample_index, sample in enumerate(rows):
            image_path = Path(sample["image_path"])
            gt = json.loads(Path(sample["ground_truth_path"]).read_text(encoding="utf-8"))
            image = Image.open(image_path).convert("RGB")
            payload, metadata = tiled_yolo_vectorsnap(
                image_path,
                weights=weights,
                confidence=confidence,
                stride=stride,
                wall_payload=[],
                method_name="tileplandet_proposal_dataset",
            )
            proposals = [*payload.get("openings", []), *payload.get("objects", [])]
            raw_proposals += int(metadata.get("raw_detections", len(proposals)))
            positive_rows: list[dict[str, Any]] = []
            negative_rows: list[dict[str, Any]] = []
            for proposal_index, proposal in enumerate(proposals):
                kind = str(proposal.get("kind", ""))
                if kind not in class_to_id:
                    continue
                crop_box = _clamp_box(
                    proposal["bbox"],
                    width=image.width,
                    height=image.height,
                    pad_ratio=pad_ratio,
                    min_size=min_crop_size,
                )
                if crop_box is None:
                    continue
                x0, y0, x1, y1 = [float(value) for value in proposal["bbox"]]
                width_px = max(x1 - x0, 1.0)
                height_px = max(y1 - y0, 1.0)
                match = _match_proposal(proposal, gt)
                crop_name = (
                    f"{sample_index:04d}_{sample['sample_id']}_{split}_"
                    f"{proposal_index:04d}_{kind}.png"
                )
                row = {
                    "id": crop_name.removesuffix(".png"),
                    "split": split,
                    "sample_id": str(sample["sample_id"]),
                    "source_image": str(image_path),
                    "ground_truth_path": str(sample["ground_truth_path"]),
                    "image": str(crop_dir / crop_name),
                    "kind": kind,
                    "class_id": class_to_id[kind],
                    "bbox": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
                    "crop_box": list(crop_box),
                    "score": float(proposal.get("score", 0.0)),
                    "source_class": str(proposal.get("source_class", kind)),
                    "area_fraction": float((width_px * height_px) / max(image.width * image.height, 1)),
                    "aspect_log": float(math.log(width_px / height_px)),
                    "dark_density": _dark_density(image, crop_box),
                    **match,
                }
                if match["keep"]:
                    positive_rows.append(row)
                else:
                    negative_rows.append(row)

            if max_negatives_per_image > 0 and len(negative_rows) > max_negatives_per_image:
                negative_rows = sorted(
                    negative_rows,
                    key=lambda item: (
                        -float(item["score"]),
                        -float(item["dark_density"]),
                        -float(item["max_iou"]),
                    ),
                )[:max_negatives_per_image]
            for row in [*positive_rows, *negative_rows]:
                crop_path = Path(row["image"])
                if not crop_path.exists():
                    image.crop(tuple(row["crop_box"])).save(crop_path)
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                written += 1
                positives += int(row["keep"])
                negatives += int(not row["keep"])

    return {
        "split": split,
        "samples": len(rows),
        "rows": written,
        "positives": positives,
        "negatives": negatives,
        "positive_ratio": round(positives / max(written, 1), 6),
        "raw_proposals": raw_proposals,
        "seconds": round(time.perf_counter() - start, 4),
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _load_manifest(args.train_manifest)
    val_rows = _load_manifest(args.val_manifest)
    train = _write_split(
        rows=train_rows,
        split="train",
        output_dir=args.output_dir,
        weights=args.weights,
        confidence=args.confidence,
        stride=args.stride,
        pad_ratio=args.pad_ratio,
        min_crop_size=args.min_crop_size,
        max_negatives_per_image=args.max_negatives_per_image,
    )
    val = _write_split(
        rows=val_rows,
        split="val",
        output_dir=args.output_dir,
        weights=args.weights,
        confidence=args.confidence,
        stride=args.stride,
        pad_ratio=args.pad_ratio,
        min_crop_size=args.min_crop_size,
        max_negatives_per_image=0,
    )
    summary = {
        "method": "TilePlanDet hard-negative proposal verifier dataset",
        "output_dir": str(args.output_dir),
        "weights": args.weights,
        "confidence": args.confidence,
        "stride": args.stride,
        "classes": PROPOSAL_CLASSES,
        "train": train,
        "val": val,
        "gpu": gpu_policy(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--confidence", type=float, default=0.03)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--pad-ratio", type=float, default=0.65)
    parser.add_argument("--min-crop-size", type=int, default=96)
    parser.add_argument("--max-negatives-per-image", type=int, default=180)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
