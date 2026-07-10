from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.buili.spatial.micro_vlm import MICRO_VLM_CLASSES, OBJECT_CLASSES, OPENING_CLASSES


TASKS = {
    "detect_plan_elements": 0,
    "detect_openings": 1,
    "detect_mep_fixtures": 2,
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_rows(gt: dict[str, Any]) -> tuple[list[dict[str, Any]], collections.Counter[str]]:
    targets: list[dict[str, Any]] = []
    skipped: collections.Counter[str] = collections.Counter()
    for opening in gt.get("openings", []):
        kind = str(opening.get("kind", ""))
        if kind not in OPENING_CLASSES:
            skipped[kind or "unknown_opening"] += 1
            continue
        x0, y0, x1, y1 = [float(v) for v in opening["bbox"]]
        targets.append(
            {
                "label": kind,
                "bbox": [x0, y0, x1, y1],
                "task": "detect_openings",
                "source_group": "openings",
            }
        )
    for obj in gt.get("objects", []):
        kind = str(obj.get("kind", ""))
        if kind not in OBJECT_CLASSES:
            skipped[kind or "unknown_object"] += 1
            continue
        x0, y0, x1, y1 = [float(v) for v in obj["bbox"]]
        targets.append(
            {
                "label": kind,
                "bbox": [x0, y0, x1, y1],
                "task": "detect_mep_fixtures",
                "source_group": "objects",
            }
        )
    return targets, skipped


def _centered_patch(
    bbox: list[float],
    image_width: int,
    image_height: int,
    *,
    patch_size: int,
    jitter_px: float,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2 + rng.uniform(-jitter_px, jitter_px)
    cy = (y0 + y1) / 2 + rng.uniform(-jitter_px, jitter_px)
    px0 = int(round(cx - patch_size / 2))
    py0 = int(round(cy - patch_size / 2))
    px0 = max(0, min(px0, max(0, image_width - patch_size)))
    py0 = max(0, min(py0, max(0, image_height - patch_size)))
    return px0, py0, min(image_width, px0 + patch_size), min(image_height, py0 + patch_size)


def _bbox_visible_ratio(bbox: list[float], patch: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    px0, py0, px1, py1 = patch
    ix0, iy0 = max(x0, px0), max(y0, py0)
    ix1, iy1 = min(x1, px1), min(y1, py1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / max((x1 - x0) * (y1 - y0), 1.0)


def _normalized_bbox(bbox: list[float], patch: tuple[int, int, int, int]) -> list[float]:
    x0, y0, x1, y1 = bbox
    px0, py0, px1, py1 = patch
    width = max(px1 - px0, 1)
    height = max(py1 - py0, 1)
    cx = ((x0 + x1) / 2 - px0) / width
    cy = ((y0 + y1) / 2 - py0) / height
    bw = (x1 - x0) / width
    bh = (y1 - y0) / height
    return [
        round(min(max(cx, 0.0), 1.0), 6),
        round(min(max(cy, 0.0), 1.0), 6),
        round(min(max(bw, 0.02), 1.0), 6),
        round(min(max(bh, 0.02), 1.0), 6),
    ]


def _target_dark_density(image: Image.Image, bbox: list[float]) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = gray.shape
    x0, y0, x1, y1 = [float(v) for v in bbox]
    ix0 = int(max(0, min(width - 1, round(x0))))
    iy0 = int(max(0, min(height - 1, round(y0))))
    ix1 = int(max(ix0 + 1, min(width, round(x1))))
    iy1 = int(max(iy0 + 1, min(height, round(y1))))
    crop = gray[iy0:iy1, ix0:ix1]
    return float((crop < 175).mean()) if crop.size else 0.0


def _patch_overlaps_targets(
    patch: tuple[int, int, int, int],
    targets: list[dict[str, Any]],
    *,
    margin_px: float,
) -> bool:
    px0, py0, px1, py1 = patch
    for target in targets:
        x0, y0, x1, y1 = [float(v) for v in target["bbox"]]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if px0 - margin_px <= cx <= px1 + margin_px and py0 - margin_px <= cy <= py1 + margin_px:
            return True
    return False


def _save_patch(image: Image.Image, patch: tuple[int, int, int, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(patch).save(path, quality=90)


def _write_overlay_sheet(rows: list[dict[str, Any]], output: Path, *, limit: int = 24) -> None:
    if not rows:
        return
    thumbs: list[Image.Image] = []
    for row in rows[:limit]:
        image = Image.open(row["image"]).convert("RGB").resize((192, 192))
        draw = ImageDraw.Draw(image)
        cx, cy, bw, bh = row["bbox"]
        x0 = (cx - bw / 2) * 192
        y0 = (cy - bh / 2) * 192
        x1 = (cx + bw / 2) * 192
        y1 = (cy + bh / 2) * 192
        color = (35, 117, 73) if row["label"] != "background" else (160, 160, 160)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.text((6, 6), str(row["label"])[:18], fill=color)
        thumbs.append(image)
    cols = 6
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 192, rows_count * 192), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 192, (index // cols) * 192))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def build_dataset(
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
    *,
    patch_size: int,
    train_jitter_px: list[int],
    val_jitter_px: list[int],
    negatives_per_image: int,
    min_visible_ratio: float,
    min_target_dark_density: float,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows_out: list[dict[str, Any]] = []
    skipped: collections.Counter[str] = collections.Counter()
    class_counts: collections.Counter[str] = collections.Counter()
    split_counts: collections.Counter[str] = collections.Counter()
    source_ids: dict[str, set[str]] = {"train": set(), "eval": set()}
    missing_files = 0
    degenerate_targets = 0
    filtered_low_density: collections.Counter[str] = collections.Counter()

    for split, manifest, jitters in (
        ("train", train_manifest, train_jitter_px),
        ("eval", val_manifest, val_jitter_px),
    ):
        for sample in _load_jsonl(manifest):
            image_path = Path(sample["image_path"])
            gt_path = Path(sample["ground_truth_path"])
            if not image_path.exists() or not gt_path.exists():
                missing_files += 1
                continue
            image = Image.open(image_path).convert("RGB")
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
            targets, skipped_counts = _target_rows(gt)
            skipped.update(skipped_counts)
            sample_id = str(sample["sample_id"])
            source_ids[split].add(sample_id)

            for target_index, target in enumerate(targets):
                x0, y0, x1, y1 = target["bbox"]
                if x1 <= x0 or y1 <= y0:
                    degenerate_targets += 1
                    continue
                target_density = _target_dark_density(image, target["bbox"])
                if target_density < min_target_dark_density:
                    filtered_low_density[str(target["label"])] += 1
                    continue
                for aug_index, jitter in enumerate(jitters):
                    patch = _centered_patch(
                        target["bbox"],
                        image.width,
                        image.height,
                        patch_size=patch_size,
                        jitter_px=float(jitter),
                        rng=rng,
                    )
                    if _bbox_visible_ratio(target["bbox"], patch) < min_visible_ratio:
                        continue
                    row_id = f"{split}_{sample_id}_target{target_index:04d}_aug{aug_index}"
                    patch_path = output_dir / "images" / split / f"{row_id}.jpg"
                    _save_patch(image, patch, patch_path)
                    label = str(target["label"])
                    rows_out.append(
                        {
                            "id": row_id,
                            "image": str(patch_path),
                            "split": split,
                            "source_sample_id": sample_id,
                            "source_image_path": str(image_path),
                            "source_group": target["source_group"],
                            "task": target["task"],
                            "task_id": TASKS[target["task"]],
                            "prompt": "Parse this floor-plan patch and emit one tracked plan element.",
                            "label": label,
                            "class_id": MICRO_VLM_CLASSES.index(label),
                            "bbox": _normalized_bbox(target["bbox"], patch),
                            "target_dark_density": round(target_density, 6),
                            "patch": list(patch),
                        }
                    )
                    class_counts[label] += 1
                    split_counts[split] += 1

            negatives_added = 0
            attempts = 0
            while negatives_added < negatives_per_image and attempts < negatives_per_image * 40:
                attempts += 1
                if image.width <= patch_size:
                    px0 = 0
                else:
                    px0 = rng.randint(0, image.width - patch_size)
                if image.height <= patch_size:
                    py0 = 0
                else:
                    py0 = rng.randint(0, image.height - patch_size)
                patch = (px0, py0, min(image.width, px0 + patch_size), min(image.height, py0 + patch_size))
                if _patch_overlaps_targets(patch, targets, margin_px=24):
                    continue
                row_id = f"{split}_{sample_id}_neg{negatives_added:03d}"
                patch_path = output_dir / "images" / split / f"{row_id}.jpg"
                _save_patch(image, patch, patch_path)
                rows_out.append(
                    {
                        "id": row_id,
                        "image": str(patch_path),
                        "split": split,
                        "source_sample_id": sample_id,
                        "source_image_path": str(image_path),
                        "source_group": "negative",
                        "task": "detect_plan_elements",
                        "task_id": TASKS["detect_plan_elements"],
                        "prompt": "Parse this floor-plan patch. It may contain no tracked element.",
                        "label": "background",
                        "class_id": MICRO_VLM_CLASSES.index("background"),
                        "bbox": [0.5, 0.5, 0.1, 0.1],
                        "patch": list(patch),
                    }
                )
                class_counts["background"] += 1
                split_counts[split] += 1
                negatives_added += 1

    rng.shuffle(rows_out)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows_out) + "\n",
        encoding="utf-8",
    )
    leakage = sorted(source_ids["train"] & source_ids["eval"])
    train_rows = [row for row in rows_out if row["split"] == "train"]
    eval_rows = [row for row in rows_out if row["split"] == "eval"]
    _write_overlay_sheet(train_rows, output_dir / "label_audit_train_sheet.jpg")
    _write_overlay_sheet(eval_rows, output_dir / "label_audit_eval_sheet.jpg")
    manifest = {
        "dataset": "cubicasa_vlm_primary_patch_dataset",
        "dataset_path": str(dataset_path),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "rows": len(rows_out),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "patch_size": patch_size,
        "train_jitter_px": train_jitter_px,
        "val_jitter_px": val_jitter_px,
        "negatives_per_image": negatives_per_image,
        "min_visible_ratio": min_visible_ratio,
        "min_target_dark_density": min_target_dark_density,
        "classes": MICRO_VLM_CLASSES,
        "class_counts": dict(sorted(class_counts.items())),
        "split_counts": dict(split_counts),
        "skipped_source_labels": dict(sorted(skipped.items())),
        "filtered_low_density_labels": dict(sorted(filtered_low_density.items())),
        "missing_files": missing_files,
        "degenerate_targets": degenerate_targets,
        "train_eval_source_overlap": leakage,
        "quality_gates": {
            "no_missing_files": missing_files == 0,
            "no_train_eval_leakage": not leakage,
            "no_degenerate_targets": degenerate_targets == 0,
            "dark_density_filter_enabled": min_target_dark_density > 0,
            "blind_eval50_excluded": "data/eval/plan2field_cubicasa50" not in str(train_manifest)
            and "data/eval/plan2field_cubicasa50" not in str(val_manifest),
        },
        "sha256": _sha256_file(dataset_path),
        "label_audit_train_sheet": str(output_dir / "label_audit_train_sheet.jpg"),
        "label_audit_eval_sheet": str(output_dir / "label_audit_eval_sheet.jpg"),
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--train-jitter-px", type=_parse_int_list, default=[0, 12, 24])
    parser.add_argument("--val-jitter-px", type=_parse_int_list, default=[0])
    parser.add_argument("--negatives-per-image", type=int, default=16)
    parser.add_argument("--min-visible-ratio", type=float, default=0.88)
    parser.add_argument("--min-target-dark-density", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1437)
    args = parser.parse_args()
    print(
        json.dumps(
            build_dataset(
                args.train_manifest,
                args.val_manifest,
                args.out_dir,
                patch_size=args.patch_size,
                train_jitter_px=args.train_jitter_px,
                val_jitter_px=args.val_jitter_px,
                negatives_per_image=args.negatives_per_image,
                min_visible_ratio=args.min_visible_ratio,
                min_target_dark_density=args.min_target_dark_density,
                seed=args.seed,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
