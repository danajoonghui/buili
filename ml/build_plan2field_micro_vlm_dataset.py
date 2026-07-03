from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.buili.spatial.micro_vlm import MICRO_VLM_CLASSES
from services.api.buili.spatial.semantic_scene import build_maricopa_source_aligned_scene


TASKS = {
    "detect_plan_elements": 0,
    "detect_openings": 1,
    "detect_mep_fixtures": 2,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_scene_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    crop = Path(str(payload.get("source_crop_png", "")))
    if not crop.is_absolute():
        crop = REPO_ROOT / crop
    if not crop.exists():
        return None
    payload["source_crop_png"] = str(crop)
    return payload


def collect_scene_payloads() -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    manual_scene = build_maricopa_source_aligned_scene().to_json()
    manual_crop = REPO_ROOT / str(manual_scene["source_crop_png"])
    if manual_crop.exists():
        manual_scene["source_crop_png"] = str(manual_crop)
        scenes.append(manual_scene)
    seen = {str(manual_crop)}
    for scene_path in sorted(REPO_ROOT.glob("docs/plan2field3d_house_conversion/**/auto_semantic_scene.json")):
        payload = read_scene_json(scene_path)
        if not payload:
            continue
        crop = str(payload["source_crop_png"])
        if crop in seen:
            continue
        seen.add(crop)
        scenes.append(payload)
    return scenes


def scene_targets(scene: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for obj in scene.get("objects") or []:
        label = str(obj.get("kind", ""))
        if label not in MICRO_VLM_CLASSES or label == "background":
            continue
        cx, cy = [float(v) for v in obj["center_px"]]
        width = max(18.0, float(obj.get("width_px") or 28.0))
        height = max(18.0, float(obj.get("depth_px") or 28.0))
        targets.append(
            {
                "label": label,
                "center_px": [cx, cy],
                "width_px": width,
                "height_px": height,
                "task": "detect_mep_fixtures" if label not in {"door", "window"} else "detect_plan_elements",
                "source": obj.get("source_note", "semantic_scene_object"),
            }
        )
    for opening in scene.get("openings") or []:
        label = str(opening.get("kind", ""))
        if label not in {"door", "window"}:
            continue
        cx, cy = [float(v) for v in opening["center_px"]]
        length = max(24.0, float(opening.get("length_px") or 36.0))
        width = length if abs(float(opening.get("angle_deg") or 0.0)) < 45 else 18.0
        height = 18.0 if width == length else length
        targets.append(
            {
                "label": label,
                "center_px": [cx, cy],
                "width_px": width,
                "height_px": height,
                "task": "detect_openings",
                "source": opening.get("source_note", "semantic_scene_opening"),
            }
        )
    return targets


def patch_box_for_target(
    image_width: int,
    image_height: int,
    target: dict[str, Any],
    *,
    patch_size: int,
    jitter_px: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    cx, cy = [float(v) for v in target["center_px"]]
    cx += rng.uniform(-jitter_px, jitter_px)
    cy += rng.uniform(-jitter_px, jitter_px)
    x0 = int(round(cx - patch_size / 2))
    y0 = int(round(cy - patch_size / 2))
    x0 = max(0, min(x0, max(0, image_width - patch_size)))
    y0 = max(0, min(y0, max(0, image_height - patch_size)))
    return x0, y0, min(image_width, x0 + patch_size), min(image_height, y0 + patch_size)


def normalized_bbox(target: dict[str, Any], patch: tuple[int, int, int, int]) -> list[float]:
    x0, y0, x1, y1 = patch
    patch_w = max(x1 - x0, 1)
    patch_h = max(y1 - y0, 1)
    cx, cy = [float(v) for v in target["center_px"]]
    width = float(target["width_px"])
    height = float(target["height_px"])
    return [
        min(max((cx - x0) / patch_w, 0.0), 1.0),
        min(max((cy - y0) / patch_h, 0.0), 1.0),
        min(max(width / patch_w, 0.02), 1.0),
        min(max(height / patch_h, 0.02), 1.0),
    ]


def overlaps_target(
    patch: tuple[int, int, int, int],
    targets: list[dict[str, Any]],
    *,
    margin_px: float = 24.0,
) -> bool:
    x0, y0, x1, y1 = patch
    for target in targets:
        cx, cy = [float(v) for v in target["center_px"]]
        if x0 - margin_px <= cx <= x1 + margin_px and y0 - margin_px <= cy <= y1 + margin_px:
            return True
    return False


def save_patch(image: Image.Image, patch: tuple[int, int, int, int], path: Path) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    crop = image.crop(patch)
    crop.save(path, quality=88)
    return {"width": crop.width, "height": crop.height}


def build_dataset(out_dir: Path, *, patch_size: int, negatives_per_scene: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    image_dir = out_dir / "images"
    rows: list[dict[str, Any]] = []
    scenes = collect_scene_payloads()
    for scene_index, scene in enumerate(scenes):
        crop_path = Path(str(scene["source_crop_png"]))
        image = Image.open(crop_path).convert("RGB")
        targets = scene_targets(scene)
        if not targets:
            continue
        for target_index, target in enumerate(targets):
            for aug_index, jitter in enumerate([0, 18, 36]):
                patch = patch_box_for_target(
                    image.width,
                    image.height,
                    target,
                    patch_size=patch_size,
                    jitter_px=jitter,
                    rng=rng,
                )
                patch_path = image_dir / f"scene{scene_index:03d}_target{target_index:03d}_aug{aug_index}.jpg"
                size = save_patch(image, patch, patch_path)
                rows.append(
                    {
                        "id": f"scene{scene_index:03d}_target{target_index:03d}_aug{aug_index}",
                        "image": str(patch_path),
                        "task": target["task"],
                        "task_id": TASKS[target["task"]],
                        "prompt": (
                            "Parse this construction drawing patch and output the primary "
                            "plan element class and source-relative box."
                        ),
                        "label": target["label"],
                        "class_id": MICRO_VLM_CLASSES.index(target["label"]),
                        "bbox": normalized_bbox(target, patch),
                        "source_scene": str(crop_path),
                        "source_note": target["source"],
                        **size,
                    }
                )
        negative_budget = max(negatives_per_scene, len(targets) // 2)
        attempts = 0
        added = 0
        while added < negative_budget and attempts < negative_budget * 20:
            attempts += 1
            if image.width <= patch_size:
                x0 = 0
            else:
                x0 = rng.randint(0, image.width - patch_size)
            if image.height <= patch_size:
                y0 = 0
            else:
                y0 = rng.randint(0, image.height - patch_size)
            patch = (x0, y0, min(image.width, x0 + patch_size), min(image.height, y0 + patch_size))
            if overlaps_target(patch, targets):
                continue
            patch_path = image_dir / f"scene{scene_index:03d}_negative{added:03d}.jpg"
            size = save_patch(image, patch, patch_path)
            rows.append(
                {
                    "id": f"scene{scene_index:03d}_negative{added:03d}",
                    "image": str(patch_path),
                    "task": "detect_plan_elements",
                    "task_id": TASKS["detect_plan_elements"],
                    "prompt": "Parse this construction drawing patch. It may contain no tracked element.",
                    "label": "background",
                    "class_id": 0,
                    "bbox": [0.5, 0.5, 0.1, 0.1],
                    "source_scene": str(crop_path),
                    "source_note": "negative_patch",
                    **size,
                }
            )
            added += 1
    rng.shuffle(rows)
    split_at = max(1, int(len(rows) * 0.86))
    for index, row in enumerate(rows):
        row["split"] = "train" if index < split_at else "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
    manifest = {
        "dataset": "plan2field_micro_vlm_patch_dataset",
        "rows": len(rows),
        "train_rows": sum(1 for row in rows if row["split"] == "train"),
        "eval_rows": sum(1 for row in rows if row["split"] == "eval"),
        "classes": MICRO_VLM_CLASSES,
        "tasks": TASKS,
        "patch_size": patch_size,
        "scenes": len(scenes),
        "sha256": sha256_file(dataset_path),
        "source": (
            "Generated from public floor-plan crops and Buili semantic scene labels; "
            "intended for lightweight VLM plan patch parsing."
        ),
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/plan2field_micro_vlm"))
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--negatives-per-scene", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.out_dir, patch_size=args.patch_size, negatives_per_scene=args.negatives_per_scene, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
