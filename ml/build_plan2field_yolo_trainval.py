from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.export_plan2field_yolo_dataset import YOLO_CLASSES


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _clip_bbox(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x0 = max(0.0, min(float(width), float(bbox[0])))
    y0 = max(0.0, min(float(height), float(bbox[1])))
    x1 = max(0.0, min(float(width), float(bbox[2])))
    y1 = max(0.0, min(float(height), float(bbox[3])))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def _line(cls_id: int, bbox: tuple[float, float, float, float], width: int, height: int) -> str:
    x0, y0, x1, y1 = bbox
    return (
        f"{cls_id} {((x0 + x1) / 2) / width:.8f} {((y0 + y1) / 2) / height:.8f} "
        f"{(x1 - x0) / width:.8f} {(y1 - y0) / height:.8f}"
    )


def _write_split(
    rows: list[dict[str, Any]],
    output_dir: Path,
    split: str,
    *,
    include_walls: bool,
) -> dict[str, int]:
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    class_to_id = {name: index for index, name in enumerate(YOLO_CLASSES)}
    sample_count = 0
    box_count = 0
    for row in rows:
        image_path = Path(row["image_path"])
        gt = json.loads(Path(row["ground_truth_path"]).read_text(encoding="utf-8"))
        image = Image.open(image_path)
        width, height = image.size
        stem = f"{int(row['sample_index']):04d}_{row['sample_id']}"
        target_image = images_dir / f"{stem}{image_path.suffix.lower()}"
        target_label = labels_dir / f"{stem}.txt"
        if not target_image.exists():
            shutil.copy2(image_path, target_image)
        labels: list[str] = []
        if include_walls:
            for wall in gt.get("walls", []):
                bbox = _clip_bbox(wall["bbox"], width, height)
                if bbox:
                    labels.append(_line(class_to_id["wall"], bbox, width, height))
        for opening in gt.get("openings", []):
            kind = opening.get("kind", "")
            bbox = _clip_bbox(opening["bbox"], width, height)
            if bbox and kind in class_to_id:
                labels.append(_line(class_to_id[kind], bbox, width, height))
        for obj in gt.get("objects", []):
            kind = obj.get("kind", "fixture")
            bbox = _clip_bbox(obj["bbox"], width, height)
            if bbox:
                labels.append(_line(class_to_id.get(kind, class_to_id["fixture"]), bbox, width, height))
        target_label.write_text("\n".join(labels) + "\n", encoding="utf-8")
        sample_count += 1
        box_count += len(labels)
    return {"samples": sample_count, "boxes": box_count}


def build_yolo_trainval(
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
    *,
    include_walls: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _write_split(_load_manifest(train_manifest), output_dir, "train", include_walls=include_walls)
    val = _write_split(_load_manifest(val_manifest), output_dir, "val", include_walls=include_walls)
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(YOLO_CLASSES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "output_dir": str(output_dir),
        "data_yaml": str(yaml_path),
        "classes": YOLO_CLASSES,
        "train": train,
        "val": val,
        "method": "TinyPlanDet training data from CubiCasa SVG boxes",
        "include_walls": include_walls,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("data/train/plan2field_cubicasa_train500/manifest.jsonl"),
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path("data/eval/plan2field_cubicasa50/manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/plan2field_yolo_train500_eval50"),
    )
    parser.add_argument("--exclude-walls", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_yolo_trainval(
                args.train_manifest,
                args.val_manifest,
                args.output_dir,
                include_walls=not args.exclude_walls,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
