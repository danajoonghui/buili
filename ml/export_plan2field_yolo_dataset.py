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


YOLO_CLASSES = [
    "wall",
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


def _yolo_line(cls_id: int, bbox: tuple[float, float, float, float], width: int, height: int) -> str:
    x0, y0, x1, y1 = bbox
    cx = ((x0 + x1) / 2) / width
    cy = ((y0 + y1) / 2) / height
    bw = (x1 - x0) / width
    bh = (y1 - y0) / height
    return f"{cls_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def export_yolo_dataset(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images" / "val"
    labels_dir = output_dir / "labels" / "val"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_manifest(manifest_path)
    class_to_id = {label: index for index, label in enumerate(YOLO_CLASSES)}
    exported = 0
    boxes = 0
    for row in rows:
        image_path = Path(row["image_path"])
        gt = json.loads(Path(row["ground_truth_path"]).read_text(encoding="utf-8"))
        image = Image.open(image_path)
        width, height = image.size
        stem = f"{int(row['sample_index']):03d}_{row['sample_id']}"
        target_image = images_dir / f"{stem}{image_path.suffix.lower()}"
        target_label = labels_dir / f"{stem}.txt"
        if not target_image.exists():
            shutil.copy2(image_path, target_image)
        lines: list[str] = []
        for wall in gt.get("walls", []):
            bbox = _clip_bbox(wall["bbox"], width, height)
            if bbox:
                lines.append(_yolo_line(class_to_id["wall"], bbox, width, height))
        for opening in gt.get("openings", []):
            kind = opening["kind"]
            if kind not in class_to_id:
                continue
            bbox = _clip_bbox(opening["bbox"], width, height)
            if bbox:
                lines.append(_yolo_line(class_to_id[kind], bbox, width, height))
        for obj in gt.get("objects", []):
            kind = obj["kind"] if obj["kind"] in class_to_id else "fixture"
            bbox = _clip_bbox(obj["bbox"], width, height)
            if bbox:
                lines.append(_yolo_line(class_to_id[kind], bbox, width, height))
        target_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        exported += 1
        boxes += len(lines)
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/val",
                "val: images/val",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(YOLO_CLASSES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "data_yaml": str(yaml_path),
        "samples": exported,
        "boxes": boxes,
        "classes": YOLO_CLASSES,
        "command_example": (
            f"CUDA_VISIBLE_DEVICES=7 yolo detect train model=yolo11n.pt "
            f"data={yaml_path} epochs=20 imgsz=1024"
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/eval/plan2field_cubicasa50/manifest.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/plan2field_yolo_eval50"))
    args = parser.parse_args()
    print(json.dumps(export_yolo_dataset(args.manifest, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
