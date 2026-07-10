from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CLASSES = [
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


def _tile_origins(width: int, height: int, tile: int, stride: int) -> list[tuple[int, int]]:
    xs = list(range(0, max(1, width - tile + 1), stride))
    ys = list(range(0, max(1, height - tile + 1), stride))
    if not xs or xs[-1] + tile < width:
        xs.append(max(0, width - tile))
    if not ys or ys[-1] + tile < height:
        ys.append(max(0, height - tile))
    return [(x, y) for y in ys for x in xs]


def _bbox_dark_density(image: np.ndarray, bbox: list[float]) -> float:
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


def _clip_box_to_tile(
    bbox: list[float],
    origin: tuple[int, int],
    tile: int,
    *,
    min_visible_ratio: float,
) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    ox, oy = origin
    tx0, ty0, tx1, ty1 = ox, oy, ox + tile, oy + tile
    ix0, iy0 = max(x0, tx0), max(y0, ty0)
    ix1, iy1 = min(x1, tx1), min(y1, ty1)
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        return None
    original_area = max((x1 - x0) * (y1 - y0), 1.0)
    visible_area = (ix1 - ix0) * (iy1 - iy0)
    if visible_area / original_area < min_visible_ratio:
        return None
    return ix0 - ox, iy0 - oy, ix1 - ox, iy1 - oy


def _yolo_line(cls_id: int, bbox: tuple[float, float, float, float], tile: int) -> str:
    x0, y0, x1, y1 = bbox
    return (
        f"{cls_id} {((x0 + x1) / 2) / tile:.8f} {((y0 + y1) / 2) / tile:.8f} "
        f"{(x1 - x0) / tile:.8f} {(y1 - y0) / tile:.8f}"
    )


def _source_items(
    gt: dict[str, Any],
    *,
    selected_classes: set[str],
) -> list[tuple[str, list[float]]]:
    items: list[tuple[str, list[float]]] = []
    if "wall" in selected_classes:
        for wall in gt.get("walls", []):
            items.append(("wall", wall["bbox"]))
    for opening in gt.get("openings", []):
        kind = str(opening.get("kind", ""))
        if kind in selected_classes:
            items.append((kind, opening["bbox"]))
    for obj in gt.get("objects", []):
        kind = str(obj.get("kind", "fixture"))
        if kind not in DEFAULT_CLASSES:
            kind = "fixture"
        if kind in selected_classes:
            items.append((kind, obj["bbox"]))
    return items


def _write_split(
    rows: list[dict[str, Any]],
    output_dir: Path,
    split: str,
    *,
    classes: list[str],
    tile: int,
    stride: int,
    min_visible_ratio: float,
    min_dark_density: float,
    negative_stride_factor: int,
) -> dict[str, Any]:
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    class_to_id = {name: index for index, name in enumerate(classes)}
    selected = set(classes)
    tile_count = 0
    positive_tiles = 0
    negative_tiles = 0
    box_count = 0
    filtered_low_dark = {name: 0 for name in classes}
    class_counts = {name: 0 for name in classes}
    for row in rows:
        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image)
        gt = json.loads(Path(row["ground_truth_path"]).read_text(encoding="utf-8"))
        items = []
        for kind, bbox in _source_items(gt, selected_classes=selected):
            if min_dark_density > 0 and _bbox_dark_density(image_array, bbox) < min_dark_density:
                filtered_low_dark[kind] += 1
                continue
            items.append((kind, bbox))
        for tile_index, origin in enumerate(_tile_origins(image.width, image.height, tile, stride)):
            labels: list[str] = []
            ox, oy = origin
            for kind, bbox in items:
                clipped = _clip_box_to_tile(
                    bbox,
                    origin,
                    tile,
                    min_visible_ratio=min_visible_ratio,
                )
                if clipped is None:
                    continue
                labels.append(_yolo_line(class_to_id[kind], clipped, tile))
                class_counts[kind] += 1
            if not labels:
                if negative_stride_factor <= 0 or tile_index % negative_stride_factor != 0:
                    continue
            stem = f"{int(row['sample_index']):04d}_{row['sample_id']}_t{tile_index:03d}_{ox}_{oy}"
            tile_path = images_dir / f"{stem}.png"
            label_path = labels_dir / f"{stem}.txt"
            if not tile_path.exists():
                image.crop((ox, oy, ox + tile, oy + tile)).save(tile_path)
            label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            tile_count += 1
            if labels:
                positive_tiles += 1
            else:
                negative_tiles += 1
            box_count += len(labels)
    return {
        "tiles": tile_count,
        "positive_tiles": positive_tiles,
        "negative_tiles": negative_tiles,
        "boxes": box_count,
        "class_counts": class_counts,
        "filtered_low_dark": filtered_low_dark,
    }


def build_filtered_tiles(
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
    *,
    classes: list[str],
    tile: int,
    stride: int,
    min_visible_ratio: float,
    min_dark_density: float,
    negative_stride_factor: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _write_split(
        _load_manifest(train_manifest),
        output_dir,
        "train",
        classes=classes,
        tile=tile,
        stride=stride,
        min_visible_ratio=min_visible_ratio,
        min_dark_density=min_dark_density,
        negative_stride_factor=negative_stride_factor,
    )
    val = _write_split(
        _load_manifest(val_manifest),
        output_dir,
        "val",
        classes=classes,
        tile=tile,
        stride=stride,
        min_visible_ratio=min_visible_ratio,
        min_dark_density=min_dark_density,
        negative_stride_factor=1,
    )
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(classes)],
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
        "classes": classes,
        "tile": tile,
        "stride": stride,
        "min_visible_ratio": min_visible_ratio,
        "min_dark_density": min_dark_density,
        "negative_stride_factor": negative_stride_factor,
        "train": train,
        "val": val,
        "method": "Filtered specialist YOLO tile dataset with contiguous class remap",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classes", required=True, help="Comma-separated classes, e.g. door,window")
    parser.add_argument("--tile", type=int, default=768)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--min-visible-ratio", type=float, default=0.85)
    parser.add_argument("--min-dark-density", type=float, default=0.012)
    parser.add_argument(
        "--negative-stride-factor",
        type=int,
        default=0,
        help="Keep every Nth negative train tile. 0 keeps no negative train tiles. Val keeps all negatives.",
    )
    args = parser.parse_args()
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    print(
        json.dumps(
            build_filtered_tiles(
                args.train_manifest,
                args.val_manifest,
                args.output_dir,
                classes=classes,
                tile=args.tile,
                stride=args.stride,
                min_visible_ratio=args.min_visible_ratio,
                min_dark_density=args.min_dark_density,
                negative_stride_factor=max(0, args.negative_stride_factor),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
