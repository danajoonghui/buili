from __future__ import annotations

import argparse
import json
import math
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


def _tile_origins(width: int, height: int, tile: int, stride: int) -> list[tuple[int, int]]:
    xs = list(range(0, max(1, width - tile + 1), stride))
    ys = list(range(0, max(1, height - tile + 1), stride))
    if not xs or xs[-1] + tile < width:
        xs.append(max(0, width - tile))
    if not ys or ys[-1] + tile < height:
        ys.append(max(0, height - tile))
    return [(x, y) for y in ys for x in xs]


def _clip_box_to_tile(
    bbox: list[float],
    origin: tuple[int, int],
    tile: int,
    *,
    min_visible_ratio: float,
) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = [float(v) for v in bbox]
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


def _gt_items(gt: dict[str, Any], *, include_walls: bool) -> list[tuple[str, list[float]]]:
    items: list[tuple[str, list[float]]] = []
    if include_walls:
        for wall in gt.get("walls", []):
            items.append(("wall", wall["bbox"]))
    for opening in gt.get("openings", []):
        if opening.get("kind") in {"door", "window"}:
            items.append((opening["kind"], opening["bbox"]))
    for obj in gt.get("objects", []):
        kind = obj.get("kind", "fixture")
        items.append((kind if kind in YOLO_CLASSES else "fixture", obj["bbox"]))
    return items


def _write_split(
    rows: list[dict[str, Any]],
    output_dir: Path,
    split: str,
    *,
    tile: int,
    stride: int,
    include_walls: bool,
    min_visible_ratio: float,
) -> dict[str, int]:
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    class_to_id = {name: index for index, name in enumerate(YOLO_CLASSES)}
    tile_count = 0
    box_count = 0
    positive_tiles = 0
    for row in rows:
        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        gt = json.loads(Path(row["ground_truth_path"]).read_text(encoding="utf-8"))
        items = _gt_items(gt, include_walls=include_walls)
        for tile_index, origin in enumerate(_tile_origins(image.width, image.height, tile, stride)):
            ox, oy = origin
            labels: list[str] = []
            for kind, bbox in items:
                clipped = _clip_box_to_tile(
                    bbox,
                    origin,
                    tile,
                    min_visible_ratio=min_visible_ratio,
                )
                if clipped is None:
                    continue
                labels.append(_yolo_line(class_to_id.get(kind, class_to_id["fixture"]), clipped, tile))
            if not labels:
                continue
            stem = f"{int(row['sample_index']):04d}_{row['sample_id']}_t{tile_index:03d}_{ox}_{oy}"
            tile_path = images_dir / f"{stem}.png"
            label_path = labels_dir / f"{stem}.txt"
            if not tile_path.exists():
                image.crop((ox, oy, ox + tile, oy + tile)).save(tile_path)
            label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
            tile_count += 1
            positive_tiles += 1
            box_count += len(labels)
    return {"tiles": tile_count, "positive_tiles": positive_tiles, "boxes": box_count}


def build_tiled_yolo_dataset(
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
    *,
    tile: int = 768,
    stride: int = 512,
    include_walls: bool = False,
    min_visible_ratio: float = 0.35,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _write_split(
        _load_manifest(train_manifest),
        output_dir,
        "train",
        tile=tile,
        stride=stride,
        include_walls=include_walls,
        min_visible_ratio=min_visible_ratio,
    )
    val = _write_split(
        _load_manifest(val_manifest),
        output_dir,
        "val",
        tile=tile,
        stride=stride,
        include_walls=include_walls,
        min_visible_ratio=min_visible_ratio,
    )
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
        "tile": tile,
        "stride": stride,
        "include_walls": include_walls,
        "min_visible_ratio": min_visible_ratio,
        "classes": YOLO_CLASSES,
        "train": train,
        "val": val,
        "method": "TilePlanDet: high-resolution tile-aware floorplan detector data",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=768)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--include-walls", action="store_true")
    parser.add_argument("--min-visible-ratio", type=float, default=0.35)
    args = parser.parse_args()
    print(
        json.dumps(
            build_tiled_yolo_dataset(
                args.train_manifest,
                args.val_manifest,
                args.output_dir,
                tile=args.tile,
                stride=args.stride,
                include_walls=args.include_walls,
                min_visible_ratio=args.min_visible_ratio,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
