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

from ml.build_plan2field_yolo_tiles import _clip_box_to_tile, _gt_items, _tile_origins, _yolo_line
from ml.export_plan2field_yolo_dataset import YOLO_CLASSES


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_split(
    rows: list[dict[str, Any]],
    output_dir: Path,
    split: str,
    *,
    tile: int,
    stride: int,
    include_walls: bool,
    min_visible_ratio: float,
    negative_stride_factor: int,
) -> dict[str, int]:
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    class_to_id = {name: index for index, name in enumerate(YOLO_CLASSES)}
    tile_count = 0
    positive_tiles = 0
    negative_tiles = 0
    box_count = 0
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
            if not labels and negative_stride_factor > 1 and tile_index % negative_stride_factor != 0:
                continue
            stem = f"{int(row['sample_index']):04d}_{row['sample_id']}_t{tile_index:03d}_{ox}_{oy}"
            tile_path = images_dir / f"{stem}.png"
            label_path = labels_dir / f"{stem}.txt"
            if not tile_path.exists():
                image.crop((ox, oy, ox + tile, oy + tile)).save(tile_path)
            label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            tile_count += 1
            box_count += len(labels)
            if labels:
                positive_tiles += 1
            else:
                negative_tiles += 1
    return {
        "tiles": tile_count,
        "positive_tiles": positive_tiles,
        "negative_tiles": negative_tiles,
        "boxes": box_count,
    }


def build_dataset(
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
    *,
    tile: int,
    stride: int,
    include_walls: bool,
    min_visible_ratio: float,
    negative_stride_factor: int,
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _write_split(
        _load_manifest(train_manifest),
        output_dir,
        "train",
        tile=tile,
        stride=stride,
        include_walls=include_walls,
        min_visible_ratio=min_visible_ratio,
        negative_stride_factor=negative_stride_factor,
    )
    val = _write_split(
        _load_manifest(val_manifest),
        output_dir,
        "val",
        tile=tile,
        stride=stride,
        include_walls=include_walls,
        min_visible_ratio=min_visible_ratio,
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
                *[f"  {index}: {name}" for index, name in enumerate(YOLO_CLASSES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "method": "TilePlanDet background-aware tile dataset",
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "output_dir": str(output_dir),
        "data_yaml": str(yaml_path),
        "tile": tile,
        "stride": stride,
        "include_walls": include_walls,
        "min_visible_ratio": min_visible_ratio,
        "negative_stride_factor": negative_stride_factor,
        "classes": YOLO_CLASSES,
        "train": train,
        "val": val,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=768)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--include-walls", action="store_true")
    parser.add_argument("--min-visible-ratio", type=float, default=0.85)
    parser.add_argument(
        "--negative-stride-factor",
        type=int,
        default=1,
        help="Keep every Nth negative tile in train. 1 keeps all negatives.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_dataset(
                args.train_manifest,
                args.val_manifest,
                args.output_dir,
                tile=args.tile,
                stride=args.stride,
                include_walls=args.include_walls,
                min_visible_ratio=args.min_visible_ratio,
                negative_stride_factor=max(1, args.negative_stride_factor),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
