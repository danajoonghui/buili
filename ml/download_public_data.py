from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def _extract_doclaynet_record(record: dict) -> dict:
    labels: list[str] = []
    boxes: list[list[float]] = []
    metadata = record.get("metadata") or {}
    width = record.get("width") or metadata.get("page_width") or metadata.get("width") or 1
    height = record.get("height") or metadata.get("page_height") or metadata.get("height") or 1
    objects = record.get("objects") or record.get("annotations") or {}

    if "bboxes" in record and "category_id" in record:
        labels = [str(item) for item in record.get("category_id") or []]
        boxes = [[float(v) for v in box[:4]] for box in (record.get("bboxes") or []) if len(box) >= 4]
    elif isinstance(objects, dict):
        cats = objects.get("category") or objects.get("categories") or objects.get("label") or []
        bboxes = objects.get("bbox") or objects.get("bboxes") or []
        labels = [str(item) for item in cats]
        boxes = [[float(v) for v in box[:4]] for box in bboxes if len(box) >= 4]
    elif isinstance(objects, list):
        for obj in objects:
            if isinstance(obj, dict):
                labels.append(str(obj.get("category", obj.get("label", "unknown"))))
                bbox = obj.get("bbox") or obj.get("box") or [0, 0, width, height]
                boxes.append([float(v) for v in bbox[:4]])

    return {
        "source": "docling-project/DocLayNet-v1.1",
        "doc_category": str(
            record.get(
                "doc_category",
                record.get("document_category", metadata.get("doc_category", metadata.get("source", "unknown"))),
            )
        ),
        "page_hash": str(record.get("image_id", record.get("id", metadata.get("page_hash", "")))),
        "width": float(width),
        "height": float(height),
        "labels": labels[:256],
        "boxes": boxes[:256],
    }


def download_doclaynet(limit: int, out_path: Path) -> int:
    from datasets import Image, load_dataset

    dataset = load_dataset("docling-project/DocLayNet-v1.1", split="train", streaming=True)
    dataset = dataset.cast_column("image", Image(decode=False))
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for record in tqdm(dataset.take(limit), total=limit, desc="DocLayNet sample"):
            parsed = _extract_doclaynet_record(dict(record))
            if parsed["labels"]:
                fh.write(json.dumps(parsed, ensure_ascii=True) + "\n")
                count += 1
    return count


def write_fallback_public_sample(limit: int, out_path: Path) -> int:
    from sklearn.datasets import load_digits

    digits = load_digits()
    with out_path.open("w", encoding="utf-8") as fh:
        for idx in range(min(limit, len(digits.images))):
            image = digits.images[idx]
            label = int(digits.target[idx])
            cells = []
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    if image[y, x] > 6:
                        cells.append([float(x), float(y), 1.0, 1.0])
            fh.write(
                json.dumps(
                    {
                        "source": "sklearn.datasets.load_digits_public_fallback",
                        "doc_category": f"digit_{label}",
                        "page_hash": str(idx),
                        "width": 8.0,
                        "height": 8.0,
                        "labels": [f"ink_{label}"] * max(1, len(cells)),
                        "boxes": cells[:128] or [[0.0, 0.0, 1.0, 1.0]],
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    return min(limit, len(digits.images))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--out", type=Path, default=Path("data/processed/public_layout_sample.jsonl"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    try:
        count = download_doclaynet(args.limit, args.out)
        source = "DocLayNet"
    except Exception as exc:
        print(f"DocLayNet streaming failed, using bundled public fallback dataset: {exc}")
        count = write_fallback_public_sample(args.limit, args.out)
        source = "sklearn digits fallback"

    print(json.dumps({"written": count, "source": source, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
