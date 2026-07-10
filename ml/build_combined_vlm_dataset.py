from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rows(rows: list[dict[str, Any]], *, prefix: str, source: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        item["id"] = f"{prefix}-{row.get('id', index)}"
        item["source_dataset"] = source
        item.setdefault("images", [])
        item.setdefault("split", "train")
        normalized.append(item)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buili",
        type=Path,
        default=Path("data/processed/buili_vlm/sft_dataset.jsonl"),
    )
    parser.add_argument(
        "--open-corpus",
        type=Path,
        default=Path("data/processed/open_construction_corpus/sft_dataset.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/buili_vlm_plus_open"),
    )
    args = parser.parse_args()

    buili_rows = normalize_rows(read_jsonl(args.buili), prefix="buili", source="buili_pm_review")
    open_rows = normalize_rows(
        read_jsonl(args.open_corpus),
        prefix="open",
        source="open_construction_corpus",
    )
    rows = buili_rows + open_rows
    for idx, row in enumerate(rows):
        row["split"] = "eval" if idx % 10 == 0 else "train"

    out_path = args.out_dir / "sft_dataset.jsonl"
    write_jsonl(out_path, rows)
    manifest = {
        "created_at": int(time.time()),
        "path": str(out_path),
        "sha256": sha256_file(out_path),
        "rows": len(rows),
        "train_rows": sum(1 for row in rows if row["split"] == "train"),
        "eval_rows": sum(1 for row in rows if row["split"] == "eval"),
        "image_rows": sum(1 for row in rows if row.get("images")),
        "sources": [
            {"path": str(args.buili), "rows": len(buili_rows), "role": "pm_review_contract"},
            {
                "path": str(args.open_corpus),
                "rows": len(open_rows),
                "role": "public_government_inspection_and_field_weak_labels",
            },
        ],
        "governance": (
            "Combined dataset keeps public source attribution in the upstream manifests. "
            "CC-BY-NC/gated sources are excluded unless separately licensed."
        ),
    }
    manifest_path = args.out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
