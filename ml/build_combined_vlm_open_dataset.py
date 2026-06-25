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


def normalize_row(row: dict[str, Any], *, prefix: str, fallback_task: str) -> dict[str, Any]:
    normalized = dict(row)
    original_id = str(normalized.get("id", "row"))
    normalized["id"] = f"{prefix}-{original_id}"
    normalized["task"] = str(normalized.get("task") or fallback_task)
    normalized["images"] = list(normalized.get("images") or [])
    normalized["split"] = str(normalized.get("split") or "train")
    normalized["source_dataset"] = prefix
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
        "--out",
        type=Path,
        default=Path("data/processed/buili_vlm_plus_open/sft_dataset.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/buili_vlm_plus_open/dataset_manifest.json"),
    )
    parser.add_argument("--max-open-train", type=int, default=260)
    parser.add_argument("--max-open-eval", type=int, default=40)
    args = parser.parse_args()

    buili_rows = [
        normalize_row(row, prefix="buili", fallback_task="pm_review_contract")
        for row in read_jsonl(args.buili)
    ]
    open_rows = [
        normalize_row(row, prefix="open", fallback_task="open_construction_record")
        for row in read_jsonl(args.open_corpus)
    ]
    open_train = [row for row in open_rows if row["split"] == "train"][: args.max_open_train]
    open_eval = [row for row in open_rows if row["split"] == "eval"][: args.max_open_eval]
    rows = buili_rows + open_train + open_eval
    write_jsonl(args.out, rows)

    manifest = {
        "created_at": int(time.time()),
        "path": str(args.out),
        "sha256": sha256_file(args.out),
        "rows": len(rows),
        "train_rows": sum(row["split"] == "train" for row in rows),
        "eval_rows": sum(row["split"] == "eval" for row in rows),
        "image_rows": sum(bool(row.get("images")) for row in rows),
        "task_counts": {
            task: sum(row["task"] == task for row in rows)
            for task in sorted({row["task"] for row in rows})
        },
        "sources": [
            {
                "path": str(args.buili),
                "rows": len(buili_rows),
                "role": "pm_review_contract",
            },
            {
                "path": str(args.open_corpus),
                "rows": len(open_train) + len(open_eval),
                "role": "public_government_inspection_and_field_weak_labels",
            },
        ],
        "governance": (
            "Combined dataset keeps public source attribution in upstream manifests. "
            "CC-BY-NC/gated sources are excluded unless separately licensed."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
