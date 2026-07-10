from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[index])


def _dark_density(image: Image.Image, bbox_norm: list[float] | None = None) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if bbox_norm is not None:
        cx, cy, bw, bh = [float(v) for v in bbox_norm]
        height, width = gray.shape
        x0 = int(max(0, min(width - 1, round((cx - bw / 2) * width))))
        y0 = int(max(0, min(height - 1, round((cy - bh / 2) * height))))
        x1 = int(max(x0 + 1, min(width, round((cx + bw / 2) * width))))
        y1 = int(max(y0 + 1, min(height, round((cy + bh / 2) * height))))
        gray = gray[y0:y1, x0:x1]
    return float((gray < 175).mean()) if gray.size else 0.0


def _write_issue_sheet(rows: list[dict[str, Any]], output: Path, *, limit: int = 36) -> None:
    if not rows:
        return
    thumbs: list[Image.Image] = []
    for row in rows[:limit]:
        image = Image.open(row["image"]).convert("RGB").resize((160, 160))
        draw = ImageDraw.Draw(image)
        cx, cy, bw, bh = row["bbox"]
        x0 = (cx - bw / 2) * 160
        y0 = (cy - bh / 2) * 160
        x1 = (cx + bw / 2) * 160
        y1 = (cy + bh / 2) * 160
        draw.rectangle((x0, y0, x1, y1), outline=(196, 64, 48), width=3)
        draw.text((5, 5), f"{row['label']} {row['target_dark_density']:.3f}", fill=(196, 64, 48))
        thumbs.append(image)
    cols = 6
    sheet = Image.new("RGB", (cols * 160, ((len(thumbs) + cols - 1) // cols) * 160), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 160, (index // cols) * 160))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def audit_dataset(dataset: Path, output: Path, *, low_density_threshold: float) -> dict[str, Any]:
    rows = _load_jsonl(dataset)
    class_counts: collections.Counter[str] = collections.Counter()
    split_counts: collections.Counter[str] = collections.Counter()
    class_density: dict[str, list[float]] = collections.defaultdict(list)
    low_density_rows: list[dict[str, Any]] = []
    missing_images = 0

    for row in rows:
        label = str(row["label"])
        split = str(row.get("split", "unknown"))
        class_counts[label] += 1
        split_counts[split] += 1
        image_path = Path(row["image"])
        if not image_path.exists():
            missing_images += 1
            continue
        image = Image.open(image_path).convert("RGB")
        patch_density = _dark_density(image)
        row["patch_dark_density"] = round(patch_density, 6)
        if label == "background":
            continue
        target_density = _dark_density(image, row["bbox"])
        row["target_dark_density"] = round(target_density, 6)
        class_density[label].append(target_density)
        if target_density < low_density_threshold:
            low_density_rows.append(row)

    density_rows: list[dict[str, Any]] = []
    for label, values in sorted(class_density.items()):
        density_rows.append(
            {
                "class": label,
                "count": len(values),
                "mean": round(float(statistics.mean(values)), 6) if values else 0.0,
                "median": round(float(statistics.median(values)), 6) if values else 0.0,
                "p05": round(_percentile(values, 0.05), 6),
                "p25": round(_percentile(values, 0.25), 6),
                "p75": round(_percentile(values, 0.75), 6),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    issue_sheet = output.with_suffix(".low_density.jpg")
    _write_issue_sheet(low_density_rows, issue_sheet)
    report = {
        "dataset": str(dataset),
        "rows": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "missing_images": missing_images,
        "low_density_threshold": low_density_threshold,
        "low_density_positive_rows": len(low_density_rows),
        "low_density_positive_rate": round(
            len(low_density_rows) / max(sum(len(v) for v in class_density.values()), 1), 6
        ),
        "density_by_class": density_rows,
        "low_density_sheet": str(issue_sheet) if low_density_rows else "",
        "quality_gates": {
            "no_missing_images": missing_images == 0,
            "low_density_positive_rate_below_10pct": (
                len(low_density_rows) / max(sum(len(v) for v in class_density.values()), 1)
            )
            < 0.10,
        },
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-density-threshold", type=float, default=0.012)
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.dataset, args.output, low_density_threshold=args.low_density_threshold), indent=2))


if __name__ == "__main__":
    main()
