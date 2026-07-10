from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.buili.spatial.vlm_primary import (
    build_vlm_primary_plan2field3d_artifacts,
)


def _measure(stage_name: str, stages: dict[str, float], fn: Any) -> Any:
    start = time.perf_counter()
    result = fn()
    stages[stage_name] = round(time.perf_counter() - start, 4)
    return result


def run_benchmark(source_pdf: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "vlm_primary_full_pipeline_timing.json"
    stages: dict[str, float] = {}
    total_start = time.perf_counter()
    auto_summary = _measure(
        "vlm_primary_pdf_to_3d_seconds",
        stages,
        lambda: build_vlm_primary_plan2field3d_artifacts(
            source_pdf,
            output_dir,
            use_ocr=True,
        ),
    )
    preview_png = Path(str(auto_summary["preview_png"]))
    image = Image.open(preview_png)
    scene_build = auto_summary["scene_build"]
    vlm = scene_build.get("vlm_primary", {})
    summary = {
        "input_pdf": str(source_pdf),
        "preview_png": str(preview_png),
        "total_automatic_pipeline_seconds": round(time.perf_counter() - total_start, 4),
        "stages": stages,
        "auto_summary": auto_summary,
        "output_png": {
            "size_px": list(image.size),
            "bytes": preview_png.stat().st_size,
        },
        "paper_claim_surface": {
            "method": "Proposal-guided VLM-primary semantic plan token generation with deterministic geometry snapping",
            "pretrained_vlm_encoder": vlm.get("config", {}).get("encoder_id", "unknown"),
            "vlm_primary_seconds": vlm.get("seconds"),
            "proposal_patches": vlm.get("proposal_patches"),
            "dense_scan_enabled": vlm.get("dense_scan_enabled"),
            "vlm_generated_objects": vlm.get("objects_generated"),
            "vlm_generated_openings": vlm.get("openings_generated"),
            "walls": scene_build.get("counts", {}).get("walls"),
            "alignment_qa_passed": auto_summary.get("alignment_qa", {}).get("quality_gate_passed"),
        },
        "included": [
            "PDF read and page raster render",
            "OCR room/dimension extraction for labels and scale context",
            "Pretrained CLIP VLM proposal-guided plan-token generation for objects/openings",
            "Deterministic wall geometry snapper for source-pixel coordinate fidelity",
            "SemanticScene compile without manual seed",
            "Wall opening cut and lightweight procedural object rendering",
            "Isometric PNG rendering",
        ],
        "not_included_in_current_measured_path": [
            "Large cloud VLM fallback",
            "Manual review time",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-pdf",
        type=Path,
        default=Path("data/sources/plan2field3d_house/maricopa_sample_floor_plan.pdf"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/plan2field3d_house_conversion/vlm_primary_timing"),
    )
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.source_pdf, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
