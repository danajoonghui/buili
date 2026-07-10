from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.build_plan2field_micro_vlm_dataset import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/plan2field_vlm_primary"))
    parser.add_argument("--patch-size", type=int, default=160)
    parser.add_argument("--negatives-per-scene", type=int, default=18)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    manifest = build_dataset(
        args.out_dir,
        patch_size=args.patch_size,
        negatives_per_scene=args.negatives_per_scene,
        seed=args.seed,
    )
    manifest.update(
        {
            "dataset": "plan2field_vlm_primary_dense_plan_token_dataset",
            "source": (
                "Generated from public floor-plan crops and Buili semantic scene labels; "
                "intended for pretrained VLM-primary dense plan-token generation."
            ),
            "primary_runtime": (
                "Dense crop scanning with a pretrained CLIP vision-language encoder and "
                "domain token heads. Deterministic geometry is retained only for source "
                "coordinate snapping and 3D rendering."
            ),
        }
    )
    (args.out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
