from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.buili.gpu import assert_gpu_7, force_gpu_7, gpu_policy

force_gpu_7()

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from services.api.buili.spatial.micro_vlm import (
    MICRO_VLM_CLASSES,
    Plan2FieldMicroVLM,
    _image_tensor,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class PatchDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], *, image_size: int) -> None:
        self.rows = rows
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = Image.open(row["image"])
        return {
            "image": _image_tensor(image, self.image_size),
            "task_id": torch.tensor(int(row.get("task_id", 0)), dtype=torch.long),
            "class_id": torch.tensor(int(row["class_id"]), dtype=torch.long),
            "bbox": torch.tensor(row["bbox"], dtype=torch.float32),
            "objectness": torch.tensor(0.0 if row["label"] == "background" else 1.0, dtype=torch.float32),
            "id": row["id"],
            "label": row["label"],
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "task_id": torch.stack([item["task_id"] for item in batch]),
        "class_id": torch.stack([item["class_id"] for item in batch]),
        "bbox": torch.stack([item["bbox"] for item in batch]),
        "objectness": torch.stack([item["objectness"] for item in batch]),
        "id": [item["id"] for item in batch],
        "label": [item["label"] for item in batch],
    }


def evaluate(model: Plan2FieldMicroVLM, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    positives = 0
    positive_correct = 0
    bbox_l1_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            task_ids = batch["task_id"].to(device)
            class_ids = batch["class_id"].to(device)
            objectness = batch["objectness"].to(device)
            bbox = batch["bbox"].to(device)
            output = model(images, task_ids)
            pred = output["class_logits"].argmax(dim=-1)
            total += int(len(class_ids))
            correct += int((pred == class_ids).sum().detach().cpu())
            pos_mask = objectness > 0.5
            positives += int(pos_mask.sum().detach().cpu())
            if bool(pos_mask.any()):
                positive_correct += int((pred[pos_mask] == class_ids[pos_mask]).sum().detach().cpu())
                per_sample_l1 = torch.abs(output["box"][pos_mask] - bbox[pos_mask]).mean(dim=1)
                bbox_l1_sum += float(per_sample_l1.sum().detach().cpu())
    return {
        "class_acc": correct / max(total, 1),
        "positive_class_acc": positive_correct / max(positives, 1),
        "positive_bbox_l1": bbox_l1_sum / max(positives, 1),
        "samples": total,
        "positive_samples": positives,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    assert_gpu_7()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to train outside GPU 7.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    rows = read_jsonl(args.dataset)
    train_rows = [row for row in rows if row.get("split") == "train"]
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    if not train_rows or not eval_rows:
        raise RuntimeError("dataset must contain train and eval rows")

    train_loader = DataLoader(
        PatchDataset(train_rows, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        PatchDataset(eval_rows, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    model = Plan2FieldMicroVLM(
        classes=MICRO_VLM_CLASSES,
        image_size=args.image_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        task_vocab=args.task_vocab,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.02)
    class_loss = nn.CrossEntropyLoss()
    objectness_loss = nn.BCEWithLogitsLoss()
    bbox_loss = nn.SmoothL1Loss()
    history: list[dict[str, Any]] = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        progress = tqdm(train_loader, desc=f"micro_vlm epoch {epoch}/{args.epochs}")
        for batch in progress:
            images = batch["image"].to(device)
            task_ids = batch["task_id"].to(device)
            class_ids = batch["class_id"].to(device)
            objectness = batch["objectness"].to(device)
            bbox = batch["bbox"].to(device)
            output = model(images, task_ids)
            pos_mask = objectness > 0.5
            loss = class_loss(output["class_logits"], class_ids)
            loss = loss + 0.55 * objectness_loss(output["objectness"], objectness)
            if bool(pos_mask.any()):
                loss = loss + args.bbox_loss_weight * bbox_loss(output["box"][pos_mask], bbox[pos_mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
            progress.set_postfix({"loss": f"{total_loss / max(batches, 1):.4f}"})
        metrics = evaluate(model, eval_loader, device)
        metrics.update({"epoch": epoch, "train_loss": total_loss / max(batches, 1)})
        history.append(metrics)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "trained",
        "model_family": "Buili Plan2Field Micro-VLM",
        "architecture": "cnn_patch_encoder_text_task_token_tiny_transformer",
        "production_role": "primary_lightweight_vlm_plan_patch_parser",
        "dataset": str(args.dataset),
        "classes": MICRO_VLM_CLASSES,
        "epochs": args.epochs,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "seconds": round(time.perf_counter() - start, 4),
        "gpu": gpu_policy(),
        "torch_device": torch.cuda.get_device_name(0),
        "config": {
            "image_size": args.image_size,
            "dim": args.dim,
            "depth": args.depth,
            "heads": args.heads,
            "task_vocab": args.task_vocab,
        },
        "final_eval": history[-1],
        "history": history,
        "runtime_contract": {
            "max_patch_grid_seconds_target": 1.4,
            "fallback": "deterministic_geometry_and_ocr_remain_enabled",
            "output_schema": "SemanticObject/SemanticOpening merged into SemanticScene JSON",
        },
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": MICRO_VLM_CLASSES,
            "config": summary["config"],
            "summary": summary,
        },
        args.out_dir / "micro_vlm.pt",
    )
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/plan2field_micro_vlm/dataset.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/artifacts/plan2field_micro_vlm"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--task-vocab", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--bbox-loss-weight", type=float, default=1.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
