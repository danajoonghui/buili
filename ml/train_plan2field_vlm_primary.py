from __future__ import annotations

# ruff: noqa: E402

import argparse
import collections
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from services.api.buili.spatial.vlm_primary import (
    DEFAULT_ENCODER_ID,
    Plan2FieldVLMPrimary,
    VLM_PRIMARY_CLASSES,
    _clip_tensor,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class PlanTokenDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], *, image_size: int) -> None:
        self.rows = rows
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = Image.open(row["image"])
        return {
            "pixel_values": _clip_tensor(image, self.image_size),
            "task_id": torch.tensor(int(row.get("task_id", 0)), dtype=torch.long),
            "class_id": torch.tensor(int(row["class_id"]), dtype=torch.long),
            "bbox": torch.tensor(row["bbox"], dtype=torch.float32),
            "objectness": torch.tensor(0.0 if row["label"] == "background" else 1.0, dtype=torch.float32),
            "id": row["id"],
            "label": row["label"],
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "task_id": torch.stack([item["task_id"] for item in batch]),
        "class_id": torch.stack([item["class_id"] for item in batch]),
        "bbox": torch.stack([item["bbox"] for item in batch]),
        "objectness": torch.stack([item["objectness"] for item in batch]),
        "id": [item["id"] for item in batch],
        "label": [item["label"] for item in batch],
    }


def evaluate(model: Plan2FieldVLMPrimary, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    positives = 0
    positive_correct = 0
    bbox_l1_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            task_ids = batch["task_id"].to(device)
            class_ids = batch["class_id"].to(device)
            objectness = batch["objectness"].to(device)
            bbox = batch["bbox"].to(device)
            output = model(pixel_values, task_ids)
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


def trainable_state_dict(model: Plan2FieldVLMPrimary) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("vision.")
    }


def _class_counts(rows: list[dict[str, Any]]) -> collections.Counter[int]:
    return collections.Counter(int(row["class_id"]) for row in rows)


def _class_loss_weights(
    rows: list[dict[str, Any]],
    *,
    num_classes: int,
    max_weight: float,
) -> torch.Tensor:
    counts = _class_counts(rows)
    total = sum(counts.values())
    weights: list[float] = []
    for class_id in range(num_classes):
        count = counts.get(class_id, 0)
        if count <= 0:
            weights.append(0.0)
            continue
        # Square-root inverse frequency is less brittle than full inverse
        # weighting for floorplan classes with a >200x long tail.
        raw = (total / max(count, 1)) ** 0.5
        weights.append(min(max_weight, raw))
    observed = [weight for weight in weights if weight > 0.0]
    mean_observed = sum(observed) / max(len(observed), 1)
    normalized = [weight / mean_observed if weight > 0.0 else 0.0 for weight in weights]
    return torch.tensor(normalized, dtype=torch.float32)


def _sample_weights(
    rows: list[dict[str, Any]],
    *,
    max_weight: float,
) -> list[float]:
    counts = _class_counts(rows)
    total = sum(counts.values())
    weights: list[float] = []
    for row in rows:
        count = counts[int(row["class_id"])]
        raw = (total / max(count, 1)) ** 0.5
        weights.append(min(max_weight, raw))
    return weights


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

    sampler = None
    shuffle = True
    if args.balanced_sampler:
        sampler = WeightedRandomSampler(
            _sample_weights(train_rows, max_weight=args.max_sample_weight),
            num_samples=len(train_rows),
            replacement=True,
        )
        shuffle = False
    train_loader = DataLoader(
        PlanTokenDataset(train_rows, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        PlanTokenDataset(eval_rows, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    model = Plan2FieldVLMPrimary(
        encoder_id=args.encoder_id,
        classes=VLM_PRIMARY_CLASSES,
        task_vocab=args.task_vocab,
        freeze_vision=True,
    ).to(device)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.02)
    ce_weight = None
    if args.class_loss_reweight:
        ce_weight = _class_loss_weights(
            train_rows,
            num_classes=len(VLM_PRIMARY_CLASSES),
            max_weight=args.max_class_weight,
        ).to(device)
    class_loss = nn.CrossEntropyLoss(weight=ce_weight)
    objectness_loss = nn.BCEWithLogitsLoss()
    bbox_loss = nn.SmoothL1Loss()
    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        progress = tqdm(train_loader, desc=f"vlm_primary epoch {epoch}/{args.epochs}")
        for batch in progress:
            pixel_values = batch["pixel_values"].to(device)
            task_ids = batch["task_id"].to(device)
            class_ids = batch["class_id"].to(device)
            objectness = batch["objectness"].to(device)
            bbox = batch["bbox"].to(device)
            output = model(pixel_values, task_ids)
            pos_mask = objectness > 0.5
            loss = class_loss(output["class_logits"], class_ids)
            loss = loss + 0.55 * objectness_loss(output["objectness"], objectness)
            if bool(pos_mask.any()):
                loss = loss + args.bbox_loss_weight * bbox_loss(output["box"][pos_mask], bbox[pos_mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
            progress.set_postfix({"loss": f"{total_loss / max(batches, 1):.4f}"})
        metrics = evaluate(model, eval_loader, device)
        metrics.update({"epoch": epoch, "train_loss": total_loss / max(batches, 1)})
        history.append(metrics)
        metric = metrics["positive_class_acc"] - metrics["positive_bbox_l1"]
        if metric > best_metric:
            best_metric = metric
            best_state = trainable_state_dict(model)
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_eval = evaluate(model, eval_loader, device)
    summary = {
        "status": "trained",
        "model_family": "Buili Plan2Field VLM-Primary Token Generator",
        "architecture": "frozen_clip_vision_language_encoder_domain_plan_token_heads",
        "production_role": "primary_pretrained_vlm_plan_token_generator",
        "encoder_id": args.encoder_id,
        "dataset": str(args.dataset),
        "classes": VLM_PRIMARY_CLASSES,
        "epochs": args.epochs,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "seconds": round(time.perf_counter() - start, 4),
        "gpu": gpu_policy(),
        "torch_device": torch.cuda.get_device_name(0),
        "trainable_params": sum(parameter.numel() for parameter in trainable_params),
        "frozen_vision_params": sum(
            parameter.numel() for parameter in model.vision.parameters() if not parameter.requires_grad
        ),
        "config": {
            "encoder_id": args.encoder_id,
            "image_size": args.image_size,
            "task_vocab": args.task_vocab,
            "freeze_vision": True,
            "balanced_sampler": args.balanced_sampler,
            "class_loss_reweight": args.class_loss_reweight,
            "max_sample_weight": args.max_sample_weight,
            "max_class_weight": args.max_class_weight,
        },
        "final_eval": final_eval,
        "best_epoch": best_epoch,
        "selection_metric": "positive_class_acc - positive_bbox_l1",
        "history": history,
        "runtime_contract": {
            "target_vlm_primary_seconds": 1.8,
            "semantic_role": "VLM generates objects/openings directly from proposal-centered plan tiles; dense scan remains optional",
            "geometry_role": "deterministic solver snaps walls/source coordinates and renders 3D",
            "fallback": "existing deterministic and Micro-VLM verifier pipelines remain available",
        },
        "artifact_contract": {
            "saved_weights": "trainable_domain_heads_only",
            "frozen_encoder_loaded_from": args.encoder_id,
            "reason": "small deployable artifact while retaining pretrained VLM semantics",
        },
        "label_quality_controls": {
            "class_counts_train": dict(sorted(_class_counts(train_rows).items())),
            "balanced_sampler": args.balanced_sampler,
            "class_loss_reweight": args.class_loss_reweight,
            "weighting": "sqrt_inverse_frequency_capped",
        },
    }
    torch.save(
        {
            "head_state": trainable_state_dict(model),
            "classes": VLM_PRIMARY_CLASSES,
            "config": summary["config"],
            "summary": summary,
        },
        args.out_dir / "vlm_primary.pt",
    )
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/plan2field_vlm_primary/dataset.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/artifacts/plan2field_vlm_primary"))
    parser.add_argument("--encoder-id", type=str, default=DEFAULT_ENCODER_ID)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--task-vocab", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--bbox-loss-weight", type=float, default=1.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--class-loss-reweight", action="store_true")
    parser.add_argument("--max-sample-weight", type=float, default=12.0)
    parser.add_argument("--max-class-weight", type=float, default=12.0)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
