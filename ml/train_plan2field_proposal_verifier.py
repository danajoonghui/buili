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

import numpy as np

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
from transformers import CLIPVisionModel

from ml.build_plan2field_proposal_verifier_dataset import PROPOSAL_CLASSES
from services.api.buili.spatial.vlm_primary import _clip_tensor


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _feature_vector(row: dict[str, Any], *, class_count: int) -> torch.Tensor:
    class_id = int(row["class_id"])
    one_hot = torch.zeros(class_count, dtype=torch.float32)
    one_hot[class_id] = 1.0
    score = float(row.get("score", 0.0))
    area = float(row.get("area_fraction", 0.0))
    aspect = float(row.get("aspect_log", 0.0))
    density = float(row.get("dark_density", 0.0))
    return torch.cat(
        [
            torch.tensor(
                [
                    score,
                    min(area * 100.0, 1.0),
                    max(-3.0, min(3.0, aspect)) / 3.0,
                    density,
                    float(row.get("target_group") == "openings"),
                ],
                dtype=torch.float32,
            ),
            one_hot,
        ]
    )


class ProposalDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], *, image_size: int) -> None:
        self.rows = rows
        self.image_size = image_size
        self.class_count = len(PROPOSAL_CLASSES)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = Image.open(row["image"]).convert("RGB")
        return {
            "pixel_values": _clip_tensor(image, self.image_size),
            "features": _feature_vector(row, class_count=self.class_count),
            "target": torch.tensor(1.0 if row["keep"] else 0.0, dtype=torch.float32),
            "class_id": torch.tensor(int(row["class_id"]), dtype=torch.long),
            "group_id": torch.tensor(1 if row["target_group"] == "openings" else 0, dtype=torch.long),
            "id": row["id"],
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "features": torch.stack([item["features"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "class_id": torch.stack([item["class_id"] for item in batch]),
        "group_id": torch.stack([item["group_id"] for item in batch]),
        "id": [item["id"] for item in batch],
    }


class ProposalVerifier(nn.Module):
    def __init__(
        self,
        *,
        encoder_id: str,
        feature_dim: int,
        hidden: int = 512,
        freeze_vision: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_id = encoder_id
        self.vision = CLIPVisionModel.from_pretrained(encoder_id)
        vision_dim = int(self.vision.config.hidden_size)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(vision_dim + feature_dim),
            nn.Linear(vision_dim + feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.keep_head = nn.Linear(hidden // 2, 1)
        self.class_head = nn.Linear(hidden // 2, len(PROPOSAL_CLASSES))
        self.group_head = nn.Linear(hidden // 2, 2)
        if freeze_vision:
            for parameter in self.vision.parameters():
                parameter.requires_grad_(False)

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.vision.eval()
        with torch.no_grad():
            return self.vision(pixel_values=pixel_values).pooler_output

    def forward(self, pixel_values: torch.Tensor, features: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.encode(pixel_values)
        token = self.head(torch.cat([pooled, self.feature_norm(features)], dim=-1))
        return {
            "keep_logit": self.keep_head(token).squeeze(-1),
            "class_logits": self.class_head(token),
            "group_logits": self.group_head(token),
        }


def _sample_weights(rows: list[dict[str, Any]]) -> list[float]:
    counts = collections.Counter(bool(row["keep"]) for row in rows)
    total = max(len(rows), 1)
    return [total / max(counts[bool(row["keep"])], 1) for row in rows]


def _binary_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    thresholds: list[float],
) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for threshold in thresholds:
        pred = probs >= threshold
        tp = int(np.logical_and(pred, targets == 1).sum())
        fp = int(np.logical_and(pred, targets == 0).sum())
        fn = int(np.logical_and(~pred, targets == 1).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        row = {
            "threshold": round(float(threshold), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        if not best or row["f1"] > best["f1"]:
            best = row
    return best


def evaluate(
    model: ProposalVerifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    probs: list[float] = []
    targets: list[int] = []
    class_correct = 0
    class_total = 0
    group_correct = 0
    group_total = 0
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["pixel_values"].to(device),
                batch["features"].to(device),
            )
            prob = torch.sigmoid(output["keep_logit"]).detach().cpu().numpy()
            target = batch["target"].numpy()
            probs.extend(float(value) for value in prob)
            targets.extend(int(value) for value in target)
            positive = batch["target"] > 0.5
            if bool(positive.any()):
                class_pred = output["class_logits"].detach().cpu().argmax(dim=-1)
                group_pred = output["group_logits"].detach().cpu().argmax(dim=-1)
                class_correct += int((class_pred[positive] == batch["class_id"][positive]).sum())
                class_total += int(positive.sum())
                group_correct += int((group_pred[positive] == batch["group_id"][positive]).sum())
                group_total += int(positive.sum())
    probs_np = np.asarray(probs, dtype=np.float32)
    targets_np = np.asarray(targets, dtype=np.int64)
    thresholds = [index / 100 for index in range(5, 96, 5)]
    return {
        "rows": int(len(targets_np)),
        "positives": int(targets_np.sum()),
        "positive_ratio": round(float(targets_np.mean()), 6) if len(targets_np) else 0.0,
        "prob_mean": round(float(probs_np.mean()), 6) if len(probs_np) else 0.0,
        "prob_positive_mean": round(float(probs_np[targets_np == 1].mean()), 6)
        if bool((targets_np == 1).any())
        else 0.0,
        "prob_negative_mean": round(float(probs_np[targets_np == 0].mean()), 6)
        if bool((targets_np == 0).any())
        else 0.0,
        "best_f1": _binary_metrics(probs_np, targets_np, thresholds=thresholds),
        "positive_class_acc": round(class_correct / max(class_total, 1), 6),
        "positive_group_acc": round(group_correct / max(group_total, 1), 6),
    }


def _trainable_state_dict(model: ProposalVerifier) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("vision.")
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    assert_gpu_7()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to train outside GPU 7.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = read_jsonl(args.dataset_dir / "train.jsonl")
    val_rows = read_jsonl(args.dataset_dir / "val.jsonl")
    feature_dim = 5 + len(PROPOSAL_CLASSES)
    device = torch.device("cuda:0")
    train_loader = DataLoader(
        ProposalDataset(train_rows, image_size=args.image_size),
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(_sample_weights(train_rows), len(train_rows), replacement=True),
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ProposalDataset(val_rows, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    model = ProposalVerifier(
        encoder_id=args.encoder_id,
        feature_dim=feature_dim,
        hidden=args.hidden,
        freeze_vision=True,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    keep_loss = nn.BCEWithLogitsLoss()
    class_loss = nn.CrossEntropyLoss()
    group_loss = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        progress = tqdm(train_loader, desc=f"proposal_verifier epoch {epoch}/{args.epochs}")
        for batch in progress:
            pixel_values = batch["pixel_values"].to(device)
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            output = model(pixel_values, features)
            loss = keep_loss(output["keep_logit"], target)
            positive = target > 0.5
            if bool(positive.any()):
                loss = loss + 0.15 * class_loss(
                    output["class_logits"][positive],
                    batch["class_id"].to(device)[positive],
                )
                loss = loss + 0.08 * group_loss(
                    output["group_logits"][positive],
                    batch["group_id"].to(device)[positive],
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
            progress.set_postfix({"loss": f"{total_loss / max(batches, 1):.4f}"})
        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = round(total_loss / max(batches, 1), 6)
        history.append(metrics)
        metric = float(metrics["best_f1"]["f1"])
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = _trainable_state_dict(model)

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    final_val = evaluate(model, val_loader, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "trained",
        "method": "Buili Proposal Verifier",
        "architecture": "frozen_clip_vision_encoder_plus_detector_geometry_features_hard_negative_head",
        "encoder_id": args.encoder_id,
        "dataset_dir": str(args.dataset_dir),
        "classes": PROPOSAL_CLASSES,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "seconds": round(time.perf_counter() - start, 4),
        "gpu": gpu_policy(),
        "torch_device": torch.cuda.get_device_name(0),
        "trainable_params": int(sum(parameter.numel() for parameter in trainable)),
        "frozen_vision_params": int(
            sum(parameter.numel() for parameter in model.vision.parameters() if not parameter.requires_grad)
        ),
        "config": {
            "encoder_id": args.encoder_id,
            "image_size": args.image_size,
            "feature_dim": feature_dim,
            "hidden": args.hidden,
            "freeze_vision": True,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "final_val": final_val,
        "history": history,
    }
    torch.save(
        {
            "head_state": _trainable_state_dict(model),
            "classes": PROPOSAL_CLASSES,
            "summary": summary,
            "config": summary["config"],
        },
        args.out_dir / "proposal_verifier.pt",
    )
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--encoder-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260704)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
