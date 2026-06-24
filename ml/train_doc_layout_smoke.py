from __future__ import annotations

# ruff: noqa: E402,I001

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.buili.gpu import assert_gpu_7, force_gpu_7

force_gpu_7()

import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def featurize(record: dict) -> list[float]:
    width = max(float(record.get("width", 1.0)), 1.0)
    height = max(float(record.get("height", 1.0)), 1.0)
    labels = record.get("labels") or []
    boxes = record.get("boxes") or []
    counts = Counter(str(label) for label in labels)
    top_counts = [float(count) for _, count in counts.most_common(12)]
    top_counts += [0.0] * (12 - len(top_counts))
    areas: list[float] = []
    aspect_ratios: list[float] = []
    for box in boxes[:128]:
        if len(box) < 4:
            continue
        _, _, bw, bh = [float(value) for value in box[:4]]
        areas.append(max(bw, 0.0) * max(bh, 0.0) / (width * height))
        aspect_ratios.append(max(bw, 1.0) / max(bh, 1.0))
    if not areas:
        areas = [0.0]
        aspect_ratios = [1.0]
    return [
        float(len(labels)),
        float(len(counts)),
        sum(areas) / len(areas),
        max(areas),
        min(areas),
        sum(aspect_ratios) / len(aspect_ratios),
        width / height,
        *top_counts,
    ]


class TinyLayoutClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_dataset(path: Path) -> tuple[torch.Tensor, list[str]]:
    xs: list[list[float]] = []
    ys: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            dominant = Counter(str(label) for label in record.get("labels", [])).most_common(1)
            doc_category = (
                f"dominant_layout_{dominant[0][0]}"
                if dominant
                else str(record.get("doc_category", "layout_unknown"))
            )
            xs.append(featurize(record))
            ys.append(doc_category)
    if not xs:
        raise RuntimeError(f"no records in {path}")
    return torch.tensor(xs, dtype=torch.float32), ys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/public_layout_sample.jsonl")
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("data/artifacts/layout_smoke"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    assert_gpu_7()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to train outside GPU 7 policy")

    device = torch.device("cuda:0")
    x, labels = load_dataset(args.data)
    encoder = LabelEncoder()
    y = torch.tensor(encoder.fit_transform(labels), dtype=torch.long)
    if len(encoder.classes_) < 2:
        y = torch.tensor([idx % 2 for idx in range(len(labels))], dtype=torch.long)
        encoder.fit(["layout_a", "layout_b"])

    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std

    counts = Counter(y.tolist())
    stratify = y.numpy() if len(counts) > 1 and min(counts.values()) >= 2 else None
    indices = list(range(len(x)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.25, random_state=42, stratify=stratify
    )
    train_ds = TensorDataset(x[train_idx], y[train_idx])
    val_ds = TensorDataset(x[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = TinyLayoutClassifier(x.shape[1], int(y.max().item()) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x.to(device)).argmax(dim=1).cpu()
                correct += int((pred == batch_y).sum())
                total += len(batch_y)
        acc = correct / max(total, 1)
        history.append(
            {"epoch": epoch, "loss": total_loss / max(len(train_loader), 1), "val_acc": acc}
        )
        print(history[-1])

    artifact = args.out_dir / "tiny_layout_classifier.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_dim": x.shape[1],
            "classes": list(encoder.classes_),
            "feature_mean": mean.squeeze(0).tolist(),
            "feature_std": std.squeeze(0).tolist(),
        },
        artifact,
    )
    joblib.dump(encoder, args.out_dir / "label_encoder.joblib")
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "cuda_visible_devices": "7",
                "torch_device": torch.cuda.get_device_name(0),
                "records": len(x),
                "classes": list(encoder.classes_),
                "history": history,
                "artifact": str(artifact),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {artifact}")


if __name__ == "__main__":
    main()
