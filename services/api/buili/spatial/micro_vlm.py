from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .semantic_scene import ObjectKind, SemanticObject, SemanticOpening, SemanticRoomLabel

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - Render can run without torch.
    torch = None
    nn = None
    F = None


MICRO_VLM_CLASSES = [
    "background",
    "bathtub",
    "cabinet_run",
    "ceiling_light",
    "door",
    "duplex_outlet",
    "fixture_tag",
    "sink",
    "shower",
    "smoke_detector",
    "switch",
    "toilet",
    "washer_dryer",
    "water_heater",
    "window",
]

OBJECT_CLASSES: set[str] = {
    "bathtub",
    "cabinet_run",
    "ceiling_light",
    "duplex_outlet",
    "fixture_tag",
    "sink",
    "shower",
    "smoke_detector",
    "switch",
    "toilet",
    "washer_dryer",
    "water_heater",
}

OPENING_CLASSES = {"door", "window"}

_MODEL_CACHE: tuple[Path, Any, dict[str, Any]] | None = None


if nn is not None:

    class Plan2FieldMicroVLM(nn.Module):
        """Tiny language-conditioned plan patch parser.

        It is intentionally small: a compact visual encoder, a task text embedding, a
        two-layer transformer, and bounded heads for class/objectness/box prediction.
        The expensive VLM is used offline as teacher/data generator; this model is the
        production path.
        """

        def __init__(
            self,
            *,
            classes: list[str] | None = None,
            image_size: int = 128,
            dim: int = 128,
            depth: int = 2,
            heads: int = 4,
            task_vocab: int = 8,
        ) -> None:
            super().__init__()
            self.classes = classes or MICRO_VLM_CLASSES
            self.image_size = image_size
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(64, dim, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, stride=2, padding=1),
                nn.GELU(),
            )
            tokens_per_side = image_size // 16
            self.pos = nn.Parameter(torch.zeros(1, tokens_per_side * tokens_per_side + 1, dim))
            self.task = nn.Embedding(task_vocab, dim)
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 3,
                dropout=0.05,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
            self.norm = nn.LayerNorm(dim)
            self.class_head = nn.Linear(dim, len(self.classes))
            self.objectness_head = nn.Linear(dim, 1)
            self.box_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 4))

        def forward(self, images: torch.Tensor, task_ids: torch.Tensor) -> dict[str, torch.Tensor]:
            feat = self.encoder(images)
            tokens = feat.flatten(2).transpose(1, 2)
            task_token = self.task(task_ids).unsqueeze(1)
            tokens = torch.cat([task_token, tokens], dim=1)
            tokens = tokens + self.pos[:, : tokens.shape[1]]
            pooled = self.norm(self.transformer(tokens)[:, 0])
            return {
                "class_logits": self.class_head(pooled),
                "objectness": self.objectness_head(pooled).squeeze(-1),
                "box": torch.sigmoid(self.box_head(pooled)),
            }

else:  # pragma: no cover
    Plan2FieldMicroVLM = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MicroVLMPrediction:
    label: str
    confidence: float
    bbox_px: tuple[float, float, float, float]
    source: str

    @property
    def center_px(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox_px
        return (x0 + x1) / 2, (y0 + y1) / 2

    @property
    def width_px(self) -> float:
        return max(1.0, self.bbox_px[2] - self.bbox_px[0])

    @property
    def height_px(self) -> float:
        return max(1.0, self.bbox_px[3] - self.bbox_px[1])


def default_artifact_path() -> Path:
    return Path(
        os.environ.get(
            "BUILI_PLAN2FIELD_MICRO_VLM",
            "data/artifacts/plan2field_micro_vlm/micro_vlm.pt",
        )
    )


def _load_model(path: Path) -> tuple[Any | None, dict[str, Any]]:
    global _MODEL_CACHE
    if torch is None or Plan2FieldMicroVLM is None:
        return None, {"enabled": False, "reason": "torch_not_installed"}
    path = path.resolve()
    if not path.exists():
        return None, {"enabled": False, "reason": "artifact_missing", "path": str(path)}
    if _MODEL_CACHE and _MODEL_CACHE[0] == path:
        return _MODEL_CACHE[1], _MODEL_CACHE[2]
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    classes = checkpoint.get("classes") or MICRO_VLM_CLASSES
    model = Plan2FieldMicroVLM(
        classes=classes,
        image_size=int(config.get("image_size", 128)),
        dim=int(config.get("dim", 128)),
        depth=int(config.get("depth", 2)),
        heads=int(config.get("heads", 4)),
        task_vocab=int(config.get("task_vocab", 8)),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = "cuda:0" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    model.to(device)
    model.eval()
    metadata = {
        "enabled": True,
        "path": str(path),
        "device": device,
        "classes": classes,
        "config": config,
        "summary": checkpoint.get("summary", {}),
    }
    _MODEL_CACHE = (path, model, metadata)
    return model, metadata


def _image_tensor(image: Image.Image, image_size: int) -> Any:
    if torch is None:
        raise RuntimeError("torch is required")
    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    data = np.asarray(resized, dtype=np.uint8).copy()
    tensor = torch.from_numpy(data).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.25, 0.25, 0.25]).view(3, 1, 1)
    return (tensor - mean) / std


def _patch_boxes(width: int, height: int, *, patch: int, stride: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    if width <= patch and height <= patch:
        return [(0, 0, width, height)]
    ys = list(range(0, max(1, height - patch + 1), stride))
    xs = list(range(0, max(1, width - patch + 1), stride))
    if not ys or ys[-1] + patch < height:
        ys.append(max(0, height - patch))
    if not xs or xs[-1] + patch < width:
        xs.append(max(0, width - patch))
    for y in ys:
        for x in xs:
            boxes.append((x, y, min(width, x + patch), min(height, y + patch)))
    return boxes


def _centered_patch_box(
    width: int,
    height: int,
    center: tuple[float, float],
    *,
    patch: int,
) -> tuple[int, int, int, int]:
    cx, cy = center
    x0 = int(round(cx - patch / 2))
    y0 = int(round(cy - patch / 2))
    x0 = max(0, min(x0, max(0, width - patch)))
    y0 = max(0, min(y0, max(0, height - patch)))
    return x0, y0, min(width, x0 + patch), min(height, y0 + patch)


def _model_predictions(
    model: Any,
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    *,
    classes: list[str],
    image_size: int,
    batch_size: int,
    task_id: int = 0,
    keep_background: bool = False,
) -> list[MicroVLMPrediction]:
    if torch is None or F is None:
        return []
    if not boxes:
        return []
    device = next(model.parameters()).device
    predictions: list[MicroVLMPrediction] = []
    for batch_start in range(0, len(boxes), batch_size):
        batch_boxes = boxes[batch_start : batch_start + batch_size]
        patches = [_image_tensor(image.crop(box), image_size) for box in batch_boxes]
        images = torch.stack(patches).to(device)
        task_ids = torch.full((len(batch_boxes),), task_id, dtype=torch.long, device=device)
        with torch.no_grad():
            output = model(images, task_ids)
            probs = F.softmax(output["class_logits"], dim=-1)
            objectness = torch.sigmoid(output["objectness"])
            boxes_norm = output["box"].detach().cpu().tolist()
        for row, patch_box in enumerate(batch_boxes):
            class_id = int(probs[row].argmax().detach().cpu().item())
            label = classes[class_id]
            if label == "background" and not keep_background:
                continue
            confidence = float(
                probs[row, class_id].detach().cpu().item()
                * objectness[row].detach().cpu().item()
            )
            px0, py0, px1, py1 = patch_box
            patch_w = px1 - px0
            patch_h = py1 - py0
            cx, cy, bw, bh = boxes_norm[row]
            width = max(10.0, bw * patch_w)
            height = max(10.0, bh * patch_h)
            center_x = px0 + cx * patch_w
            center_y = py0 + cy * patch_h
            predictions.append(
                MicroVLMPrediction(
                    label=label,
                    confidence=confidence,
                    bbox_px=(
                        max(0.0, center_x - width / 2),
                        max(0.0, center_y - height / 2),
                        min(float(image.width), center_x + width / 2),
                        min(float(image.height), center_y + height / 2),
                    ),
                    source=f"patch={patch_box}",
                )
            )
    return predictions


def _nms(predictions: list[MicroVLMPrediction], iou_threshold: float = 0.25) -> list[MicroVLMPrediction]:
    def iou(a: MicroVLMPrediction, b: MicroVLMPrediction) -> float:
        ax0, ay0, ax1, ay1 = a.bbox_px
        bx0, by0, bx1, by1 = b.bbox_px
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        intersection = (ix1 - ix0) * (iy1 - iy0)
        area_a = max((ax1 - ax0) * (ay1 - ay0), 1.0)
        area_b = max((bx1 - bx0) * (by1 - by0), 1.0)
        return intersection / (area_a + area_b - intersection)

    kept: list[MicroVLMPrediction] = []
    for pred in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        if any(pred.label == item.label and iou(pred, item) > iou_threshold for item in kept):
            continue
        kept.append(pred)
    return kept


def _near_existing(
    pred: MicroVLMPrediction,
    objects: list[SemanticObject],
    openings: list[SemanticOpening],
) -> bool:
    cx, cy = pred.center_px
    if pred.label in OBJECT_CLASSES:
        return any(
            obj.kind == pred.label
            and math.hypot(cx - obj.center_px[0], cy - obj.center_px[1])
            < max(42.0, min(obj.width_px, obj.depth_px) * 0.9)
            for obj in objects
        )
    return any(
        opening.kind == pred.label
        and math.hypot(cx - opening.center_px[0], cy - opening.center_px[1])
        < max(44.0, opening.length_px * 0.38)
        for opening in openings
    )


def _nearest_room_distance(center: tuple[float, float], labels: list[SemanticRoomLabel]) -> float:
    if not labels:
        return 0.0
    cx, cy = center
    return min(math.hypot(cx - label.center_px[0], cy - label.center_px[1]) for label in labels)


def _prediction_to_object(pred: MicroVLMPrediction, index: int) -> SemanticObject | None:
    if pred.label not in OBJECT_CLASSES:
        return None
    return SemanticObject(
        id=f"micro_vlm_object_{index:03d}",
        kind=pred.label,  # type: ignore[arg-type]
        center_px=pred.center_px,
        width_px=max(22.0, pred.width_px),
        depth_px=max(22.0, pred.height_px),
        angle_deg=90.0 if pred.height_px > pred.width_px * 1.35 else 0.0,
        label=pred.label,
        source_note=f"micro_vlm_plan_patch:conf={pred.confidence:.2f}:{pred.source}",
    )


def _prediction_to_opening(pred: MicroVLMPrediction, index: int) -> SemanticOpening | None:
    if pred.label not in OPENING_CLASSES:
        return None
    return SemanticOpening(
        id=f"micro_vlm_opening_{index:03d}",
        kind=pred.label,  # type: ignore[arg-type]
        center_px=pred.center_px,
        length_px=max(pred.width_px, pred.height_px, 34.0),
        angle_deg=90.0 if pred.height_px > pred.width_px * 1.25 else 0.0,
        mark="µVLM",
        source_note=f"micro_vlm_plan_patch:conf={pred.confidence:.2f}:{pred.source}",
    )


def _verify_existing_candidates(
    model: Any,
    image: Image.Image,
    *,
    classes: list[str],
    image_size: int,
    patch: int,
    batch_size: int,
    existing_objects: list[SemanticObject],
    existing_openings: list[SemanticOpening],
    verify_threshold: float,
    correction_threshold: float,
) -> tuple[list[SemanticObject], list[SemanticOpening], dict[str, Any]]:
    verified_objects: list[SemanticObject] = []
    verified_openings: list[SemanticOpening] = []
    corrected_objects = 0
    corrected_openings = 0

    object_boxes = [
        _centered_patch_box(image.width, image.height, obj.center_px, patch=patch)
        for obj in existing_objects
    ]
    object_predictions = _model_predictions(
        model,
        image,
        object_boxes,
        classes=classes,
        image_size=image_size,
        batch_size=batch_size,
        task_id=2,
        keep_background=True,
    )
    for obj, pred in zip(existing_objects, object_predictions):
        if pred.confidence < verify_threshold:
            continue
        if pred.label == "background":
            continue
        if pred.label == obj.kind:
            verified_objects.append(
                replace(
                    obj,
                    id=f"micro_vlm_verified_{obj.id}",
                    source_note=(
                        f"micro_vlm_verified:kind={pred.label},conf={pred.confidence:.2f};"
                        f"base={obj.source_note}"
                    ),
                )
            )
        elif pred.label in OBJECT_CLASSES and pred.confidence >= correction_threshold:
            corrected_objects += 1
            verified_objects.append(
                replace(
                    obj,
                    id=f"micro_vlm_corrected_{obj.id}",
                    kind=pred.label,  # type: ignore[arg-type]
                    label=pred.label,
                    source_note=(
                        f"micro_vlm_corrected:{obj.kind}->{pred.label},"
                        f"conf={pred.confidence:.2f};base={obj.source_note}"
                    ),
                )
            )

    opening_boxes = [
        _centered_patch_box(image.width, image.height, opening.center_px, patch=patch)
        for opening in existing_openings
    ]
    opening_predictions = _model_predictions(
        model,
        image,
        opening_boxes,
        classes=classes,
        image_size=image_size,
        batch_size=batch_size,
        task_id=1,
        keep_background=True,
    )
    for opening, pred in zip(existing_openings, opening_predictions):
        if pred.confidence < verify_threshold:
            continue
        if pred.label == "background":
            continue
        if pred.label == opening.kind:
            verified_openings.append(
                replace(
                    opening,
                    id=f"micro_vlm_verified_{opening.id}",
                    source_note=(
                        f"micro_vlm_verified:kind={pred.label},conf={pred.confidence:.2f};"
                        f"base={opening.source_note}"
                    ),
                )
            )
        elif pred.label in OPENING_CLASSES and pred.confidence >= correction_threshold:
            corrected_openings += 1
            verified_openings.append(
                replace(
                    opening,
                    id=f"micro_vlm_corrected_{opening.id}",
                    kind=pred.label,  # type: ignore[arg-type]
                    source_note=(
                        f"micro_vlm_corrected:{opening.kind}->{pred.label},"
                        f"conf={pred.confidence:.2f};base={opening.source_note}"
                    ),
                )
            )

    return verified_objects, verified_openings, {
        "proposal_objects": len(existing_objects),
        "proposal_openings": len(existing_openings),
        "proposal_object_predictions": len(object_predictions),
        "proposal_opening_predictions": len(opening_predictions),
        "verified_objects": len(verified_objects),
        "verified_openings": len(verified_openings),
        "corrected_objects": corrected_objects,
        "corrected_openings": corrected_openings,
        "verify_threshold": verify_threshold,
        "correction_threshold": correction_threshold,
    }


def detect_micro_vlm_plan_elements(
    crop_png: Path,
    *,
    labels: list[SemanticRoomLabel],
    existing_objects: list[SemanticObject],
    existing_openings: list[SemanticOpening],
    max_seconds: float = 1.4,
) -> tuple[list[SemanticObject], list[SemanticOpening], dict[str, Any]]:
    start = time.perf_counter()
    if os.environ.get("BUILI_PLAN2FIELD_MICRO_VLM_DISABLE", "").lower() in {"1", "true", "yes"}:
        return [], [], {"enabled": False, "reason": "disabled_by_env"}
    model, metadata = _load_model(default_artifact_path())
    if model is None or torch is None or F is None:
        metadata["seconds"] = round(time.perf_counter() - start, 4)
        return [], [], metadata

    image_size = int(metadata["config"].get("image_size", 128))
    patch = int(os.environ.get("BUILI_PLAN2FIELD_MICRO_VLM_PATCH", "224"))
    stride = int(os.environ.get("BUILI_PLAN2FIELD_MICRO_VLM_STRIDE", "176"))
    threshold = float(os.environ.get("BUILI_PLAN2FIELD_MICRO_VLM_CONF", "0.48"))
    verify_threshold = float(os.environ.get("BUILI_PLAN2FIELD_MICRO_VLM_VERIFY_CONF", "0.55"))
    correction_threshold = float(os.environ.get("BUILI_PLAN2FIELD_MICRO_VLM_CORRECT_CONF", "0.82"))
    max_patches = int(os.environ.get("BUILI_PLAN2FIELD_MICRO_VLM_MAX_PATCHES", "96"))
    classes = list(metadata["classes"])
    image = Image.open(crop_png).convert("RGB")
    boxes = _patch_boxes(image.width, image.height, patch=patch, stride=stride)[:max_patches]
    deadline = start + max_seconds
    batch_size = int(os.environ.get("BUILI_PLAN2FIELD_MICRO_VLM_BATCH", "16"))
    accepted_objects, accepted_openings, proposal_metadata = _verify_existing_candidates(
        model,
        image,
        classes=classes,
        image_size=image_size,
        patch=patch,
        batch_size=batch_size,
        existing_objects=existing_objects,
        existing_openings=existing_openings,
        verify_threshold=verify_threshold,
        correction_threshold=correction_threshold,
    )

    predictions: list[MicroVLMPrediction] = []
    for batch_start in range(0, len(boxes), batch_size):
        if time.perf_counter() >= deadline:
            break
        batch_predictions = _model_predictions(
            model,
            image,
            boxes[batch_start : batch_start + batch_size],
            classes=classes,
            image_size=image_size,
            batch_size=batch_size,
            task_id=0,
        )
        predictions.extend(pred for pred in batch_predictions if pred.confidence >= threshold)

    rejected = 0
    for pred in _nms(predictions):
        if _near_existing(pred, [*existing_objects, *accepted_objects], [*existing_openings, *accepted_openings]):
            rejected += 1
            continue
        if labels and _nearest_room_distance(pred.center_px, labels) > 280:
            rejected += 1
            continue
        obj = _prediction_to_object(pred, len(accepted_objects) + 1)
        if obj:
            accepted_objects.append(obj)
            continue
        opening = _prediction_to_opening(pred, len(accepted_openings) + 1)
        if opening:
            accepted_openings.append(opening)

    metadata.update(
        {
            "seconds": round(time.perf_counter() - start, 4),
            "patches_scanned": min(len(boxes), max_patches),
            "grid_raw_predictions": len(predictions),
            "raw_predictions": (
                len(predictions)
                + proposal_metadata["proposal_object_predictions"]
                + proposal_metadata["proposal_opening_predictions"]
            ),
            "objects_added": len(accepted_objects),
            "openings_added": len(accepted_openings),
            "rejected_predictions": rejected,
            "threshold": threshold,
            "patch_px": patch,
            "stride_px": stride,
            "proposal_verification": proposal_metadata,
            "production_role": "primary_lightweight_vlm_plan_patch_parser",
        }
    )
    return accepted_objects, accepted_openings, metadata
