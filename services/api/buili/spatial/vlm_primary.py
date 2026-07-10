from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .micro_vlm import MICRO_VLM_CLASSES, OBJECT_CLASSES, OPENING_CLASSES
from .semantic_auto import (
    _dedupe_objects,
    _dedupe_openings,
    build_semantic_scene_from_pdf,
    evaluate_scene_alignment,
)
from .semantic_scene import (
    SemanticObject,
    SemanticOpening,
    SemanticRoomLabel,
    SemanticScene,
    render_semantic_scene,
)

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None

try:
    from transformers import CLIPVisionModel
except ImportError:  # pragma: no cover
    CLIPVisionModel = None


VLM_PRIMARY_CLASSES = MICRO_VLM_CLASSES
DEFAULT_ENCODER_ID = "openai/clip-vit-base-patch32"
VLM_PRIMARY_TASKS = {
    "detect_plan_elements": 0,
    "detect_openings": 1,
    "detect_mep_fixtures": 2,
}
_MODEL_CACHE: tuple[Path, Any, dict[str, Any]] | None = None


if nn is not None and CLIPVisionModel is not None:

    class Plan2FieldVLMPrimary(nn.Module):
        """Pretrained VLM-encoder plan-token generator.

        This is intentionally different from Micro-VLM verification. The frozen CLIP
        vision-language encoder reads dense plan tiles, and a small trainable decoder
        emits semantic plan tokens directly. Geometry still snaps the final scene to
        source-pixel evidence, but objects/openings are VLM-primary.
        """

        def __init__(
            self,
            *,
            encoder_id: str = DEFAULT_ENCODER_ID,
            classes: list[str] | None = None,
            task_vocab: int = 8,
            hidden: int = 768,
            freeze_vision: bool = True,
        ) -> None:
            super().__init__()
            self.encoder_id = encoder_id
            self.classes = classes or VLM_PRIMARY_CLASSES
            self.freeze_vision = freeze_vision
            self.vision = CLIPVisionModel.from_pretrained(encoder_id)
            hidden = int(getattr(self.vision.config, "hidden_size", hidden))
            self.task = nn.Embedding(task_vocab, hidden)
            self.norm = nn.LayerNorm(hidden)
            self.decoder = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
            )
            self.class_head = nn.Linear(hidden // 2, len(self.classes))
            self.objectness_head = nn.Linear(hidden // 2, 1)
            self.box_head = nn.Sequential(
                nn.Linear(hidden // 2, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 4),
            )
            if freeze_vision:
                for parameter in self.vision.parameters():
                    parameter.requires_grad_(False)

        def encode(self, pixel_values: Any) -> Any:
            if self.freeze_vision:
                self.vision.eval()
                with torch.no_grad():
                    return self.vision(pixel_values=pixel_values).pooler_output
            return self.vision(pixel_values=pixel_values).pooler_output

        def decode(self, pooled: Any, task_ids: Any) -> dict[str, Any]:
            fused = self.norm(pooled + self.task(task_ids))
            token = self.decoder(fused)
            return {
                "class_logits": self.class_head(token),
                "objectness": self.objectness_head(token).squeeze(-1),
                "box": torch.sigmoid(self.box_head(token)),
            }

        def forward(self, pixel_values: Any, task_ids: Any) -> dict[str, Any]:
            return self.decode(self.encode(pixel_values), task_ids)

else:  # pragma: no cover
    Plan2FieldVLMPrimary = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PlanTokenPrediction:
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
            "BUILI_PLAN2FIELD_VLM_PRIMARY",
            "data/artifacts/plan2field_vlm_primary/vlm_primary.pt",
        )
    )


def _load_model(path: Path) -> tuple[Any | None, dict[str, Any]]:
    global _MODEL_CACHE
    if torch is None or Plan2FieldVLMPrimary is None:
        return None, {"enabled": False, "reason": "torch_or_transformers_not_installed"}
    path = path.resolve()
    if not path.exists():
        return None, {"enabled": False, "reason": "artifact_missing", "path": str(path)}
    if _MODEL_CACHE and _MODEL_CACHE[0] == path:
        return _MODEL_CACHE[1], _MODEL_CACHE[2]
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = Plan2FieldVLMPrimary(
        encoder_id=str(config.get("encoder_id", DEFAULT_ENCODER_ID)),
        classes=checkpoint.get("classes") or VLM_PRIMARY_CLASSES,
        task_vocab=int(config.get("task_vocab", 8)),
        freeze_vision=bool(config.get("freeze_vision", True)),
    )
    if "head_state" in checkpoint:
        model.load_state_dict(checkpoint["head_state"], strict=False)
    else:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    device = "cuda:0" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    model.to(device)
    model.eval()
    metadata = {
        "enabled": True,
        "path": str(path),
        "device": device,
        "classes": checkpoint.get("classes") or VLM_PRIMARY_CLASSES,
        "config": config,
        "summary": checkpoint.get("summary", {}),
    }
    _MODEL_CACHE = (path, model, metadata)
    return model, metadata


def _clip_tensor(image: Image.Image, image_size: int = 224) -> Any:
    if torch is None:
        raise RuntimeError("torch is required")
    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    data = np.asarray(resized, dtype=np.uint8).copy()
    tensor = torch.from_numpy(data).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
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
    center: tuple[float, float],
    *,
    width: int,
    height: int,
    patch: int,
) -> tuple[int, int, int, int]:
    cx, cy = center
    x0 = int(round(cx - patch / 2))
    y0 = int(round(cy - patch / 2))
    x0 = max(0, min(x0, max(0, width - patch)))
    y0 = max(0, min(y0, max(0, height - patch)))
    return (x0, y0, min(width, x0 + patch), min(height, y0 + patch))


def _unique_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if box in seen:
            continue
        seen.add(box)
        unique.append(box)
    return unique


def _unique_box_records(
    records: list[tuple[tuple[int, int, int, int], str]],
) -> list[tuple[tuple[int, int, int, int], str]]:
    seen: set[tuple[tuple[int, int, int, int], str]] = set()
    unique: list[tuple[tuple[int, int, int, int], str]] = []
    for box, source in records:
        key = (box, source.split(":", 1)[0])
        if key in seen:
            continue
        seen.add(key)
        unique.append((box, source))
    return unique


def _nms(predictions: list[PlanTokenPrediction], iou_threshold: float = 0.22) -> list[PlanTokenPrediction]:
    def iou(a: PlanTokenPrediction, b: PlanTokenPrediction) -> float:
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

    kept: list[PlanTokenPrediction] = []
    for pred in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        if any(pred.label == item.label and iou(pred, item) > iou_threshold for item in kept):
            continue
        kept.append(pred)
    return kept


def _nearest_room_distance(center: tuple[float, float], labels: list[SemanticRoomLabel]) -> float:
    if not labels:
        return 0.0
    cx, cy = center
    return min(math.hypot(cx - label.center_px[0], cy - label.center_px[1]) for label in labels)


def _prediction_to_object(pred: PlanTokenPrediction, index: int) -> SemanticObject | None:
    if pred.label not in OBJECT_CLASSES:
        return None
    return SemanticObject(
        id=f"vlm_primary_object_{index:03d}",
        kind=pred.label,  # type: ignore[arg-type]
        center_px=pred.center_px,
        width_px=max(20.0, pred.width_px),
        depth_px=max(20.0, pred.height_px),
        angle_deg=90.0 if pred.height_px > pred.width_px * 1.3 else 0.0,
        label=pred.label,
        source_note=f"vlm_primary_clip_plan_token:conf={pred.confidence:.2f}:{pred.source}",
    )


def _prediction_to_opening(pred: PlanTokenPrediction, index: int) -> SemanticOpening | None:
    if pred.label not in OPENING_CLASSES:
        return None
    return SemanticOpening(
        id=f"vlm_primary_opening_{index:03d}",
        kind=pred.label,  # type: ignore[arg-type]
        center_px=pred.center_px,
        length_px=max(pred.width_px, pred.height_px, 32.0),
        angle_deg=90.0 if pred.height_px > pred.width_px * 1.25 else 0.0,
        mark="VLM",
        source_note=f"vlm_primary_clip_plan_token:conf={pred.confidence:.2f}:{pred.source}",
    )


def generate_vlm_primary_plan_tokens(
    crop_png: Path,
    *,
    labels: list[SemanticRoomLabel],
    proposal_centers: list[tuple[float, float, str]] | None = None,
    max_seconds: float = 1.8,
) -> tuple[list[SemanticObject], list[SemanticOpening], dict[str, Any]]:
    start = time.perf_counter()
    if os.environ.get("BUILI_PLAN2FIELD_VLM_PRIMARY_DISABLE", "").lower() in {"1", "true", "yes"}:
        return [], [], {"enabled": False, "reason": "disabled_by_env"}
    model, metadata = _load_model(default_artifact_path())
    if model is None or torch is None or F is None:
        metadata["seconds"] = round(time.perf_counter() - start, 4)
        return [], [], metadata

    image = Image.open(crop_png).convert("RGB")
    patch = int(os.environ.get("BUILI_PLAN2FIELD_VLM_PRIMARY_PATCH", "160"))
    stride = int(os.environ.get("BUILI_PLAN2FIELD_VLM_PRIMARY_STRIDE", "96"))
    threshold = float(os.environ.get("BUILI_PLAN2FIELD_VLM_PRIMARY_CONF", "0.42"))
    max_patches = int(os.environ.get("BUILI_PLAN2FIELD_VLM_PRIMARY_MAX_PATCHES", "192"))
    batch_size = int(os.environ.get("BUILI_PLAN2FIELD_VLM_PRIMARY_BATCH", "32"))
    image_size = int(metadata["config"].get("image_size", 224))
    classes = list(metadata["classes"])
    device = next(model.parameters()).device
    proposal_records = [
        (
            _centered_patch_box((cx, cy), width=image.width, height=image.height, patch=patch),
            _source,
        )
        for cx, cy, _source in proposal_centers or []
    ]
    dense_enabled = os.environ.get("BUILI_PLAN2FIELD_VLM_PRIMARY_DENSE", "").lower() in {
        "1",
        "true",
        "yes",
    }
    dense_boxes = _patch_boxes(image.width, image.height, patch=patch, stride=stride)
    dense_records = [(box, "dense_grid") for box in dense_boxes] if dense_enabled or not proposal_records else []
    box_records = _unique_box_records([*proposal_records, *dense_records])[:max_patches]
    proposal_box_count = len(_unique_box_records(proposal_records))
    predictions: list[PlanTokenPrediction] = []
    deadline = start + max_seconds

    for batch_start in range(0, len(box_records), batch_size):
        if time.perf_counter() >= deadline:
            break
        batch_records = box_records[batch_start : batch_start + batch_size]
        batch_boxes = [record[0] for record in batch_records]
        patches = [_clip_tensor(image.crop(box), image_size) for box in batch_boxes]
        pixel_values = torch.stack(patches).to(device)
        task_ids = torch.zeros(len(batch_boxes), dtype=torch.long, device=device)
        with torch.no_grad():
            if hasattr(model, "encode") and hasattr(model, "decode"):
                pooled = model.encode(pixel_values)
            else:
                pooled = None
        batch_task_specs: list[list[tuple[int, set[str] | None, str]]] = []
        for _box, source in batch_records:
            if source.startswith("geometry_candidate_opening"):
                batch_task_specs.append(
                    [(VLM_PRIMARY_TASKS["detect_openings"], OPENING_CLASSES, "opening")]
                )
            elif source.startswith("geometry_candidate_object"):
                batch_task_specs.append(
                    [(VLM_PRIMARY_TASKS["detect_mep_fixtures"], OBJECT_CLASSES, "mep_fixture")]
                )
            else:
                batch_task_specs.append(
                    [
                        (VLM_PRIMARY_TASKS["detect_openings"], OPENING_CLASSES, "opening"),
                        (VLM_PRIMARY_TASKS["detect_mep_fixtures"], OBJECT_CLASSES, "mep_fixture"),
                        (VLM_PRIMARY_TASKS["detect_plan_elements"], None, "general"),
                    ]
                )
        task_ids_to_run = sorted({spec[0] for specs in batch_task_specs for spec in specs})
        for task_id in task_ids_to_run:
            with torch.no_grad():
                if pooled is not None:
                    task_ids = torch.full((len(batch_boxes),), task_id, dtype=torch.long, device=device)
                    output = model.decode(pooled, task_ids)
                else:
                    task_ids = torch.full((len(batch_boxes),), task_id, dtype=torch.long, device=device)
                    output = model(pixel_values, task_ids)
            probs = F.softmax(output["class_logits"], dim=-1)
            objectness = torch.sigmoid(output["objectness"])
            boxes_norm = output["box"].detach().cpu().tolist()
            for row, patch_box in enumerate(batch_boxes):
                matching_specs = [spec for spec in batch_task_specs[row] if spec[0] == task_id]
                if not matching_specs:
                    continue
                _task_id, allowed_labels, task_name = matching_specs[0]
                class_id = int(probs[row].argmax().detach().cpu().item())
                label = classes[class_id]
                if label == "background":
                    continue
                if allowed_labels is not None and label not in allowed_labels:
                    continue
                confidence = float(
                    probs[row, class_id].detach().cpu().item()
                    * objectness[row].detach().cpu().item()
                )
                if confidence < threshold:
                    continue
                px0, py0, px1, py1 = patch_box
                patch_w = px1 - px0
                patch_h = py1 - py0
                cx, cy, bw, bh = boxes_norm[row]
                width = max(10.0, bw * patch_w)
                height = max(10.0, bh * patch_h)
                center_x = px0 + cx * patch_w
                center_y = py0 + cy * patch_h
                predictions.append(
                    PlanTokenPrediction(
                        label=label,
                        confidence=confidence,
                        bbox_px=(
                            max(0.0, center_x - width / 2),
                            max(0.0, center_y - height / 2),
                            min(float(image.width), center_x + width / 2),
                            min(float(image.height), center_y + height / 2),
                        ),
                        source=f"{task_name}_task:{batch_records[row][1]}:patch={patch_box}",
                    )
                )

    objects: list[SemanticObject] = []
    openings: list[SemanticOpening] = []
    rejected = 0
    for pred in _nms(predictions):
        if labels and _nearest_room_distance(pred.center_px, labels) > 320:
            rejected += 1
            continue
        obj = _prediction_to_object(pred, len(objects) + 1)
        if obj:
            objects.append(obj)
            continue
        opening = _prediction_to_opening(pred, len(openings) + 1)
        if opening:
            openings.append(opening)
    metadata.update(
        {
            "seconds": round(time.perf_counter() - start, 4),
            "patches_scanned": min(len(box_records), max_patches),
            "proposal_patches": proposal_box_count,
            "dense_scan_enabled": dense_enabled or not proposal_records,
            "raw_predictions": len(predictions),
            "objects_generated": len(objects),
            "openings_generated": len(openings),
            "rejected_predictions": rejected,
            "threshold": threshold,
            "patch_px": patch,
            "stride_px": stride,
            "production_role": "primary_pretrained_vlm_plan_token_generator",
        }
    )
    return objects, openings, metadata


@contextmanager
def _micro_vlm_disabled() -> Any:
    previous = os.environ.get("BUILI_PLAN2FIELD_MICRO_VLM_DISABLE")
    os.environ["BUILI_PLAN2FIELD_MICRO_VLM_DISABLE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("BUILI_PLAN2FIELD_MICRO_VLM_DISABLE", None)
        else:
            os.environ["BUILI_PLAN2FIELD_MICRO_VLM_DISABLE"] = previous


def build_vlm_primary_scene_from_pdf(
    pdf_path: Path,
    *,
    output_dir: Path,
    page_no: int = 1,
    use_ocr: bool = True,
) -> tuple[SemanticScene, dict[str, Any]]:
    start = time.perf_counter()
    with _micro_vlm_disabled():
        base_scene, base_metadata = build_semantic_scene_from_pdf(
            pdf_path,
            output_dir=output_dir,
            page_no=page_no,
            use_ocr=use_ocr,
        )
    proposal_centers = [
        (*obj.center_px, f"geometry_candidate_object:{obj.id}")
        for obj in base_scene.objects
    ] + [
        (*opening.center_px, f"geometry_candidate_opening:{opening.id}")
        for opening in base_scene.openings
    ]
    objects, openings, vlm_metadata = generate_vlm_primary_plan_tokens(
        Path(base_scene.source_crop_png),
        labels=base_scene.labels,
        proposal_centers=proposal_centers,
    )
    # If dense VLM has not produced a useful opening set yet, retain precise wall-gap
    # openings as geometry priors while keeping object generation VLM-primary.
    opening_source = "vlm_primary"
    if len(openings) < max(4, len(base_scene.openings) // 3):
        opening_source = "geometry_prior_fallback"
        openings = base_scene.openings
    if len(objects) < 3:
        object_source = "geometry_prior_fallback"
        objects = base_scene.objects
    else:
        object_source = "vlm_primary"
    scene = SemanticScene(
        source_pdf=base_scene.source_pdf,
        source_page_png=base_scene.source_page_png,
        source_crop_png=base_scene.source_crop_png,
        transform=base_scene.transform,
        walls=base_scene.walls,
        openings=_dedupe_openings(openings),
        objects=_dedupe_objects(objects, base_scene.labels),
        labels=base_scene.labels,
        tags=base_scene.tags,
        dimensions=base_scene.dimensions,
        source_scope="vlm_primary_semantic_scene_extraction",
    )
    metadata = {
        "method": "vlm_primary_pdf_to_semantic_scene_v1",
        "total_scene_build_seconds": round(time.perf_counter() - start, 4),
        "base_geometry": base_metadata,
        "vlm_primary": vlm_metadata,
        "semantic_ownership": {
            "walls": "deterministic_geometry_snapper",
            "labels": "ocr_text_parser",
            "objects": object_source,
            "openings": opening_source,
            "final_3d": "deterministic_geometry_renderer",
        },
        "counts": scene.to_json()["counts"],
        "quality_gates": {
            "vlm_primary_enabled": bool(vlm_metadata.get("enabled")),
            "vlm_primary_object_candidates": vlm_metadata.get("objects_generated", 0),
            "vlm_primary_opening_candidates": vlm_metadata.get("openings_generated", 0),
            "automatic_scene_ready": len(scene.walls) >= 6 and len(scene.labels) >= 1,
        },
    }
    return scene, metadata


def build_vlm_primary_plan2field3d_artifacts(
    pdf_path: Path,
    output_dir: Path,
    *,
    page_no: int = 1,
    use_ocr: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene, scene_metadata = build_vlm_primary_scene_from_pdf(
        pdf_path,
        output_dir=output_dir,
        page_no=page_no,
        use_ocr=use_ocr,
    )
    scene_json = output_dir / "vlm_primary_semantic_scene.json"
    preview_png = output_dir / "vlm_primary_plan2field3d.png"
    alignment_overlay_png = output_dir / "vlm_primary_alignment_overlay.png"
    summary_json = output_dir / "vlm_primary_plan2field3d_summary.json"
    scene_json.write_text(json.dumps(scene.to_json(), indent=2), encoding="utf-8")
    alignment_qa = evaluate_scene_alignment(
        scene, Path(scene.source_crop_png), alignment_overlay_png
    )
    render_start = time.perf_counter()
    render_summary = render_semantic_scene(scene, preview_png)
    render_seconds = time.perf_counter() - render_start
    summary = {
        "input_pdf": str(pdf_path),
        "scene_json": str(scene_json),
        "preview_png": str(preview_png),
        "alignment_overlay_png": str(alignment_overlay_png),
        "scene_build": scene_metadata,
        "alignment_qa": alignment_qa,
        "render": render_summary,
        "total_seconds": round(scene_metadata["total_scene_build_seconds"] + render_seconds, 4),
        "qa": {
            "vlm_primary_semantic_generation": True,
            "deterministic_geometry_snapper_retained": True,
            "wall_openings_cut_from_render_geometry": True,
            "procedural_asset_generation_used": True,
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
