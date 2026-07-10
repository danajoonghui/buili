from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.buili.spatial.eval_metrics import evaluate_plan_elements
from services.api.buili.spatial.floorplan_extractor import _segments_from_image
from services.api.buili.spatial.semantic_auto import (
    _dedupe_openings,
    _filter_floorplan_segments,
    _openings_from_wall_gaps,
    _snap_segments_to_dark_evidence,
    _walls_from_segments,
    _windows_from_colored_markers,
)
from services.api.buili.spatial.semantic_scene import SemanticObject, SemanticOpening, SemanticWall
from services.api.buili.spatial.vlm_primary import generate_vlm_primary_plan_tokens


YOLO_TO_BUILI = {
    "wall": "wall",
    "door": "door",
    "window": "window",
    "bathtub": "bathtub",
    "cabinet_run": "cabinet_run",
    "column": "column",
    "fixture": "fixture",
    "shower": "shower",
    "water_heater": "water_heater",
    "toilet": "toilet",
    "sink": "sink",
    "refrigerator": "cabinet_run",
    "oven": "cabinet_run",
    "microwave": "cabinet_run",
    "bed": "fixture",
    "chair": "fixture",
    "couch": "fixture",
    "dining table": "fixture",
}
_YOLO_MODEL_CACHE: dict[str, Any] = {}


def _tile_origins(width: int, height: int, tile: int, stride: int) -> list[tuple[int, int]]:
    xs = list(range(0, max(1, width - tile + 1), stride))
    ys = list(range(0, max(1, height - tile + 1), stride))
    if not xs or xs[-1] + tile < width:
        xs.append(max(0, width - tile))
    if not ys or ys[-1] + tile < height:
        ys.append(max(0, height - tile))
    return [(x, y) for y in ys for x in xs]


def _bbox_iou(left: list[float], right: list[float]) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    la = max((lx1 - lx0) * (ly1 - ly0), 1e-6)
    ra = max((rx1 - rx0) * (ry1 - ry0), 1e-6)
    return inter / (la + ra - inter)


def _nms_rows(rows: list[dict[str, Any]], *, iou_threshold: float = 0.35) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        if any(
            row.get("kind") == item.get("kind")
            and _bbox_iou(row.get("bbox", [0, 0, 0, 0]), item.get("bbox", [0, 0, 0, 0]))
            > iou_threshold
            for item in kept
        ):
            continue
        kept.append(row)
    return kept


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _objects_to_rows(objects: list[SemanticObject]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obj in objects:
        cx, cy = obj.center_px
        rows.append(
            {
                "id": obj.id,
                "kind": str(obj.kind),
                "bbox": [
                    cx - obj.width_px / 2,
                    cy - obj.depth_px / 2,
                    cx + obj.width_px / 2,
                    cy + obj.depth_px / 2,
                ],
                "angle_deg": obj.angle_deg,
                "length_px": max(obj.width_px, obj.depth_px),
            }
        )
    return rows


def _openings_to_rows(openings: list[SemanticOpening]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for opening in openings:
        cx, cy = opening.center_px
        width = opening.length_px if abs(opening.angle_deg) < 45 else 24.0
        height = opening.length_px if abs(opening.angle_deg) >= 45 else 24.0
        rows.append(
            {
                "id": opening.id,
                "kind": str(opening.kind),
                "bbox": [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
                "angle_deg": opening.angle_deg,
                "length_px": opening.length_px,
            }
        )
    return rows


def _walls_to_rows(walls: list[SemanticWall]) -> list[dict[str, Any]]:
    return [
        {
            "id": wall.id,
            "kind": "wall",
            "segment": [
                float(wall.start_px[0]),
                float(wall.start_px[1]),
                float(wall.end_px[0]),
                float(wall.end_px[1]),
            ],
            "wall_type": str(wall.wall_type),
        }
        for wall in walls
    ]


def _prediction_payload(
    *,
    walls: list[SemanticWall],
    openings: list[SemanticOpening],
    objects: list[SemanticObject],
) -> dict[str, Any]:
    return {
        "walls": _walls_to_rows(walls),
        "openings": _openings_to_rows(openings),
        "objects": _objects_to_rows(objects),
    }


def deterministic_image_baseline(image_path: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    segments, wall_metadata = _segments_from_image(rgb)
    floorplan_segments = _filter_floorplan_segments(segments, rgb.shape[1], rgb.shape[0])
    snapped = _snap_segments_to_dark_evidence(image_path, floorplan_segments)
    walls = _walls_from_segments(snapped, crop_width=rgb.shape[1], crop_height=rgb.shape[0])
    openings = _dedupe_openings(
        [
            *_openings_from_wall_gaps(snapped, crop_width=rgb.shape[1], crop_height=rgb.shape[0]),
            *_windows_from_colored_markers(image_path, walls),
        ]
    )
    payload = _prediction_payload(walls=walls, openings=openings, objects=[])
    metadata = {
        "method": "deterministic_only_image_cv",
        "seconds": round(time.perf_counter() - start, 4),
        "image_path": str(image_path),
        "wall_metadata": wall_metadata,
        "counts": {
            "walls": len(payload["walls"]),
            "openings": len(payload["openings"]),
            "objects": len(payload["objects"]),
        },
        "preserves_existing_pdf_pipeline": True,
    }
    return payload, metadata


@contextlib.contextmanager
def _env(**updates: str | None) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def vlm_primary_image_variant(
    image_path: Path,
    deterministic_payload: dict[str, Any],
    *,
    use_proposals: bool,
    dense_scan: bool,
    max_seconds: float,
    artifact: str | None = None,
    patch: int | None = None,
    stride: int | None = None,
    confidence: float | None = None,
    max_patches: int | None = None,
    batch_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()
    proposals: list[tuple[float, float, str]] = []
    if use_proposals:
        for row in deterministic_payload.get("objects", []):
            x0, y0, x1, y1 = row["bbox"]
            proposals.append(((x0 + x1) / 2, (y0 + y1) / 2, f"geometry_candidate_object:{row['id']}"))
        for row in deterministic_payload.get("openings", []):
            x0, y0, x1, y1 = row["bbox"]
            proposals.append(((x0 + x1) / 2, (y0 + y1) / 2, f"geometry_candidate_opening:{row['id']}"))
    with _env(
        BUILI_PLAN2FIELD_VLM_PRIMARY_DENSE="1" if dense_scan else None,
        BUILI_PLAN2FIELD_MICRO_VLM_DISABLE="1",
        BUILI_PLAN2FIELD_VLM_PRIMARY=artifact,
        BUILI_PLAN2FIELD_VLM_PRIMARY_PATCH=str(patch) if patch else None,
        BUILI_PLAN2FIELD_VLM_PRIMARY_STRIDE=str(stride) if stride else None,
        BUILI_PLAN2FIELD_VLM_PRIMARY_CONF=str(confidence) if confidence is not None else None,
        BUILI_PLAN2FIELD_VLM_PRIMARY_MAX_PATCHES=str(max_patches) if max_patches else None,
        BUILI_PLAN2FIELD_VLM_PRIMARY_BATCH=str(batch_size) if batch_size else None,
    ):
        objects, openings, vlm_metadata = generate_vlm_primary_plan_tokens(
            image_path,
            labels=[],
            proposal_centers=proposals if use_proposals else [],
            max_seconds=max_seconds,
        )
    payload = {
        "walls": deterministic_payload.get("walls", []),
        "openings": _openings_to_rows(openings) or deterministic_payload.get("openings", []),
        "objects": _objects_to_rows(objects),
    }
    metadata = {
        "method": "vlm_primary_image_proposal_ablation",
        "variant": "proposal_guided" if use_proposals else "dense_without_proposals",
        "dense_scan": dense_scan,
        "proposal_count": len(proposals),
        "seconds": round(time.perf_counter() - start, 4),
        "vlm": vlm_metadata,
        "counts": {
            "walls": len(payload["walls"]),
            "openings": len(payload["openings"]),
            "objects": len(payload["objects"]),
        },
    }
    return payload, metadata


def vlm_primary_payload_guided_variant(
    image_path: Path,
    deterministic_payload: dict[str, Any],
    proposal_payload: dict[str, Any],
    *,
    proposal_name: str,
    max_seconds: float,
    artifact: str | None = None,
    patch: int | None = None,
    stride: int | None = None,
    confidence: float | None = None,
    max_patches: int | None = None,
    batch_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()
    proposals: list[tuple[float, float, str]] = []
    for row in proposal_payload.get("objects", []):
        x0, y0, x1, y1 = row["bbox"]
        proposals.append(((x0 + x1) / 2, (y0 + y1) / 2, f"geometry_candidate_object:{proposal_name}:{row['id']}"))
    for row in proposal_payload.get("openings", []):
        x0, y0, x1, y1 = row["bbox"]
        proposals.append(((x0 + x1) / 2, (y0 + y1) / 2, f"geometry_candidate_opening:{proposal_name}:{row['id']}"))
    with _env(
        BUILI_PLAN2FIELD_VLM_PRIMARY_DENSE=None,
        BUILI_PLAN2FIELD_MICRO_VLM_DISABLE="1",
        BUILI_PLAN2FIELD_VLM_PRIMARY=artifact,
        BUILI_PLAN2FIELD_VLM_PRIMARY_PATCH=str(patch) if patch else None,
        BUILI_PLAN2FIELD_VLM_PRIMARY_STRIDE=str(stride) if stride else None,
        BUILI_PLAN2FIELD_VLM_PRIMARY_CONF=str(confidence) if confidence is not None else None,
        BUILI_PLAN2FIELD_VLM_PRIMARY_MAX_PATCHES=str(max_patches) if max_patches else None,
        BUILI_PLAN2FIELD_VLM_PRIMARY_BATCH=str(batch_size) if batch_size else None,
    ):
        objects, openings, vlm_metadata = generate_vlm_primary_plan_tokens(
            image_path,
            labels=[],
            proposal_centers=proposals,
            max_seconds=max_seconds,
        )
    payload = {
        "walls": deterministic_payload.get("walls", []),
        "openings": _openings_to_rows(openings),
        "objects": _objects_to_rows(objects),
    }
    metadata = {
        "method": "vlm_primary_payload_guided_ablation",
        "proposal_source": proposal_name,
        "dense_scan": False,
        "proposal_count": len(proposals),
        "seconds": round(time.perf_counter() - start, 4),
        "vlm": vlm_metadata,
        "counts": {
            "walls": len(payload["walls"]),
            "openings": len(payload["openings"]),
            "objects": len(payload["objects"]),
        },
        "fairness_note": "No TilePlanDet fallback rows are copied into this ablation; counted detections are VLM-generated.",
    }
    return payload, metadata


def yolo_baseline(
    image_path: Path,
    *,
    weights: str,
    confidence: float = 0.25,
    method_name: str = "yolo_floorplan_baseline",
) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()
    try:
        from ultralytics import YOLO
    except ImportError:
        return (
            {"walls": [], "openings": [], "objects": []},
            {
                "method": method_name,
                "enabled": False,
                "reason": "ultralytics_not_installed",
                "weights": weights,
                "seconds": round(time.perf_counter() - start, 4),
            },
        )
    if weights not in _YOLO_MODEL_CACHE:
        _YOLO_MODEL_CACHE[weights] = YOLO(weights)
    model = _YOLO_MODEL_CACHE[weights]
    device = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    result = model.predict(str(image_path), conf=confidence, verbose=False, device=device)[0]
    names = result.names
    walls: list[dict[str, Any]] = []
    openings: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    for index, box in enumerate(result.boxes):
        cls_id = int(box.cls.detach().cpu().item())
        name = str(names.get(cls_id, cls_id))
        kind = YOLO_TO_BUILI.get(name)
        if not kind:
            continue
        x0, y0, x1, y1 = [float(value) for value in box.xyxy.detach().cpu().numpy()[0]]
        width = x1 - x0
        height = y1 - y0
        score = float(box.conf.detach().cpu().item())
        if kind == "wall":
            if width >= height:
                segment = [x0, (y0 + y1) / 2, x1, (y0 + y1) / 2]
            else:
                segment = [(x0 + x1) / 2, y0, (x0 + x1) / 2, y1]
            walls.append(
                {
                    "id": f"yolo_wall_{index:04d}",
                    "kind": "wall",
                    "segment": segment,
                    "bbox": [x0, y0, x1, y1],
                    "score": score,
                    "source_class": name,
                }
            )
        elif kind in {"door", "window"}:
            openings.append(
                {
                    "id": f"yolo_opening_{index:04d}",
                    "kind": kind,
                    "bbox": [x0, y0, x1, y1],
                    "angle_deg": 0.0 if width >= height else 90.0,
                    "length_px": max(width, height),
                    "score": score,
                    "source_class": name,
                }
            )
        else:
            objects.append(
                {
                    "id": f"yolo_object_{index:04d}",
                    "kind": kind,
                    "bbox": [x0, y0, x1, y1],
                    "angle_deg": 0.0,
                    "length_px": max(width, height),
                    "score": score,
                    "source_class": name,
                }
            )
    payload = {"walls": walls, "openings": openings, "objects": objects}
    metadata = {
        "method": method_name,
        "enabled": True,
        "weights": weights,
        "seconds": round(time.perf_counter() - start, 4),
        "mapped_walls": len(walls),
        "mapped_openings": len(openings),
        "mapped_objects": len(objects),
        "raw_detections": int(len(result.boxes)),
        "note": (
            "TinyPlanDet-VectorSnap when trained floorplan weights are supplied; "
            "off-the-shelf YOLO baseline otherwise."
        ),
    }
    return payload, metadata


def tiled_yolo_vectorsnap(
    image_path: Path,
    *,
    weights: str,
    tile: int = 768,
    stride: int = 512,
    confidence: float = 0.05,
    wall_payload: list[dict[str, Any]] | None = None,
    method_name: str = "tileplandet_vectorsnap",
) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()
    try:
        from ultralytics import YOLO
    except ImportError:
        return (
            {"walls": [], "openings": [], "objects": []},
            {
                "method": method_name,
                "enabled": False,
                "reason": "ultralytics_not_installed",
                "weights": weights,
                "seconds": round(time.perf_counter() - start, 4),
            },
        )
    if weights not in _YOLO_MODEL_CACHE:
        _YOLO_MODEL_CACHE[weights] = YOLO(weights)
    model = _YOLO_MODEL_CACHE[weights]
    image = Image.open(image_path).convert("RGB")
    origins = _tile_origins(image.width, image.height, tile, stride)
    crops = [image.crop((x, y, x + tile, y + tile)) for x, y in origins]
    device = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    results = model.predict(
        crops,
        conf=confidence,
        verbose=False,
        device=device,
        imgsz=tile,
        max_det=600,
    )
    walls: list[dict[str, Any]] = []
    openings: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    raw = 0
    for result, (ox, oy) in zip(results, origins, strict=False):
        names = result.names
        for box in result.boxes:
            raw += 1
            cls_id = int(box.cls.detach().cpu().item())
            name = str(names.get(cls_id, cls_id))
            kind = YOLO_TO_BUILI.get(name)
            if not kind:
                continue
            x0, y0, x1, y1 = [float(value) for value in box.xyxy.detach().cpu().numpy()[0]]
            x0 += ox
            x1 += ox
            y0 += oy
            y1 += oy
            width = x1 - x0
            height = y1 - y0
            score = float(box.conf.detach().cpu().item())
            if kind == "wall":
                if width >= height:
                    segment = [x0, (y0 + y1) / 2, x1, (y0 + y1) / 2]
                else:
                    segment = [(x0 + x1) / 2, y0, (x0 + x1) / 2, y1]
                walls.append(
                    {
                        "id": f"tile_wall_{raw:05d}",
                        "kind": "wall",
                        "segment": segment,
                        "bbox": [x0, y0, x1, y1],
                        "score": score,
                        "source_class": name,
                    }
                )
            elif kind in {"door", "window"}:
                openings.append(
                    {
                        "id": f"tile_opening_{raw:05d}",
                        "kind": kind,
                        "bbox": [x0, y0, x1, y1],
                        "angle_deg": 0.0 if width >= height else 90.0,
                        "length_px": max(width, height),
                        "score": score,
                        "source_class": name,
                    }
                )
            else:
                objects.append(
                    {
                        "id": f"tile_object_{raw:05d}",
                        "kind": kind,
                        "bbox": [x0, y0, x1, y1],
                        "angle_deg": 0.0,
                        "length_px": max(width, height),
                        "score": score,
                        "source_class": name,
                    }
                )
    detector_walls = _nms_rows(walls, iou_threshold=0.25)
    payload = {
        "walls": wall_payload if wall_payload is not None else detector_walls,
        "openings": _nms_rows(openings, iou_threshold=0.35),
        "objects": _nms_rows(objects, iou_threshold=0.35),
    }
    metadata = {
        "method": method_name,
        "enabled": True,
        "weights": weights,
        "tile": tile,
        "stride": stride,
        "confidence": confidence,
        "tiles": len(origins),
        "raw_detections": raw,
        "seconds": round(time.perf_counter() - start, 4),
        "wall_source": "deterministic_vector_snap" if wall_payload is not None else "detector_bbox_snap",
        "detector_wall_count": len(detector_walls),
        "counts": {
            "walls": len(payload["walls"]),
            "openings": len(payload["openings"]),
            "objects": len(payload["objects"]),
        },
    }
    return payload, metadata


def _empty_metrics() -> dict[str, Any]:
    return {
        "objects": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "true_positive": 0},
        "openings": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "true_positive": 0},
        "walls": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "true_positive": 0},
    }


def _aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    subset = [row for row in rows if row["variant"] == variant]
    if not subset:
        return {"variant": variant, "samples": 0, **_empty_metrics()}
    summary: dict[str, Any] = {"variant": variant, "samples": len(subset)}
    for group in ("objects", "openings", "walls"):
        f1 = [row["metrics"][group]["f1"] for row in subset]
        precision = [row["metrics"][group]["precision"] for row in subset]
        recall = [row["metrics"][group]["recall"] for row in subset]
        tp = [row["metrics"][group].get("true_positive", 0) for row in subset]
        summary[group] = {
            "mean_precision": round(float(np.mean(precision)), 4),
            "mean_recall": round(float(np.mean(recall)), 4),
            "mean_f1": round(float(np.mean(f1)), 4),
            "true_positive_sum": int(sum(tp)),
        }
    summary["mean_seconds"] = round(float(np.mean([row["metadata"].get("seconds", 0.0) for row in subset])), 4)
    return summary


def evaluate_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    yolo_weights: str = "data/artifacts/yolo/yolo11n.pt",
    tinyplandet_weights: str | None = None,
    tileplandet_weights: str | None = None,
    run_yolo: bool = False,
    run_tinyplandet: bool = False,
    run_tileplandet: bool = False,
    run_vlm: bool = True,
    max_vlm_seconds: float = 1.2,
    vlm_artifact: str | None = None,
    vlm_patch: int | None = None,
    vlm_stride: int | None = None,
    vlm_confidence: float | None = None,
    vlm_max_patches: int | None = None,
    vlm_batch_size: int | None = None,
    tile_confidence: float = 0.03,
    tile_stride: int = 512,
    tile_with_deterministic_walls: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_manifest(manifest_path)
    result_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

    for sample in rows:
        image_path = Path(sample["image_path"])
        ground_truth = json.loads(Path(sample["ground_truth_path"]).read_text(encoding="utf-8"))
        deterministic_payload, deterministic_metadata = deterministic_image_baseline(image_path, output_dir)
        variants: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
            ("deterministic_only", deterministic_payload, deterministic_metadata)
        ]
        if run_yolo:
            yolo_payload, yolo_metadata = yolo_baseline(
                image_path,
                weights=yolo_weights,
                method_name="off_the_shelf_yolo_baseline",
            )
            variants.append(("yolo_baseline", yolo_payload, yolo_metadata))
        if run_tinyplandet and tinyplandet_weights:
            tiny_payload, tiny_metadata = yolo_baseline(
                image_path,
                weights=tinyplandet_weights,
                confidence=0.15,
                method_name="tinyplandet_vectorsnap",
            )
            variants.append(("tinyplandet_vectorsnap", tiny_payload, tiny_metadata))
        if run_tileplandet and tileplandet_weights:
            tile_payload, tile_metadata = tiled_yolo_vectorsnap(
                image_path,
                weights=tileplandet_weights,
                stride=tile_stride,
                confidence=tile_confidence,
                wall_payload=deterministic_payload.get("walls", [])
                if tile_with_deterministic_walls
                else None,
            )
            variants.append(("tileplandet_vectorsnap", tile_payload, tile_metadata))
        if run_vlm:
            if run_tileplandet and tileplandet_weights:
                tile_vlm_payload, tile_vlm_metadata = vlm_primary_payload_guided_variant(
                    image_path,
                    deterministic_payload,
                    tile_payload,
                    proposal_name="tileplandet",
                    max_seconds=max_vlm_seconds,
                    artifact=vlm_artifact,
                    patch=vlm_patch,
                    stride=vlm_stride,
                    confidence=vlm_confidence,
                    max_patches=vlm_max_patches,
                    batch_size=vlm_batch_size,
                )
                variants.append(("tileplandet_vlm_guided", tile_vlm_payload, tile_vlm_metadata))
            guided_payload, guided_metadata = vlm_primary_image_variant(
                image_path,
                deterministic_payload,
                use_proposals=True,
                dense_scan=False,
                max_seconds=max_vlm_seconds,
                artifact=vlm_artifact,
                patch=vlm_patch,
                stride=vlm_stride,
                confidence=vlm_confidence,
                max_patches=vlm_max_patches,
                batch_size=vlm_batch_size,
            )
            dense_payload, dense_metadata = vlm_primary_image_variant(
                image_path,
                deterministic_payload,
                use_proposals=False,
                dense_scan=True,
                max_seconds=max_vlm_seconds,
                artifact=vlm_artifact,
                patch=vlm_patch,
                stride=vlm_stride,
                confidence=vlm_confidence,
                max_patches=vlm_max_patches,
                batch_size=vlm_batch_size,
            )
            variants.extend(
                [
                    ("vlm_primary_proposal_guided", guided_payload, guided_metadata),
                    ("vlm_primary_dense_no_guidance", dense_payload, dense_metadata),
                ]
            )
        for variant, payload, metadata in variants:
            metrics = evaluate_plan_elements(payload, ground_truth)
            result_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "sample_index": sample["sample_index"],
                    "variant": variant,
                    "metrics": metrics,
                    "metadata": metadata,
                    "counts": {
                        "pred": {
                            "walls": len(payload.get("walls", [])),
                            "openings": len(payload.get("openings", [])),
                            "objects": len(payload.get("objects", [])),
                        },
                        "gt": ground_truth.get("counts", {}),
                    },
                }
            )

    detail_path = output_dir / "eval50_results.jsonl"
    summary_path = output_dir / "eval50_summary.json"
    detail_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in result_rows) + "\n",
        encoding="utf-8",
    )
    variants = sorted({row["variant"] for row in result_rows})
    summary = {
        "manifest_path": str(manifest_path),
        "samples": len(rows),
        "detail_path": str(detail_path),
        "seconds": round(time.perf_counter() - start, 4),
        "gpu_policy": "CUDA_VISIBLE_DEVICES=7 for VLM/YOLO calls",
        "baselines": {
            "deterministic_only": "OpenCV wall/opening geometry, Micro-VLM disabled, no existing path removed.",
            "yolo_baseline": "Optional Ultralytics YOLO adapter; off-the-shelf or supplied floorplan weights.",
            "tinyplandet_vectorsnap": (
                "Trained lightweight floorplan detector whose boxes are converted into "
                "wall centerlines, openings, and fixture primitives."
            ),
            "tileplandet_vectorsnap": (
                "High-resolution tiled lightweight detector for openings/objects, "
                "class-wise NMS, and deterministic wall vector snapping for thin "
                "line structures."
            ),
            "tileplandet_vlm_guided": (
                "TilePlanDet supplies object/opening proposal centers; CLIP VLM "
                "domain heads regenerate counted semantic tokens without copying "
                "TilePlanDet detections as fallback."
            ),
            "vlm_primary_proposal_guided": "CLIP VLM plan-token head with deterministic proposal patches.",
            "vlm_primary_dense_no_guidance": "Same VLM head with dense scan and no proposal centers.",
        },
        "metrics": {
            "object": "class-aware bbox IoU@0.50 precision/recall/F1 plus center error",
            "opening": "door/window bbox IoU@0.35 plus center/angle/length error",
            "wall": "segment distance <= 12 px and angular gate <= 12 deg plus GT sample coverage",
        },
        "tileplandet_config": {
            "confidence": tile_confidence,
            "stride": tile_stride,
            "wall_source": "deterministic_vector_snap"
            if tile_with_deterministic_walls
            else "detector_bbox_snap",
        },
        "vlm_primary_config": {
            "artifact": vlm_artifact or "default",
            "patch": vlm_patch or "default",
            "stride": vlm_stride or "default",
            "confidence": vlm_confidence if vlm_confidence is not None else "default",
            "max_patches": vlm_max_patches or "default",
            "batch_size": vlm_batch_size or "default",
            "max_seconds": max_vlm_seconds,
        },
        "aggregate": [_aggregate(result_rows, variant) for variant in variants],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/eval/plan2field_cubicasa50/manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/plan2field3d_eval50"),
    )
    parser.add_argument("--run-yolo", action="store_true")
    parser.add_argument("--run-tinyplandet", action="store_true")
    parser.add_argument("--run-tileplandet", action="store_true")
    parser.add_argument("--no-vlm", action="store_true")
    parser.add_argument("--yolo-weights", default="data/artifacts/yolo/yolo11n.pt")
    parser.add_argument("--tinyplandet-weights", default="")
    parser.add_argument("--tileplandet-weights", default="")
    parser.add_argument("--max-vlm-seconds", type=float, default=1.2)
    parser.add_argument("--vlm-artifact", default="")
    parser.add_argument("--vlm-patch", type=int, default=0)
    parser.add_argument("--vlm-stride", type=int, default=0)
    parser.add_argument("--vlm-confidence", type=float, default=-1.0)
    parser.add_argument("--vlm-max-patches", type=int, default=0)
    parser.add_argument("--vlm-batch-size", type=int, default=0)
    parser.add_argument("--tile-confidence", type=float, default=0.03)
    parser.add_argument("--tile-stride", type=int, default=512)
    parser.add_argument("--tile-detector-walls-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_manifest(
                args.manifest,
                args.output_dir,
                yolo_weights=args.yolo_weights,
                tinyplandet_weights=args.tinyplandet_weights or None,
                tileplandet_weights=args.tileplandet_weights or None,
                run_yolo=args.run_yolo,
                run_tinyplandet=args.run_tinyplandet,
                run_tileplandet=args.run_tileplandet,
                run_vlm=not args.no_vlm,
                max_vlm_seconds=args.max_vlm_seconds,
                vlm_artifact=args.vlm_artifact or None,
                vlm_patch=args.vlm_patch or None,
                vlm_stride=args.vlm_stride or None,
                vlm_confidence=args.vlm_confidence if args.vlm_confidence >= 0 else None,
                vlm_max_patches=args.vlm_max_patches or None,
                vlm_batch_size=args.vlm_batch_size or None,
                tile_confidence=args.tile_confidence,
                tile_stride=args.tile_stride,
                tile_with_deterministic_walls=not args.tile_detector_walls_only,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
