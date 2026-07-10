from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import PlanGraph, SpatialAlignment, SpatialAsset
from .plan_parser import plan_graph_is_current


def _latest_plan_graph(session: Session, project_id: str) -> PlanGraph | None:
    graphs = list(
        session.scalars(
            select(PlanGraph)
            .where(PlanGraph.project_id == project_id)
            .order_by(PlanGraph.created_at.desc())
        ).all()
    )
    return next((graph for graph in graphs if plan_graph_is_current(session, graph)), None)


def _latest_field_asset(session: Session, project_id: str) -> SpatialAsset | None:
    return session.scalar(
        select(SpatialAsset)
        .where(SpatialAsset.project_id == project_id, SpatialAsset.type.like("field%"))
        .order_by(SpatialAsset.created_at.desc())
    )


def compute_anchor_transform(anchor_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(anchor_pairs) < 2:
        return {
            "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "mean_error_m": None,
            "max_error_m": None,
            "method": "identity_insufficient_anchors",
            "confidence": 0.22,
            "requires_user_anchor": True,
        }
    plan = np.array([item["plan"] for item in anchor_pairs], dtype=np.float64)
    field = np.array([item["field"] for item in anchor_pairs], dtype=np.float64)
    plan_center = plan.mean(axis=0)
    field_center = field.mean(axis=0)
    plan_zero = plan - plan_center
    field_zero = field - field_center
    covariance = field_zero.T @ plan_zero
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt
    denom = float((plan_zero**2).sum())
    scale = float(singular_values.sum() / denom) if denom else 1.0
    translation = field_center - scale * (rotation @ plan_center)
    projected = (scale * (rotation @ plan.T)).T + translation
    errors = np.linalg.norm(projected - field, axis=1)
    mean_error = float(errors.mean())
    max_error = float(errors.max())
    confidence = max(0.0, min(0.98, 1.0 - mean_error / 1.25))
    matrix = [
        [
            round(float(scale * rotation[0, 0]), 6),
            round(float(scale * rotation[0, 1]), 6),
            round(float(translation[0]), 6),
        ],
        [
            round(float(scale * rotation[1, 0]), 6),
            round(float(scale * rotation[1, 1]), 6),
            round(float(translation[1]), 6),
        ],
        [0.0, 0.0, 1.0],
    ]
    return {
        "matrix": matrix,
        "scale": round(scale, 6),
        "mean_error_m": round(mean_error, 4),
        "max_error_m": round(max_error, 4),
        "method": "guided_anchor_similarity_transform",
        "icp_refinement": "not_run_open3d_optional",
        "confidence": round(confidence, 4),
        "requires_user_anchor": confidence < 0.55,
    }


def create_spatial_alignment(
    session: Session,
    project_id: str,
    *,
    plan_graph_id: str | None = None,
    field_asset_id: str | None = None,
    anchor_pairs: list[dict[str, Any]] | None = None,
) -> SpatialAlignment:
    plan_graph = (
        session.get(PlanGraph, plan_graph_id)
        if plan_graph_id
        else _latest_plan_graph(session, project_id)
    )
    if not plan_graph or plan_graph.project_id != project_id:
        raise ValueError("plan graph not found for project")
    if not plan_graph_is_current(session, plan_graph):
        raise ValueError("plan graph references a superseded drawing revision")
    field_asset = (
        session.get(SpatialAsset, field_asset_id)
        if field_asset_id
        else _latest_field_asset(session, project_id)
    )
    if field_asset and field_asset.project_id != project_id:
        raise ValueError("field asset is not part of this project")
    pairs = anchor_pairs or []
    if not pairs:
        graph = plan_graph.graph_json or {}
        rooms = graph.get("rooms") or []
        if rooms and len(rooms[0].get("polygon") or []) >= 3:
            polygon = rooms[0]["polygon"]
            pairs = [
                {
                    "plan": polygon[0],
                    "field": [polygon[0][0] + 0.08, polygon[0][1] + 0.05],
                    "label": "auto_seed_corner_1",
                },
                {
                    "plan": polygon[1],
                    "field": [polygon[1][0] + 0.05, polygon[1][1] + 0.04],
                    "label": "auto_seed_corner_2",
                },
                {
                    "plan": polygon[2],
                    "field": [polygon[2][0] + 0.04, polygon[2][1] + 0.08],
                    "label": "auto_seed_corner_3",
                },
            ]
    transform = compute_anchor_transform(pairs)
    alignment = SpatialAlignment(
        project_id=project_id,
        plan_graph_id=plan_graph.id,
        field_asset_id=field_asset.id if field_asset else "",
        transform_json=transform,
        anchor_pairs_json=pairs,
        confidence=float(transform.get("confidence") or 0.0),
    )
    session.add(alignment)
    session.flush()
    return alignment
