from __future__ import annotations

import math
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..models import (
    Issue,
    Observation,
    PlanGraph,
    SiteMedia,
    SpatialAlignment,
    SpatialAsset,
    SpatialEvidence,
)


def _latest_plan_graph(session: Session, project_id: str) -> PlanGraph | None:
    return session.scalar(
        select(PlanGraph)
        .where(PlanGraph.project_id == project_id)
        .order_by(PlanGraph.created_at.desc())
    )


def _latest_alignment(session: Session, project_id: str) -> SpatialAlignment | None:
    return session.scalar(
        select(SpatialAlignment)
        .where(SpatialAlignment.project_id == project_id)
        .order_by(SpatialAlignment.created_at.desc())
    )


def _latest_asset(session: Session, project_id: str, asset_type: str) -> SpatialAsset | None:
    return session.scalar(
        select(SpatialAsset)
        .where(SpatialAsset.project_id == project_id, SpatialAsset.type == asset_type)
        .order_by(SpatialAsset.created_at.desc())
    )


def _room_for_issue(graph: dict[str, Any], issue: Issue) -> dict[str, Any]:
    rooms = graph.get("rooms") or []
    issue_room = (issue.room or "").lower()
    for room in rooms:
        if issue_room and issue_room in str(room.get("name", "")).lower():
            return room
    return rooms[0] if rooms else {"id": "", "name": issue.room, "polygon": []}


def _required_count(issue: Issue, graph: dict[str, Any], room_id: str) -> int:
    title = f"{issue.title} {issue.requirement.get('text', '')}".lower()
    fixtures = graph.get("fixtures") or []
    candidates = [fixture for fixture in fixtures if fixture.get("room_id") == room_id]
    if "afci" in title or "outlet" in title or "gfci" in title:
        outlet_counts = [
            int(fixture.get("required_count") or 1)
            for fixture in candidates
            if "outlet" in str(fixture.get("type", "")).lower()
            or "switch" in str(fixture.get("type", "")).lower()
        ]
        return max(outlet_counts or [2])
    if "smoke" in title:
        return 1
    if "fan" in title or "light" in title:
        return 1
    return max([int(fixture.get("required_count") or 1) for fixture in candidates] or [1])


def _observed_count(observations: list[Observation], issue: Issue) -> int:
    text = f"{issue.title} {issue.observation.get('text', '')}".lower()
    expected_tokens = []
    if "outlet" in text or "gfci" in text or "afci" in text:
        expected_tokens = ["outlet", "receptacle"]
    elif "smoke" in text:
        expected_tokens = ["smoke"]
    elif "fan" in text or "light" in text:
        expected_tokens = ["fan", "light", "switch"]
    elif "panel" in text:
        expected_tokens = ["panel", "equipment"]
    count = 0
    for observation in observations:
        haystack = f"{observation.object_type} {observation.text}".lower()
        if not expected_tokens or any(token in haystack for token in expected_tokens):
            count += 1
    return count


def _distance_to_wall(issue: Issue, room: dict[str, Any]) -> float:
    loc = issue.plan_location or {}
    x = float(loc.get("x") or 0.5)
    y = float(loc.get("y") or 0.5)
    polygon = room.get("polygon") or []
    if len(polygon) < 2:
        return 0.5
    distances = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        vx = end[0] - start[0]
        vy = end[1] - start[1]
        denom = vx * vx + vy * vy
        t = (
            0.0
            if denom == 0
            else max(0.0, min(1.0, ((x - start[0]) * vx + (y - start[1]) * vy) / denom))
        )
        px = start[0] + t * vx
        py = start[1] + t * vy
        distances.append(math.hypot(x - px, y - py))
    return round(min(distances), 3)


def build_geometry_features(
    issue: Issue,
    graph: dict[str, Any],
    observations: list[Observation],
    alignment: SpatialAlignment | None,
    *,
    field_asset_present: bool,
) -> tuple[str, dict[str, Any], str]:
    room = _room_for_issue(graph, issue)
    alignment_confidence = float(alignment.confidence if alignment else 0.0)
    field_coverage_ratio = min(1.0, len(observations) / 5.0) if field_asset_present else 0.0
    required_count = _required_count(issue, graph, str(room.get("id") or ""))
    observed_count = _observed_count(observations, issue)
    visible_in_frame = observed_count > 0 and field_asset_present
    needs_more_evidence = alignment_confidence < 0.5 or field_coverage_ratio < 0.25
    geometry_confidence = max(
        0.0,
        min(
            0.98,
            alignment_confidence * 0.45
            + field_coverage_ratio * 0.2
            + float(issue.confidence or 0.0) * 0.35,
        ),
    )
    distance = _distance_to_wall(issue, room)
    features = {
        "room_alignment_confidence": round(alignment_confidence, 3),
        "field_coverage_ratio": round(field_coverage_ratio, 3),
        "required_count": required_count,
        "observed_count": observed_count,
        "distance_to_required_wall_m": distance,
        "visible_in_frame": visible_in_frame,
        "needs_more_evidence": needs_more_evidence,
        "geometry_confidence": round(geometry_confidence, 3),
        "comparison_mode": "feature_comparison_not_mesh_diff",
        "tolerance_m": 0.5,
    }
    note = (
        f"{room.get('name') or issue.room}: required {required_count}, observed {observed_count}, "
        f"alignment {alignment_confidence:.2f}, field coverage {field_coverage_ratio:.2f}. "
    )
    if needs_more_evidence:
        note += (
            "Spatial evidence is not strong enough for a final defect claim; "
            "request more field evidence."
        )
    else:
        note += "Spatial evidence is review-ready but still requires PM approval."
    return str(room.get("id") or ""), features, note


def compare_project_spatial(
    session: Session,
    project_id: str,
    *,
    plan_graph_id: str | None = None,
    alignment_id: str | None = None,
    issue_ids: list[str] | None = None,
    update_issue_status: bool = True,
) -> tuple[str, str, list[SpatialEvidence]]:
    plan_graph = (
        session.get(PlanGraph, plan_graph_id)
        if plan_graph_id
        else _latest_plan_graph(session, project_id)
    )
    if not plan_graph or plan_graph.project_id != project_id:
        raise ValueError("plan graph not found for project")
    alignment = (
        session.get(SpatialAlignment, alignment_id)
        if alignment_id
        else _latest_alignment(session, project_id)
    )
    if alignment and alignment.project_id != project_id:
        raise ValueError("alignment is not part of this project")
    design_asset = _latest_asset(session, project_id, "design_glb")
    field_asset = _latest_asset(session, project_id, "field_evidence_json")
    issue_query = select(Issue).where(Issue.project_id == project_id)
    if issue_ids:
        issue_query = issue_query.where(Issue.issue_id.in_(issue_ids))
    issues = list(session.scalars(issue_query.order_by(Issue.confidence.desc())).all())
    if not issues:
        return plan_graph.id, alignment.id if alignment else "", []
    session.execute(
        delete(SpatialEvidence).where(
            SpatialEvidence.issue_id.in_([issue.issue_id for issue in issues])
        )
    )
    media_ids = select(SiteMedia.media_id).where(SiteMedia.project_id == project_id)
    observations = list(
        session.scalars(select(Observation).where(Observation.media_id.in_(media_ids))).all()
    )
    evidence_rows: list[SpatialEvidence] = []
    for issue in issues:
        room_id, features, note = build_geometry_features(
            issue,
            plan_graph.graph_json or {},
            observations,
            alignment,
            field_asset_present=field_asset is not None,
        )
        if (
            update_issue_status
            and features.get("needs_more_evidence")
            and issue.status == "review_ready"
        ):
            issue.status = "needs_more_evidence"
        plan_location = dict(issue.plan_location or {})
        plan_location["spatial_context"] = {
            "room_graph_id": room_id,
            "geometry_features": features,
            "alignment_id": alignment.id if alignment else "",
        }
        issue.plan_location = plan_location
        row = SpatialEvidence(
            issue_id=issue.issue_id,
            room_graph_id=room_id,
            design_asset_id=design_asset.id if design_asset else "",
            field_asset_id=field_asset.id if field_asset else "",
            geometry_features_json=features,
            snapshot_uri=design_asset.uri if design_asset else "",
            spatial_note=note,
        )
        session.add(row)
        evidence_rows.append(row)
    session.flush()
    return plan_graph.id, alignment.id if alignment else "", evidence_rows
