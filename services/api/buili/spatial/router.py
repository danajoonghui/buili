from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_session
from ..models import (
    FieldPoseFrame,
    PlanGraph,
    Project,
    SpatialAlignment,
    SpatialAsset,
    SpatialEvidence,
    new_id,
)
from .alignment import create_spatial_alignment
from .compare import compare_project_spatial
from .field_capture import create_field_asset_from_frames, ingest_field_pose_frame
from .geometry import build_design_glb
from .plan_parser import create_plan_graph_record
from .schemas import (
    Design3DRequest,
    FieldPoseFrameCreate,
    FieldPoseFrameOut,
    PlanGraphCreateRequest,
    PlanGraphOut,
    SpatialAlignmentCreate,
    SpatialAlignmentOut,
    SpatialAssetOut,
    SpatialCompareOut,
    SpatialCompareRequest,
    SpatialEvidenceOut,
)

router = APIRouter(prefix="/v1", tags=["spatial"])


def _project_or_404(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    return project


def _latest_plan_graph(session: Session, project_id: str) -> PlanGraph | None:
    return session.scalar(
        select(PlanGraph)
        .where(PlanGraph.project_id == project_id)
        .order_by(PlanGraph.created_at.desc())
    )


@router.post("/projects/{project_id}/spatial/plan-graph", response_model=PlanGraphOut)
def create_plan_graph(
    project_id: str,
    payload: PlanGraphCreateRequest,
    session: Session = Depends(get_session),
) -> PlanGraph:
    project = _project_or_404(session, project_id)
    try:
        graph = create_plan_graph_record(
            session,
            project,
            source_doc_id=payload.source_doc_id,
            preferred_sheet_id=payload.sheet_id,
            calibration_px=payload.calibration_px,
            calibration_m=payload.calibration_m,
            replace_existing=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session.commit()
    session.refresh(graph)
    return graph


@router.get("/projects/{project_id}/spatial/plan-graph", response_model=PlanGraphOut)
def get_plan_graph(project_id: str, session: Session = Depends(get_session)) -> PlanGraph:
    _project_or_404(session, project_id)
    graph = _latest_plan_graph(session, project_id)
    if not graph:
        raise HTTPException(
            status_code=404, detail="plan graph not found; run review or create one"
        )
    return graph


@router.post("/projects/{project_id}/spatial/design-3d", response_model=SpatialAssetOut)
def create_design_3d(
    project_id: str,
    payload: Design3DRequest,
    session: Session = Depends(get_session),
) -> SpatialAsset:
    _project_or_404(session, project_id)
    graph = (
        session.get(PlanGraph, payload.plan_graph_id)
        if payload.plan_graph_id
        else _latest_plan_graph(session, project_id)
    )
    if not graph or graph.project_id != project_id:
        raise HTTPException(status_code=404, detail="plan graph not found")
    if not payload.force:
        existing = session.scalar(
            select(SpatialAsset)
            .where(SpatialAsset.project_id == project_id, SpatialAsset.type == "design_glb")
            .order_by(SpatialAsset.created_at.desc())
        )
        if existing:
            return existing
    asset_id = new_id("spa")
    uri, metadata = build_design_glb(graph.graph_json or {}, project_id, asset_id)
    asset = SpatialAsset(
        id=asset_id,
        project_id=project_id,
        type="design_glb",
        uri=uri,
        metadata_json={**metadata, "plan_graph_id": graph.id},
    )
    session.add(asset)
    session.commit()
    session.refresh(asset)
    return asset


@router.post("/projects/{project_id}/spatial/field-frame", response_model=FieldPoseFrameOut)
def create_field_frame(
    project_id: str,
    payload: FieldPoseFrameCreate,
    session: Session = Depends(get_session),
) -> FieldPoseFrame:
    _project_or_404(session, project_id)
    try:
        frame = ingest_field_pose_frame(
            session,
            project_id,
            media_id=payload.media_id,
            timestamp=payload.timestamp,
            rgb_uri=payload.rgb_uri,
            depth_uri=payload.depth_uri,
            intrinsics_json=payload.intrinsics_json,
            pose_json=payload.pose_json,
            blur_score=payload.blur_score,
            room_hint=payload.room_hint,
        )
        create_field_asset_from_frames(session, project_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session.commit()
    session.refresh(frame)
    return frame


@router.post("/projects/{project_id}/spatial/align", response_model=SpatialAlignmentOut)
def create_alignment(
    project_id: str,
    payload: SpatialAlignmentCreate,
    session: Session = Depends(get_session),
) -> SpatialAlignment:
    _project_or_404(session, project_id)
    try:
        alignment = create_spatial_alignment(
            session,
            project_id,
            plan_graph_id=payload.plan_graph_id,
            field_asset_id=payload.field_asset_id,
            anchor_pairs=[item.model_dump() for item in payload.anchor_pairs],
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not payload.allow_low_confidence and alignment.confidence < 0.5:
        session.rollback()
        raise HTTPException(status_code=409, detail="alignment confidence below threshold")
    session.commit()
    session.refresh(alignment)
    return alignment


@router.post("/projects/{project_id}/spatial/compare", response_model=SpatialCompareOut)
def compare_spatial(
    project_id: str,
    payload: SpatialCompareRequest,
    session: Session = Depends(get_session),
) -> SpatialCompareOut:
    _project_or_404(session, project_id)
    try:
        plan_graph_id, alignment_id, evidence = compare_project_spatial(
            session,
            project_id,
            plan_graph_id=payload.plan_graph_id,
            alignment_id=payload.alignment_id,
            issue_ids=payload.issue_ids,
            update_issue_status=payload.update_issue_status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session.commit()
    for item in evidence:
        session.refresh(item)
    return SpatialCompareOut(
        plan_graph_id=plan_graph_id, alignment_id=alignment_id, evidence=evidence
    )


@router.get("/issues/{issue_id}/spatial", response_model=list[SpatialEvidenceOut])
def get_issue_spatial(
    issue_id: str, session: Session = Depends(get_session)
) -> list[SpatialEvidence]:
    return list(
        session.scalars(
            select(SpatialEvidence)
            .where(SpatialEvidence.issue_id == issue_id)
            .order_by(SpatialEvidence.created_at.desc())
        ).all()
    )


@router.get("/spatial-assets/{asset_id}", response_model=SpatialAssetOut)
def get_spatial_asset(asset_id: str, session: Session = Depends(get_session)) -> SpatialAsset:
    asset = session.get(SpatialAsset, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="spatial asset not found")
    return asset


@router.get("/spatial-assets/{asset_id}/download")
def download_spatial_asset(asset_id: str, session: Session = Depends(get_session)) -> FileResponse:
    asset = session.get(SpatialAsset, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="spatial asset not found")
    path = get_settings().storage_root / Path(asset.uri)
    if not path.exists():
        raise HTTPException(status_code=404, detail="spatial asset file not found")
    media_type = "model/gltf-binary" if path.suffix == ".glb" else "application/json"
    return FileResponse(path, media_type=media_type, filename=path.name)
