from __future__ import annotations

import base64
import binascii
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session, selectinload

from .config import get_settings
from .auth import current_principal
from .database import get_session
from .models import (
    AuditEvent,
    DirectoryMember,
    Document,
    DocumentRevision,
    EvidenceLink,
    FieldEvidence,
    Issue,
    IssueEvidence,
    IssueWorkflow,
    Notification,
    PlanEntity,
    PlanGraph,
    Project,
    ReportRecord,
    ReportScope,
    ReportVersion,
    ReviewRecord,
    Sheet,
    SiteMedia,
    SpecChunk,
    SpatialAlignment,
    SpatialAsset,
    SpatialEvidence,
)
from .schemas import (
    DirectoryCreate,
    DirectoryPatch,
    EvidenceLinkRequest,
    EvidenceLocationPatch,
    EvidencePatch,
    EvidenceSyncRequest,
    IssueCreate,
    ProjectPatch,
    ProjectSettingsPatch,
    ReportExportRequest,
    RequestEvidenceCreate,
    ReviewCreate,
    RevisionActivateRequest,
)
from .reports import build_report
from .spatial.geometry import build_design_glb
from .spatial.plan_parser import create_plan_graph_record, plan_graph_provenance
from .storage import object_path
from .workflows import (
    APPROVER_ROLES,
    EDIT_ROLES,
    EXPORT_ROLES,
    actor_context,
    get_or_create_issue_workflow,
    get_or_create_project_profile,
    get_or_create_revision,
    issue_action_blockers,
    issue_snapshot,
    project_snapshot,
    record_audit,
    require_role,
    sha256_path,
    source_snapshot_for_issue,
    utcnow,
)

router = APIRouter(prefix="/v1", tags=["product-workflows"])

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav"}
MEDIA_EXTENSIONS = PHOTO_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
MIME_PREFIX = {"photo": "image/", "video": "video/", "audio": "audio/"}
OPEN_ISSUE_STATUSES = {
    "draft",
    "review_ready",
    "in_review",
    "approved",
    "rejected",
    "needs_more_evidence",
    "stale_source_review",
    "stale_evidence_review",
}


def _project_or_404(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    return project


def _issue_or_404(session: Session, issue_id: str) -> Issue:
    issue = session.scalar(
        select(Issue)
        .options(selectinload(Issue.evidence), selectinload(Issue.spatial_evidence))
        .where(Issue.issue_id == issue_id)
    )
    if not issue:
        raise HTTPException(status_code=404, detail="issue not found")
    return issue


def _directory_dict(item: DirectoryMember) -> dict[str, Any]:
    return {
        "directory_id": item.directory_id,
        "project_id": item.project_id,
        "person_name": item.person_name,
        "email": item.email,
        "company": item.company,
        "role": item.role,
        "trade": item.trade,
        "status": item.status,
        "notification": dict(item.notification_json or {}),
        "access_expires_at": item.access_expires_at,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


def _revision_dict(document: Document, revision: DocumentRevision) -> dict[str, Any]:
    return {
        "revision_id": revision.revision_id,
        "document_id": document.doc_id,
        "project_id": document.project_id,
        "logical_key": revision.logical_key,
        "sheet_number": revision.sheet_number,
        "filename": document.filename,
        "document_type": document.type,
        "revision": revision.revision or document.revision,
        "issue_date": revision.issue_date,
        "discipline": revision.discipline,
        "state": revision.state,
        "is_current": revision.state == "current",
        "supersedes_document_id": revision.supersedes_document_id,
        "source_hash": revision.source_hash or document.hash,
        "parsed_status": document.parsed_status,
        "activated_at": revision.activated_at,
        "created_at": document.created_at,
    }


def _evidence_dict(item: FieldEvidence, *, deduplicated: bool = False) -> dict[str, Any]:
    return {
        "evidence_id": item.evidence_id,
        "project_id": item.project_id,
        "client_capture_id": item.client_capture_id,
        "media_id": item.media_id,
        "media_type": item.media_type,
        "evidence_type": item.media_type,
        "filename": item.filename,
        "mime": item.mime,
        "uri": item.uri,
        "hash": item.hash,
        "sha256": item.hash,
        "captured_at": item.captured_at,
        "author": item.author,
        "location": dict(item.location_json or {}),
        "location_method": item.location_method,
        "metadata": dict(item.metadata_json or {}),
        "quality": dict(item.quality_json or {}),
        "sufficiency": item.sufficiency,
        "status": item.status,
        "deduplicated": deduplicated,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "download_url": (
            f"{get_settings().public_base_url}/v1/media/{item.media_id}/download"
            if item.media_id
            else ""
        ),
    }


def _review_dict(review: ReviewRecord) -> dict[str, Any]:
    return {
        "review_id": review.review_id,
        "project_id": review.project_id,
        "issue_id": review.issue_id,
        "reviewer": review.reviewer,
        "decision": review.decision,
        "reason_code": review.reason_code,
        "reason": review.reason,
        "issue_version": review.issue_version,
        "timestamp": review.created_at,
        "created_at": review.created_at,
    }


def _risk_flags(workflow: IssueWorkflow) -> list[str]:
    flags: list[str] = []
    if workflow.source_status != "current":
        flags.append(f"source_{workflow.source_status}")
    if workflow.evidence_gaps_json:
        flags.append("insufficient_evidence")
    impact = workflow.impact_json or {}
    if impact.get("cost") or impact.get("safety"):
        flags.append("high_impact")
    return flags


def issue_detail(session: Session, issue: Issue) -> dict[str, Any]:
    workflow = get_or_create_issue_workflow(session, issue)
    reviews = session.scalars(
        select(ReviewRecord)
        .where(ReviewRecord.issue_id == issue.issue_id)
        .order_by(ReviewRecord.created_at.desc())
    ).all()
    payload = issue_snapshot(issue, workflow)
    evidence_payload: list[dict[str, Any]] = []
    for item in list(issue.evidence or []):
        media_id = ""
        if item.evidence_type == "field_evidence":
            field = session.get(FieldEvidence, item.ref_id)
            media_id = field.media_id if field else ""
        elif item.evidence_type == "frame" and session.get(SiteMedia, item.ref_id):
            media_id = item.ref_id
        evidence_payload.append(
            {
                "evidence_id": item.evidence_id,
                "evidence_type": item.evidence_type,
                "ref_id": item.ref_id,
                "r2_key": item.r2_key,
                "page": item.page,
                "bbox": list(item.bbox or []),
                "frame_ts": item.frame_ts,
                "label": item.label,
                "download_url": (
                    f"{get_settings().public_base_url}/v1/media/{media_id}/download"
                    if media_id
                    else ""
                ),
            }
        )
    payload.update(
        {
            "evidence": evidence_payload,
            "spatial_context": issue.spatial_context,
            "priority": workflow.priority,
            "expected_condition": workflow.expected_condition,
            "difference": workflow.difference,
            "recommended_route": workflow.recommended_route,
            "evidence_gaps": list(workflow.evidence_gaps_json or []),
            "source_status": workflow.source_status,
            "review_status": workflow.review_status,
            "issue_version": workflow.version,
            "risk_flags": _risk_flags(workflow),
            "reviews": [_review_dict(item) for item in reviews],
        }
    )
    return payload


def ensure_project_revisions(session: Session, project_id: str) -> list[tuple[Document, DocumentRevision]]:
    documents = list(
        session.scalars(
            select(Document)
            .where(Document.project_id == project_id)
            .order_by(Document.created_at.asc())
        ).all()
    )
    rows = [(document, get_or_create_revision(session, document)) for document in documents]
    groups: dict[str, list[tuple[Document, DocumentRevision]]] = {}
    for pair in rows:
        groups.setdefault(pair[1].logical_key, []).append(pair)
    for pairs in groups.values():
        if not any(revision.state == "current" for _, revision in pairs):
            # Backfill the newest legacy upload as current.  Later uploads receive an
            # explicit unclassified row at completion and therefore never replace an
            # already-current document until activate is called.
            _, newest = pairs[-1]
            newest.state = "current"
            newest.activated_at = newest.activated_at or utcnow()
            for _, older in pairs[:-1]:
                older.state = "superseded"
    session.flush()
    return rows


def _notification(
    session: Session,
    *,
    project_id: str,
    event_type: str,
    title: str,
    body: str,
    entity_type: str,
    entity_id: str,
    recipient: str = "",
) -> Notification:
    item = Notification(
        project_id=project_id,
        recipient=recipient,
        event_type=event_type,
        title=title,
        body=body,
        entity_type=entity_type,
        entity_id=entity_id,
        channel_json=["in_app"],
    )
    session.add(item)
    return item


@router.get("/projects/{project_id}")
def get_project_detail(project_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    project = _project_or_404(session, project_id)
    profile = get_or_create_project_profile(session, project_id)
    session.commit()
    return project_snapshot(project, profile)


@router.patch("/projects/{project_id}")
def patch_project(
    project_id: str,
    payload: ProjectPatch,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    actor, role = actor_context(request)
    require_role(role, EDIT_ROLES, "edit projects")
    project = _project_or_404(session, project_id)
    profile = get_or_create_project_profile(session, project_id)
    before = project_snapshot(project, profile)
    values = payload.model_dump(exclude_unset=True)
    for key in ("name", "address", "project_type", "status"):
        if key in values and values[key] is not None:
            setattr(project, key, values[key])
    for key in ("client", "timezone", "unit_system"):
        if key in values and values[key] is not None:
            setattr(profile, key, values[key])
    after = project_snapshot(project, profile)
    record_audit(
        session,
        project=project,
        actor=actor,
        action="PROJECT_UPDATED",
        entity_type="project",
        entity_id=project_id,
        before=before,
        after=after,
    )
    session.commit()
    return after


@router.get("/projects/{project_id}/settings")
def get_project_settings(project_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    _project_or_404(session, project_id)
    profile = get_or_create_project_profile(session, project_id)
    session.commit()
    return {
        "project_id": project_id,
        "timezone": profile.timezone,
        "unit_system": profile.unit_system,
        "settings": dict(profile.settings_json or {}),
        "workflow": dict(profile.workflow_json or {}),
    }


@router.patch("/projects/{project_id}/settings")
def patch_project_settings(
    project_id: str,
    payload: ProjectSettingsPatch,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    actor, role = actor_context(request)
    require_role(role, APPROVER_ROLES, "change project settings")
    project = _project_or_404(session, project_id)
    profile = get_or_create_project_profile(session, project_id)
    before = {
        "timezone": profile.timezone,
        "unit_system": profile.unit_system,
        "settings": dict(profile.settings_json or {}),
        "workflow": dict(profile.workflow_json or {}),
    }
    values = payload.model_dump(exclude_unset=True)
    if values.get("timezone") is not None:
        profile.timezone = values["timezone"]
    if values.get("unit_system") is not None:
        profile.unit_system = values["unit_system"]
    if values.get("settings") is not None:
        profile.settings_json = {**(profile.settings_json or {}), **values["settings"]}
    if values.get("workflow") is not None:
        profile.workflow_json = {**(profile.workflow_json or {}), **values["workflow"]}
    after = {
        "project_id": project_id,
        "timezone": profile.timezone,
        "unit_system": profile.unit_system,
        "settings": dict(profile.settings_json or {}),
        "workflow": dict(profile.workflow_json or {}),
    }
    record_audit(
        session,
        project=project,
        actor=actor,
        action="PROJECT_SETTINGS_UPDATED",
        entity_type="project_settings",
        entity_id=project_id,
        before=before,
        after=after,
    )
    session.commit()
    return after


@router.get("/projects/{project_id}/directory")
def list_directory(project_id: str, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    _project_or_404(session, project_id)
    items = session.scalars(
        select(DirectoryMember)
        .where(DirectoryMember.project_id == project_id)
        .order_by(DirectoryMember.person_name.asc())
    ).all()
    return [_directory_dict(item) for item in items]


@router.post("/projects/{project_id}/directory", status_code=201)
def create_directory_member(
    project_id: str,
    payload: DirectoryCreate,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    actor, role = actor_context(request)
    require_role(role, APPROVER_ROLES, "manage the directory")
    project = _project_or_404(session, project_id)
    if payload.email:
        duplicate = session.scalar(
            select(DirectoryMember).where(
                DirectoryMember.project_id == project_id,
                DirectoryMember.email == payload.email,
                DirectoryMember.status != "disabled",
            )
        )
        if duplicate:
            raise HTTPException(status_code=409, detail="an active directory member uses this email")
    item = DirectoryMember(
        project_id=project_id,
        person_name=payload.person_name,
        email=payload.email,
        company=payload.company,
        role=payload.role,
        trade=payload.trade,
        status=payload.status,
        notification_json=payload.notification,
        access_expires_at=payload.access_expires_at,
    )
    session.add(item)
    session.flush()
    after = _directory_dict(item)
    record_audit(
        session,
        project=project,
        actor=actor,
        action="PERMISSION_CHANGED",
        entity_type="directory_member",
        entity_id=item.directory_id,
        after=after,
    )
    session.commit()
    return after


@router.patch("/directory/{directory_id}")
def patch_directory_member(
    directory_id: str,
    payload: DirectoryPatch,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    actor, role = actor_context(request)
    require_role(role, APPROVER_ROLES, "manage the directory")
    item = session.get(DirectoryMember, directory_id)
    if not item:
        raise HTTPException(status_code=404, detail="directory member not found")
    project = _project_or_404(session, item.project_id)
    before = _directory_dict(item)
    field_map = {"notification": "notification_json"}
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(item, field_map.get(key, key), value)
    after = _directory_dict(item)
    record_audit(
        session,
        project=project,
        actor=actor,
        action="PERMISSION_CHANGED",
        entity_type="directory_member",
        entity_id=item.directory_id,
        before=before,
        after=after,
    )
    session.commit()
    return after


@router.get("/projects/{project_id}/drawing-sets")
def list_drawing_sets(project_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    _project_or_404(session, project_id)
    rows = ensure_project_revisions(session, project_id)
    session.commit()
    items = [_revision_dict(document, revision) for document, revision in rows]
    return {
        "project_id": project_id,
        "current": [item for item in items if item["state"] == "current"],
        "superseded": [item for item in items if item["state"] == "superseded"],
        "unclassified": [item for item in items if item["state"] == "unclassified"],
        "items": items,
    }


def _affected_issues_for_documents(
    session: Session, project_id: str, document_ids: set[str]
) -> list[Issue]:
    chunk_ids = set(
        session.scalars(select(SpecChunk.chunk_id).where(SpecChunk.doc_id.in_(document_ids))).all()
    )
    sheets = list(session.scalars(select(Sheet).where(Sheet.doc_id.in_(document_ids))).all())
    sheet_ids = {item.sheet_id for item in sheets}
    sheet_numbers = {item.sheet_number for item in sheets}
    entity_ids = set(
        session.scalars(select(PlanEntity.entity_id).where(PlanEntity.sheet_id.in_(sheet_ids))).all()
    )
    refs = document_ids | chunk_ids | sheet_ids | entity_ids
    issues = session.scalars(
        select(Issue)
        .options(selectinload(Issue.evidence))
        .where(Issue.project_id == project_id, Issue.status.in_(OPEN_ISSUE_STATUSES))
    ).all()
    affected: list[Issue] = []
    for issue in issues:
        workflow = get_or_create_issue_workflow(session, issue)
        snapshot_docs = {
            str(item.get("document_id")) for item in (workflow.source_snapshot_json or [])
        }
        direct_ref = any(item.ref_id in refs for item in list(issue.evidence or []))
        requirement = issue.requirement or {}
        requirement_ref = (
            requirement.get("source") in sheet_numbers
            or requirement.get("document_id") in document_ids
            or requirement.get("source_document_id") in document_ids
        )
        if direct_ref or requirement_ref or snapshot_docs.intersection(document_ids):
            workflow.source_status = "stale"
            workflow.review_status = "stale_source_review"
            workflow.version += 1
            issue.status = "stale_source_review"
            affected.append(issue)
    return affected


def _activate_document(
    document_id: str,
    payload: RevisionActivateRequest,
    request: Request,
    session: Session,
) -> dict[str, Any]:
    actor, role = actor_context(request)
    require_role(role, APPROVER_ROLES, "activate revisions")
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="document not found")
    project = _project_or_404(session, document.project_id)
    target = get_or_create_revision(session, document, actor=actor)
    original_target_state = target.state
    ensure_project_revisions(session, document.project_id)
    values = payload.model_dump(exclude_none=True)
    for key in ("logical_key", "sheet_number", "issue_date", "discipline"):
        if values.get(key):
            setattr(target, key, values[key].strip())
    current = list(
        session.scalars(
            select(DocumentRevision).where(
                DocumentRevision.project_id == document.project_id,
                DocumentRevision.logical_key == target.logical_key,
                DocumentRevision.state == "current",
                DocumentRevision.document_id != document_id,
            )
        ).all()
    )
    if not current and original_target_state != "current":
        # Legacy rows may have been backfilled before the explicit activation.  A
        # revision activation still supersedes every older version in the same
        # logical set and must name it in the audit record.
        current = list(
            session.scalars(
                select(DocumentRevision)
                .where(
                    DocumentRevision.project_id == document.project_id,
                    DocumentRevision.logical_key == target.logical_key,
                    DocumentRevision.document_id != document_id,
                )
                .order_by(DocumentRevision.created_at.asc())
            ).all()
        )
    before = {
        "target_state": original_target_state,
        "current_document_ids": [item.document_id for item in current],
    }
    superseded_ids: list[str] = []
    for previous in current:
        previous.state = "superseded"
        superseded_ids.append(previous.document_id)
    target.state = "current"
    target.activated_at = utcnow()
    target.source_hash = document.hash
    target.supersedes_document_id = superseded_ids[-1] if superseded_ids else target.supersedes_document_id
    affected = _affected_issues_for_documents(session, document.project_id, set(superseded_ids))
    affected_ids = [item.issue_id for item in affected]
    for issue in affected:
        _notification(
            session,
            project_id=document.project_id,
            recipient=issue.assignee,
            event_type="drawing_revision_impacts_issue",
            title=f"Revision {document.revision} requires source review",
            body=f"{issue.issue_id} references a superseded source. Verify the current revision.",
            entity_type="issue",
            entity_id=issue.issue_id,
        )
    stale_graphs = list(
        session.scalars(select(PlanGraph).where(PlanGraph.project_id == document.project_id)).all()
    )
    stale_design_assets = list(
        session.scalars(
            select(SpatialAsset).where(
                SpatialAsset.project_id == document.project_id,
                SpatialAsset.type == "design_glb",
            )
        ).all()
    )
    for asset in stale_design_assets:
        root = get_settings().storage_root.resolve()
        artifact = (root / Path(asset.uri)).resolve()
        try:
            artifact.relative_to(root)
        except ValueError:
            continue
        artifact.unlink(missing_ok=True)
    issue_ids = select(Issue.issue_id).where(Issue.project_id == document.project_id)
    session.execute(
        delete(SpatialEvidence).where(SpatialEvidence.issue_id.in_(issue_ids))
    )
    session.execute(
        delete(SpatialAlignment).where(SpatialAlignment.project_id == document.project_id)
    )
    session.execute(
        delete(SpatialAsset).where(
            SpatialAsset.project_id == document.project_id,
            SpatialAsset.type == "design_glb",
        )
    )
    session.execute(delete(PlanGraph).where(PlanGraph.project_id == document.project_id))
    session.flush()
    regenerated_spatial: dict[str, str] = {}
    has_parsed_sheet = session.scalar(
        select(Sheet.sheet_id).where(Sheet.doc_id == document.doc_id).limit(1)
    )
    if has_parsed_sheet:
        try:
            graph = create_plan_graph_record(
                session,
                project,
                source_doc_id=document.doc_id,
                replace_existing=True,
            )
            asset_id = f"spa_{hashlib.sha256((graph.id + document.hash).encode()).hexdigest()[:12]}"
            uri, metadata = build_design_glb(
                graph.graph_json or {}, project.project_id, asset_id
            )
            asset = SpatialAsset(
                id=asset_id,
                project_id=project.project_id,
                type="design_glb",
                uri=uri,
                metadata_json={
                    **metadata,
                    "plan_graph_id": graph.id,
                    **plan_graph_provenance(graph),
                },
            )
            session.add(asset)
            session.flush()
            regenerated_spatial = {"plan_graph_id": graph.id, "design_asset_id": asset.id}
        except (ValueError, OSError) as exc:
            regenerated_spatial = {"error": str(exc)}
    after = {
        "current_document_id": document_id,
        "superseded_document_ids": superseded_ids,
        "affected_issue_ids": affected_ids,
        "logical_key": target.logical_key,
        "revision": target.revision,
        "invalidated_plan_graph_ids": [item.id for item in stale_graphs],
        "invalidated_design_asset_ids": [item.id for item in stale_design_assets],
        "regenerated_spatial": regenerated_spatial,
    }
    record_audit(
        session,
        project=project,
        actor=actor,
        action="REVISION_ACTIVATED",
        entity_type="document_revision",
        entity_id=target.revision_id,
        before=before,
        after=after,
    )
    session.commit()
    return after


@router.post("/documents/{document_id}/activate")
def activate_document(
    document_id: str,
    request: Request,
    payload: RevisionActivateRequest | None = None,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    return _activate_document(document_id, payload or RevisionActivateRequest(), request, session)


@router.post("/revisions/{document_id}/activate")
def activate_revision_alias(
    document_id: str,
    request: Request,
    payload: RevisionActivateRequest | None = None,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    return _activate_document(document_id, payload or RevisionActivateRequest(), request, session)


def _decode_capture(
    payload: EvidenceSyncRequest, *, verify_declared: bool = True
) -> tuple[bytes, str]:
    if not payload.content_base64:
        return b"", payload.sha256 or payload.hash
    encoded = payload.content_base64
    if encoded.startswith("data:"):
        if "," not in encoded:
            raise HTTPException(status_code=400, detail="invalid data URL")
        encoded = encoded.split(",", 1)[1]
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="content_base64 is invalid") from exc
    if not content:
        raise HTTPException(status_code=400, detail="capture content is empty")
    if len(content) > get_settings().max_upload_bytes:
        raise HTTPException(status_code=413, detail="capture exceeds maximum allowed size")
    digest = hashlib.sha256(content).hexdigest()
    declared = (payload.sha256 or payload.hash).lower().strip()
    if verify_declared and declared and declared != digest:
        raise HTTPException(status_code=422, detail="declared sha256 does not match capture bytes")
    return content, digest


def _validate_capture_type(payload: EvidenceSyncRequest, has_content: bool) -> tuple[str, str]:
    media_type = payload.evidence_type or payload.media_type
    filename = Path(payload.filename).name.strip()
    if filename != payload.filename.strip() or not filename:
        if media_type != "measurement":
            raise HTTPException(status_code=400, detail="a safe filename is required")
        filename = filename or "measurement.json"
    suffix = Path(filename).suffix.lower()
    if media_type == "photo" and suffix not in PHOTO_EXTENSIONS:
        raise HTTPException(status_code=415, detail="unsupported photo extension")
    if media_type == "video" and suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail="unsupported video extension")
    if media_type == "audio" and suffix not in AUDIO_EXTENSIONS:
        raise HTTPException(status_code=415, detail="unsupported audio extension")
    expected_mime = MIME_PREFIX.get(media_type)
    if expected_mime and (not payload.mime or not payload.mime.lower().startswith(expected_mime)):
        raise HTTPException(status_code=415, detail=f"mime must be {expected_mime}*")
    if media_type != "measurement" and not has_content and not payload.media_id:
        raise HTTPException(status_code=400, detail="capture bytes or a completed media_id are required")
    return media_type, filename


@router.post("/evidence/sync")
@router.post("/evidence")
def sync_evidence(
    payload: EvidenceSyncRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    actor, role = actor_context(request)
    require_role(role, EDIT_ROLES, "capture evidence")
    project = _project_or_404(session, payload.project_id)
    client_capture_id = (payload.client_capture_id or payload.client_id).strip()
    if not client_capture_id:
        raise HTTPException(status_code=400, detail="client_capture_id is required for idempotent sync")
    existing = session.scalar(
        select(FieldEvidence).where(
            FieldEvidence.project_id == payload.project_id,
            FieldEvidence.client_capture_id == client_capture_id,
        )
    )
    content, digest = _decode_capture(payload, verify_declared=existing is None)
    media_type, filename = _validate_capture_type(payload, bool(content))
    if existing:
        if digest and existing.hash and digest != existing.hash:
            raise HTTPException(status_code=409, detail="client_capture_id already has different bytes")
        return _evidence_dict(existing, deduplicated=True)

    media_id = payload.media_id
    uri = payload.uri
    if media_id:
        media = session.get(SiteMedia, media_id)
        if not media or media.project_id != payload.project_id:
            raise HTTPException(status_code=404, detail="completed media not found in project")
        digest = digest or media.hash
        uri = uri or media.r2_key
        filename = media.filename or filename
    elif content:
        r2_key = (
            f"org/{project.org_id}/project/{project.project_id}/raw/"
            f"offline_{digest[:16]}_{filename}"
        )
        path = object_path(r2_key)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise HTTPException(status_code=409, detail="stored capture hash conflict")
        if not path.exists():
            path.write_bytes(content)
        media = SiteMedia(
            project_id=project.project_id,
            filename=filename,
            mime=payload.mime,
            r2_key=r2_key,
            hash=digest,
            metadata_json={
                "source": "offline_capture_sync",
                "client_capture_id": client_capture_id,
                "size": len(content),
                "location": payload.location,
                "observation": payload.observation,
            },
        )
        session.add(media)
        session.flush()
        media_id = media.media_id
        uri = r2_key
    elif not digest:
        canonical = json.dumps(
            {
                "client_capture_id": client_capture_id,
                "location": payload.location,
                "metadata": payload.metadata,
                "observation": payload.observation,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(canonical).hexdigest()

    location_method = str(payload.location.get("method") or payload.location_method or "manual")
    item = FieldEvidence(
        project_id=project.project_id,
        client_capture_id=client_capture_id,
        media_id=media_id,
        media_type=media_type,
        filename=filename,
        mime=payload.mime,
        uri=uri,
        hash=digest,
        captured_at=payload.captured_at or utcnow(),
        author=payload.author or actor,
        location_json=payload.location,
        location_method=location_method,
        metadata_json={**payload.metadata, "observation": payload.observation},
        quality_json=payload.quality,
        sufficiency=payload.sufficiency,
        status="unlinked",
    )
    session.add(item)
    session.flush()
    after = _evidence_dict(item)
    record_audit(
        session,
        project=project,
        actor=actor,
        action="EVIDENCE_CAPTURED",
        entity_type="field_evidence",
        entity_id=item.evidence_id,
        after=after,
        metadata={"client_capture_id": client_capture_id, "sha256": digest},
    )
    session.commit()
    return after


@router.post("/projects/{project_id}/evidence/sync")
def sync_project_evidence(
    project_id: str,
    payload: EvidenceSyncRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    if payload.project_id and payload.project_id != project_id:
        raise HTTPException(status_code=409, detail="project_id does not match route")
    return sync_evidence(payload.model_copy(update={"project_id": project_id}), request, session)


@router.get("/projects/{project_id}/evidence")
def list_evidence(
    project_id: str,
    status: str | None = None,
    room: str | None = None,
    session: Session = Depends(get_session),
) -> list[dict[str, Any]]:
    _project_or_404(session, project_id)
    statement = select(FieldEvidence).where(FieldEvidence.project_id == project_id)
    if status:
        statement = statement.where(FieldEvidence.status == status)
    items = list(session.scalars(statement.order_by(FieldEvidence.captured_at.desc())).all())
    if room:
        items = [item for item in items if str((item.location_json or {}).get("room")) == room]
    return [_evidence_dict(item) for item in items]


def _invalidate_issues_for_evidence_change(
    session: Session,
    evidence: FieldEvidence,
    *,
    actor: str,
    reason: str,
    only_issue_ids: set[str] | None = None,
) -> list[str]:
    """Version linked issues and revoke any approval based on changed evidence."""

    links = list(
        session.scalars(
            select(EvidenceLink).where(EvidenceLink.evidence_id == evidence.evidence_id)
        ).all()
    )
    affected: list[str] = []
    for link in links:
        if only_issue_ids is not None and link.issue_id not in only_issue_ids:
            continue
        issue = _issue_or_404(session, link.issue_id)
        project = _project_or_404(session, issue.project_id)
        workflow = get_or_create_issue_workflow(session, issue)
        before = issue_snapshot(issue, workflow)
        was_approved = workflow.review_status == "approved" or issue.status == "approved"
        workflow.version += 1
        if was_approved:
            workflow.review_status = "stale_evidence_review"
            workflow.reviewer = ""
            issue.status = "stale_evidence_review"
        after = issue_snapshot(issue, workflow)
        record_audit(
            session,
            project=project,
            actor=actor,
            action=(
                "ISSUE_APPROVAL_INVALIDATED"
                if was_approved
                else "ISSUE_EVIDENCE_VERSION_CHANGED"
            ),
            entity_type="issue",
            entity_id=issue.issue_id,
            before=before,
            after=after,
            metadata={"evidence_id": evidence.evidence_id, "reason": reason},
        )
        if was_approved:
            _notification(
                session,
                project_id=issue.project_id,
                recipient=issue.assignee,
                event_type="evidence_change_invalidated_approval",
                title=f"{issue.issue_id} requires evidence re-review",
                body="Linked evidence changed after approval. Review the updated package.",
                entity_type="issue",
                entity_id=issue.issue_id,
            )
        affected.append(issue.issue_id)
    return affected


def _patch_evidence(
    evidence_id: str,
    values: dict[str, Any],
    request: Request,
    session: Session,
) -> dict[str, Any]:
    actor, role = actor_context(request)
    require_role(role, EDIT_ROLES, "edit evidence metadata")
    item = session.get(FieldEvidence, evidence_id)
    if not item:
        raise HTTPException(status_code=404, detail="evidence not found")
    project = _project_or_404(session, item.project_id)
    before = _evidence_dict(item)
    mapping = {
        "location": "location_json",
        "metadata": "metadata_json",
        "quality": "quality_json",
    }
    for key, value in values.items():
        if value is not None:
            setattr(item, mapping.get(key, key), value)
    after = _evidence_dict(item)
    affected_issue_ids: list[str] = []
    if before != after:
        affected_issue_ids = _invalidate_issues_for_evidence_change(
            session,
            item,
            actor=actor,
            reason="evidence_metadata_or_location_changed",
        )
    record_audit(
        session,
        project=project,
        actor=actor,
        action="EVIDENCE_METADATA_UPDATED",
        entity_type="field_evidence",
        entity_id=evidence_id,
        before=before,
        after=after,
        metadata={"affected_issue_ids": affected_issue_ids},
    )
    session.commit()
    return after


@router.patch("/evidence/{evidence_id}")
def patch_evidence(
    evidence_id: str,
    payload: EvidencePatch,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    return _patch_evidence(evidence_id, payload.model_dump(exclude_unset=True), request, session)


@router.patch("/evidence/{evidence_id}/location")
def patch_evidence_location(
    evidence_id: str,
    payload: EvidenceLocationPatch,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    return _patch_evidence(evidence_id, payload.model_dump(), request, session)


def _link_evidence_internal(
    session: Session,
    evidence: FieldEvidence,
    issue: Issue,
    *,
    relevance: str,
    annotation: str,
    actor: str,
) -> EvidenceLink:
    existing = session.scalar(
        select(EvidenceLink).where(
            EvidenceLink.evidence_id == evidence.evidence_id,
            EvidenceLink.issue_id == issue.issue_id,
        )
    )
    if existing:
        return existing
    link = EvidenceLink(
        evidence_id=evidence.evidence_id,
        issue_id=issue.issue_id,
        relevance=relevance,
        annotation=annotation,
        linked_by=actor,
    )
    session.add(link)
    if not any(
        item.evidence_type == "field_evidence" and item.ref_id == evidence.evidence_id
        for item in list(issue.evidence or [])
    ):
        issue.evidence.append(
            IssueEvidence(
                evidence_type="field_evidence",
                ref_id=evidence.evidence_id,
                r2_key=evidence.uri,
                page=0,
                bbox=[],
                frame_ts=0,
                label=annotation or evidence.filename or evidence.media_type,
            )
        )
    evidence.status = "linked"
    workflow = get_or_create_issue_workflow(session, issue)
    workflow.evidence_gaps_json = [
        gap for gap in (workflow.evidence_gaps_json or []) if gap.get("type") != "field_evidence"
    ]
    session.flush()
    _invalidate_issues_for_evidence_change(
        session,
        evidence,
        actor=actor,
        reason="new_evidence_linked",
        only_issue_ids={issue.issue_id},
    )
    return link


@router.post("/evidence/{evidence_id}/link")
def link_evidence(
    evidence_id: str,
    payload: EvidenceLinkRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    actor, role = actor_context(request)
    require_role(role, EDIT_ROLES, "link evidence")
    evidence = session.get(FieldEvidence, evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="evidence not found")
    issue = _issue_or_404(session, payload.issue_id)
    if evidence.project_id != issue.project_id:
        raise HTTPException(status_code=409, detail="evidence and issue belong to different projects")
    project = _project_or_404(session, issue.project_id)
    link = _link_evidence_internal(
        session,
        evidence,
        issue,
        relevance=payload.relevance,
        annotation=payload.annotation,
        actor=actor,
    )
    record_audit(
        session,
        project=project,
        actor=actor,
        action="EVIDENCE_LINKED",
        entity_type="issue",
        entity_id=issue.issue_id,
        after={"evidence_id": evidence_id, "relevance": link.relevance},
    )
    session.commit()
    return {
        "link_id": link.link_id,
        "evidence_id": evidence_id,
        "issue_id": issue.issue_id,
        "relevance": link.relevance,
        "annotation": link.annotation,
    }


@router.post("/projects/{project_id}/issues", status_code=201)
def create_issue(
    project_id: str,
    payload: IssueCreate,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    actor, role = actor_context(request)
    require_role(role, EDIT_ROLES, "create issues")
    if payload.project_id != project_id:
        raise HTTPException(status_code=409, detail="project_id does not match route")
    project = _project_or_404(session, project_id)
    issue = Issue(
        project_id=project_id,
        type=payload.type,
        discipline=payload.discipline,
        severity=payload.severity,
        room=payload.room,
        status="draft",
        confidence=payload.confidence,
        title=payload.title,
        description=payload.description,
        recommended_action=payload.recommended_action,
        assignee=payload.assignee,
        due_date=payload.due_date,
        subcontractor=payload.subcontractor,
        requirement=payload.requirement,
        observation=payload.observation,
        plan_location=payload.plan_location,
        rfi_draft=payload.rfi_draft,
    )
    session.add(issue)
    session.flush()
    for source in payload.source_references:
        document_id = str(source.get("document_id") or source.get("doc_id") or "")
        document = session.get(Document, document_id) if document_id else None
        if not document or document.project_id != project_id:
            raise HTTPException(status_code=422, detail=f"source document not found: {document_id}")
        issue.evidence.append(
            IssueEvidence(
                evidence_type="source",
                ref_id=document_id,
                r2_key=document.r2_key,
                page=int(source.get("page") or 1),
                bbox=list(source.get("bbox") or []),
                frame_ts=0,
                label=str(source.get("label") or document.filename),
            )
        )
    workflow = IssueWorkflow(
        issue_id=issue.issue_id,
        priority=payload.priority,
        expected_condition=payload.expected_condition or str(payload.requirement.get("text") or ""),
        difference=payload.difference or payload.description,
        recommended_route=payload.recommended_route,
        evidence_gaps_json=payload.evidence_gaps,
        source_status="unresolved",
        impact_json=payload.impact,
        review_status="draft",
        version=1,
    )
    session.add(workflow)
    session.flush()
    for evidence_id in payload.evidence_ids:
        evidence = session.get(FieldEvidence, evidence_id)
        if not evidence or evidence.project_id != project_id:
            raise HTTPException(status_code=422, detail=f"field evidence not found: {evidence_id}")
        _link_evidence_internal(
            session,
            evidence,
            issue,
            relevance="supports",
            annotation="Linked at issue creation",
            actor=actor,
        )
    ensure_project_revisions(session, project_id)
    workflow.source_snapshot_json = source_snapshot_for_issue(session, issue)
    workflow.source_status = (
        "current"
        if workflow.source_snapshot_json
        and all(item.get("state") == "current" for item in workflow.source_snapshot_json)
        else "unresolved"
    )
    if workflow.source_status == "current":
        workflow.evidence_gaps_json = [
            gap for gap in (workflow.evidence_gaps_json or []) if gap.get("type") != "source"
        ]
    record_audit(
        session,
        project=project,
        actor=actor,
        action="ISSUE_CREATED",
        entity_type="issue",
        entity_id=issue.issue_id,
        after=issue_snapshot(issue, workflow),
    )
    session.commit()
    return issue_detail(session, issue)


@router.get("/issues/{issue_id}")
def get_issue_detail(issue_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    issue = _issue_or_404(session, issue_id)
    payload = issue_detail(session, issue)
    session.commit()
    return payload


def refresh_review_readiness(session: Session, issue: Issue, workflow: IssueWorkflow) -> None:
    ensure_project_revisions(session, issue.project_id)
    snapshot = source_snapshot_for_issue(session, issue)
    workflow.source_snapshot_json = snapshot
    if not snapshot:
        workflow.source_status = "unresolved"
    elif any(item.get("state") == "superseded" for item in snapshot):
        workflow.source_status = "stale"
    elif any(item.get("state") != "current" for item in snapshot):
        workflow.source_status = "unresolved"
    else:
        workflow.source_status = "current"
    field_links = session.scalars(
        select(EvidenceLink).where(EvidenceLink.issue_id == issue.issue_id)
    ).all()
    has_uploaded_frame = any(
        item.evidence_type in {"frame", "field_evidence"}
        and item.ref_id not in {"", "field_verification_pending"}
        for item in list(issue.evidence or [])
    )
    if field_links or has_uploaded_frame:
        workflow.evidence_gaps_json = [
            gap for gap in (workflow.evidence_gaps_json or []) if gap.get("type") != "field_evidence"
        ]
    if workflow.source_status == "current":
        workflow.evidence_gaps_json = [
            gap for gap in (workflow.evidence_gaps_json or []) if gap.get("type") != "source"
        ]


@router.post("/issues/{issue_id}/reviews")
def review_issue(
    issue_id: str,
    payload: ReviewCreate,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    actor, role = actor_context(request)
    require_role(role, APPROVER_ROLES, "review issues")
    principal = current_principal()
    effective_reviewer = principal.email if principal else payload.reviewer
    if not effective_reviewer.strip():
        raise HTTPException(status_code=422, detail="reviewer is required")
    if payload.decision in {"reject", "request_evidence"} and not payload.reason.strip():
        raise HTTPException(status_code=422, detail="a reason is required for this decision")
    issue = _issue_or_404(session, issue_id)
    project = _project_or_404(session, issue.project_id)
    workflow = get_or_create_issue_workflow(session, issue)
    before = issue_snapshot(issue, workflow)
    refresh_review_readiness(session, issue, workflow)
    if payload.decision == "approve":
        if workflow.source_status != "current":
            raise HTTPException(
                status_code=409,
                detail=f"cannot approve with {workflow.source_status} source version",
            )
        if workflow.evidence_gaps_json:
            raise HTTPException(status_code=409, detail="cannot approve while evidence gaps remain")
        blockers = issue_action_blockers(session, issue, workflow)
        if blockers:
            raise HTTPException(
                status_code=409,
                detail={"message": "issue is not ready for approval", "blockers": blockers},
            )
        workflow.review_status = "approved"
        issue.status = "approved"
    elif payload.decision == "reject":
        workflow.review_status = "rejected"
        issue.status = "rejected"
    else:
        gaps = payload.evidence_gaps or [
            {"type": "requested", "message": payload.reason.strip()}
        ]
        workflow.evidence_gaps_json = [*(workflow.evidence_gaps_json or []), *gaps]
        workflow.review_status = "evidence_requested"
        issue.status = "needs_more_evidence"
    workflow.reviewer = effective_reviewer
    review = ReviewRecord(
        project_id=issue.project_id,
        issue_id=issue.issue_id,
        reviewer=effective_reviewer,
        decision=payload.decision,
        reason_code=payload.reason_code,
        reason=payload.reason,
        issue_version=workflow.version,
        snapshot_json=issue_snapshot(issue, workflow),
    )
    session.add(review)
    session.flush()
    after = {
        **issue_snapshot(issue, workflow),
        "review_decision": payload.decision,
        "reviewer": effective_reviewer,
        "review_reason": payload.reason,
    }
    record_audit(
        session,
        project=project,
        actor=actor,
        action="REVIEW_DECIDED",
        entity_type="issue",
        entity_id=issue.issue_id,
        before=before,
        after=after,
        metadata={
            "review_id": review.review_id,
            "decision": payload.decision,
            "reviewer": effective_reviewer,
            "reason": payload.reason,
        },
    )
    _notification(
        session,
        project_id=issue.project_id,
        recipient=issue.assignee,
        event_type="review_decided",
        title=f"{issue.issue_id} review: {payload.decision.replace('_', ' ')}",
        body=payload.reason or f"Reviewed by {effective_reviewer}.",
        entity_type="issue",
        entity_id=issue.issue_id,
    )
    session.commit()
    return {**_review_dict(review), "issue": issue_detail(session, issue)}


@router.get("/issues/{issue_id}/reviews")
def list_issue_reviews(issue_id: str, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    _issue_or_404(session, issue_id)
    reviews = session.scalars(
        select(ReviewRecord)
        .where(ReviewRecord.issue_id == issue_id)
        .order_by(ReviewRecord.created_at.desc())
    ).all()
    return [_review_dict(item) for item in reviews]


@router.post("/issues/{issue_id}/request-evidence", status_code=201)
def request_issue_evidence(
    issue_id: str,
    payload: RequestEvidenceCreate,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    actor, role = actor_context(request)
    require_role(role, APPROVER_ROLES, "request evidence")
    reviewer = payload.requested_by or actor
    return review_issue(
        issue_id,
        ReviewCreate(
            decision="request_evidence",
            reviewer=reviewer,
            reason=payload.reason,
            reason_code="insufficient_evidence",
            evidence_gaps=payload.evidence_gaps,
        ),
        request,
        session,
    )


def _version_dict(version: ReportVersion, report: ReportRecord) -> dict[str, Any]:
    try:
        relative = Path(version.path).resolve().relative_to(
            (get_settings().storage_root / "reports").resolve()
        )
        download_url = f"{get_settings().public_base_url}/v1/reports/{relative.as_posix()}"
    except ValueError:
        download_url = ""
    return {
        "version_id": version.version_id,
        "report_id": report.report_id,
        "project_id": report.project_id,
        "report_type": report.report_type,
        "title": report.title,
        "status": version.status,
        "version": version.version,
        "format": version.format,
        "checksum": version.checksum,
        "source_snapshot": list(version.source_snapshot_json or []),
        "issue_snapshot": list(version.issue_snapshot_json or []),
        "reviewer": version.reviewer,
        "issued_at": version.issued_at,
        "created_at": version.created_at,
        "download_url": download_url,
    }


def _build_issued_artifact(
    source: Path,
    *,
    report: ReportRecord,
    version: int,
    reviewer: str,
    issue_ids: list[str],
    source_labels: list[str],
) -> Path:
    destination = source.with_name(f"{source.stem}_issued_v{version}{source.suffix}")
    if source.suffix.lower() != ".pdf":
        shutil.copy2(source, destination)
        return destination

    # Issuance produces a distinct immutable artifact.  A generated cover makes the
    # human approval, version, issue scope and checksum-visible chain of custody part
    # of the PDF itself instead of relying only on database metadata.
    import fitz

    output = fitz.open()
    page = output.new_page(width=612, height=792)
    page.insert_text((54, 72), "BUILI — ISSUED EVIDENCE PACKAGE", fontsize=18)
    lines = [
        f"Report: {report.title or report.report_type}",
        f"Report ID: {report.report_id}",
        f"Version: {version}",
        f"Reviewer: {reviewer or 'Authorized project reviewer'}",
        f"Issued: {utcnow().isoformat()}Z",
        "",
        "Approved issues:",
        *[f"• {issue_id}" for issue_id in issue_ids],
        "",
        "Source revision snapshot:",
        *[f"• {label}" for label in source_labels],
        "",
        "This issued version is immutable. Source revisions and evidence hashes are",
        "preserved in the package manifest and Buili audit trail.",
    ]
    y = 112
    for line in lines:
        page.insert_text((54, y), line, fontsize=10)
        y += 16
    with fitz.open(source) as draft:
        output.insert_pdf(draft)
    output.save(destination)
    output.close()
    return destination


@router.get("/projects/{project_id}/reports")
def list_project_reports(project_id: str, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    _project_or_404(session, project_id)
    reports = session.scalars(
        select(ReportRecord)
        .where(ReportRecord.project_id == project_id)
        .order_by(ReportRecord.created_at.desc())
    ).all()
    items: list[dict[str, Any]] = []
    for report in reports:
        version = session.scalar(
            select(ReportVersion)
            .where(ReportVersion.report_id == report.report_id)
            .order_by(ReportVersion.version.desc())
        )
        if version:
            items.append(_version_dict(version, report))
    return items


@router.get("/reports/{report_id}/versions")
def list_report_versions(report_id: str, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    report = session.get(ReportRecord, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="report not found")
    versions = session.scalars(
        select(ReportVersion)
        .where(ReportVersion.report_id == report_id)
        .order_by(ReportVersion.version.desc())
    ).all()
    return [_version_dict(version, report) for version in versions]


@router.post("/reports/{report_id}/export")
def export_report(
    report_id: str,
    payload: ReportExportRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    actor, role = actor_context(request)
    require_role(role, EXPORT_ROLES, "export official reports")
    report = session.get(ReportRecord, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="report not found")
    project = _project_or_404(session, report.project_id)
    latest = session.scalar(
        select(ReportVersion)
        .where(ReportVersion.report_id == report_id)
        .order_by(ReportVersion.version.desc())
    )
    if not latest:
        raise HTTPException(status_code=409, detail="report has no draft version")
    issue_ids = [str(item.get("issue_id")) for item in (latest.issue_snapshot_json or [])]
    scope = session.scalar(select(ReportScope).where(ReportScope.report_id == report_id))
    workflows = list(
        session.scalars(select(IssueWorkflow).where(IssueWorkflow.issue_id.in_(issue_ids))).all()
    )
    approved_ids = {item.issue_id for item in workflows if item.review_status == "approved"}
    approved_issue_ids = [issue_id for issue_id in issue_ids if issue_id in approved_ids]
    if scope and scope.explicit_selection and len(approved_issue_ids) != len(issue_ids):
        missing_approvals = [issue_id for issue_id in issue_ids if issue_id not in approved_ids]
        raise HTTPException(
            status_code=409,
            detail={
                "message": "every selected report issue requires approval",
                "issue_ids": missing_approvals,
            },
        )
    if not approved_issue_ids:
        raise HTTPException(
            status_code=409,
            detail={"message": "at least one report issue requires approval", "issue_ids": issue_ids},
        )
    # A project-wide draft may contain review-ready candidates.  The official
    # version is intentionally narrowed to the approved subset so unapproved AI
    # candidates can never leak into an issued evidence package.
    issue_ids = approved_issue_ids
    path = Path(latest.path)
    if not path.exists():
        raise HTTPException(status_code=409, detail="report artifact is missing")
    principal = current_principal()
    reviewer = (
        principal.email
        if principal
        else ", ".join(sorted({item.reviewer for item in workflows if item.reviewer}))
    )
    next_version = latest.version + 1
    current_issue_snapshots: list[dict[str, Any]] = []
    current_sources: dict[tuple[str, str], dict[str, Any]] = {}
    readiness_failures: list[dict[str, Any]] = []
    for issue_id in issue_ids:
        issue = _issue_or_404(session, issue_id)
        workflow = get_or_create_issue_workflow(session, issue)
        refresh_review_readiness(session, issue, workflow)
        blockers = issue_action_blockers(
            session,
            issue,
            workflow,
            report_type=report.report_type,
            require_approval=True,
        )
        if blockers:
            readiness_failures.append({"issue_id": issue.issue_id, "blockers": blockers})
        sources = source_snapshot_for_issue(session, issue)
        workflow.source_snapshot_json = sources
        current_issue_snapshots.append(issue_snapshot(issue, workflow))
        for source in sources:
            current_sources[(str(source.get("document_id")), str(source.get("revision")))] = source
    if readiness_failures:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "message": "report cannot be issued until readiness blockers are resolved",
                "issues": readiness_failures,
            },
        )
    source_labels = [
        " · ".join(
            part
            for part in (
                str(source.get("sheet_number") or source.get("filename") or "Source"),
                f"Rev {source.get('revision')}" if source.get("revision") else "",
            )
            if part
        )
        for source in current_sources.values()
    ]
    # Never wrap the original draft: a legacy project-wide draft can contain
    # unapproved candidates, and fields may have changed after draft creation.
    # Rebuild from the exact current approved scope before adding the immutable
    # issuance cover.
    try:
        _, fresh_path = build_report(
            session,
            report.project_id,
            report.report_type,
            latest.format,
            issue_ids=issue_ids,
            artifact_status="issued",
            artifact_version=next_version,
            artifact_reviewer=reviewer,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    issued_path = _build_issued_artifact(
        fresh_path,
        report=report,
        version=next_version,
        reviewer=reviewer,
        issue_ids=issue_ids,
        source_labels=source_labels,
    )
    if fresh_path.resolve() != issued_path.resolve():
        fresh_path.unlink(missing_ok=True)
    issued = ReportVersion(
        report_id=report_id,
        version=next_version,
        format=latest.format,
        path=str(issued_path.resolve()),
        checksum=sha256_path(issued_path),
        source_snapshot_json=list(current_sources.values()),
        issue_snapshot_json=current_issue_snapshots,
        status="issued",
        created_by=actor,
        reviewer=reviewer,
        issued_at=utcnow(),
    )
    session.add(issued)
    report.status = "issued"
    session.flush()
    record_audit(
        session,
        project=project,
        actor=actor,
        action="REPORT_ISSUED",
        entity_type="report",
        entity_id=report_id,
        before={"status": latest.status, "version": latest.version},
        after={
            "status": "issued",
            "version": issued.version,
            "checksum": issued.checksum,
            "reviewer": reviewer,
        },
        metadata={"recipients": payload.recipients, "external_id": payload.external_id},
    )
    session.commit()
    return _version_dict(issued, report)


@router.get("/search")
def universal_search(
    q: str = Query(min_length=1, max_length=200),
    project_id: str = Query(),
    include_historical: bool = False,
    limit: int = Query(default=30, ge=1, le=100),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    _project_or_404(session, project_id)
    revision_rows = ensure_project_revisions(session, project_id)
    current_docs = {
        document.doc_id for document, revision in revision_rows if revision.state == "current"
    }
    needle = f"%{q}%"
    issues = session.scalars(
        select(Issue)
        .where(
            Issue.project_id == project_id,
            or_(
                Issue.issue_id.ilike(needle),
                Issue.title.ilike(needle),
                Issue.description.ilike(needle),
                Issue.room.ilike(needle),
            ),
        )
        .limit(limit)
    ).all()
    doc_query = select(Document).where(
        Document.project_id == project_id,
        or_(Document.filename.ilike(needle), Document.revision.ilike(needle), Document.type.ilike(needle)),
    )
    if not include_historical:
        doc_query = doc_query.where(Document.doc_id.in_(current_docs))
    documents = session.scalars(doc_query.limit(limit)).all()
    chunk_query = (
        select(SpecChunk)
        .join(Document, Document.doc_id == SpecChunk.doc_id)
        .where(Document.project_id == project_id, SpecChunk.text.ilike(needle))
    )
    if not include_historical:
        chunk_query = chunk_query.where(SpecChunk.doc_id.in_(current_docs))
    chunks = session.scalars(chunk_query.limit(limit)).all()
    evidence_all = session.scalars(
        select(FieldEvidence).where(FieldEvidence.project_id == project_id)
    ).all()
    query_lower = q.lower()
    evidence = [
        item
        for item in evidence_all
        if query_lower
        in " ".join(
            [
                item.filename,
                item.author,
                json.dumps(item.location_json or {}, ensure_ascii=False),
                json.dumps(item.metadata_json or {}, ensure_ascii=False),
            ]
        ).lower()
    ][:limit]
    people = session.scalars(
        select(DirectoryMember)
        .where(
            DirectoryMember.project_id == project_id,
            or_(
                DirectoryMember.person_name.ilike(needle),
                DirectoryMember.company.ilike(needle),
                DirectoryMember.trade.ilike(needle),
                DirectoryMember.email.ilike(needle),
            ),
        )
        .limit(limit)
    ).all()
    results: list[dict[str, Any]] = []
    results.extend(
        {
            "type": "issue",
            "id": item.issue_id,
            "title": item.title,
            "subtitle": f"{item.room} · {item.status}",
            "url": f"/projects/{project_id}/issues/{item.issue_id}",
        }
        for item in issues
    )
    results.extend(
        {
            "type": "document",
            "id": item.doc_id,
            "title": item.filename,
            "subtitle": f"{item.type} · Rev {item.revision}",
            "revision": item.revision,
            "url": f"/projects/{project_id}/files?document={item.doc_id}",
        }
        for item in documents
    )
    results.extend(
        {
            "type": "requirement",
            "id": item.chunk_id,
            "title": item.text[:120],
            "subtitle": f"Page {item.page}",
            "document_id": item.doc_id,
            "revision": (item.metadata_json or {}).get("revision", ""),
            "url": f"/projects/{project_id}/drawings?chunk={item.chunk_id}",
        }
        for item in chunks
    )
    results.extend(
        {
            "type": "evidence",
            "id": item.evidence_id,
            "title": item.filename or item.media_type.title(),
            "subtitle": f"{(item.location_json or {}).get('room', 'Unlinked')} · {item.author}",
            "url": f"/projects/{project_id}/evidence?evidence={item.evidence_id}",
        }
        for item in evidence
    )
    results.extend(
        {
            "type": "person",
            "id": item.directory_id,
            "title": item.person_name,
            "subtitle": f"{item.company} · {item.role}",
            "url": f"/projects/{project_id}/directory?member={item.directory_id}",
        }
        for item in people
    )
    session.commit()
    return {
        "query": q,
        "project_id": project_id,
        "scope": "current_and_historical" if include_historical else "current_project_current_revision",
        "count": len(results[:limit]),
        "results": results[:limit],
    }


@router.get("/projects/{project_id}/notifications")
def list_notifications(
    project_id: str,
    unread_only: bool = False,
    recipient: str = "",
    session: Session = Depends(get_session),
) -> list[dict[str, Any]]:
    _project_or_404(session, project_id)
    statement = select(Notification).where(Notification.project_id == project_id)
    if unread_only:
        statement = statement.where(Notification.read_at.is_(None))
    if recipient:
        statement = statement.where(Notification.recipient.in_(["", recipient]))
    items = session.scalars(statement.order_by(Notification.created_at.desc())).all()
    return [
        {
            "notification_id": item.notification_id,
            "project_id": item.project_id,
            "recipient": item.recipient,
            "event_type": item.event_type,
            "title": item.title,
            "body": item.body,
            "entity_type": item.entity_type,
            "entity_id": item.entity_id,
            "channels": list(item.channel_json or []),
            "read_at": item.read_at,
            "created_at": item.created_at,
        }
        for item in items
    ]


@router.patch("/notifications/{notification_id}/read")
def read_notification(
    notification_id: str,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    actor, _ = actor_context(request)
    item = session.get(Notification, notification_id)
    if not item:
        raise HTTPException(status_code=404, detail="notification not found")
    item.read_at = item.read_at or utcnow()
    session.commit()
    return {"notification_id": notification_id, "read_at": item.read_at, "read_by": actor}


def _audit_dict(item: AuditEvent) -> dict[str, Any]:
    return {
        "audit_id": item.audit_id,
        "event_type": item.action,
        "action": item.action,
        "org_id": item.org_id,
        "project_id": item.project_id,
        "actor": item.actor,
        "entity_type": item.entity_type,
        "entity_id": item.entity_id,
        "before": dict(item.before_json or {}),
        "after": dict(item.after_json or {}),
        "metadata": dict(item.metadata_json or {}),
        "timestamp": item.created_at,
        "created_at": item.created_at,
    }


@router.get("/audit-events")
def list_audit_events(
    project_id: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    session: Session = Depends(get_session),
) -> list[dict[str, Any]]:
    statement = select(AuditEvent)
    if project_id:
        _project_or_404(session, project_id)
        statement = statement.where(AuditEvent.project_id == project_id)
    if entity_type:
        statement = statement.where(AuditEvent.entity_type == entity_type)
    if entity_id:
        statement = statement.where(AuditEvent.entity_id == entity_id)
    # Audit streams are chronological so replaying the returned sequence rebuilds
    # entity state deterministically; the final item is the latest transition.
    items = session.scalars(statement.order_by(AuditEvent.created_at.asc()).limit(limit)).all()
    return [_audit_dict(item) for item in items]


@router.get("/projects/{project_id}/audit-events")
def list_project_audit_events(
    project_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    session: Session = Depends(get_session),
) -> list[dict[str, Any]]:
    return list_audit_events(project_id=project_id, limit=limit, session=session)
