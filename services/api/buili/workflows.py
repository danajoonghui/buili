from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import (
    AuditEvent,
    Document,
    DocumentRevision,
    EvidenceLink,
    FieldEvidence,
    Issue,
    IssueWorkflow,
    PlanEntity,
    Project,
    ProjectProfile,
    Sheet,
    SpecChunk,
)

APPROVER_ROLES = {"project_manager", "pm", "admin", "org_admin", "owner"}
EXPORT_ROLES = APPROVER_ROLES | {"project_engineer", "pe"}
EDIT_ROLES = EXPORT_ROLES | {"superintendent", "foreman", "field_user"}


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def actor_context(request: Request) -> tuple[str, str]:
    """Resolve identity from the authenticated server-side session.

    Header identity is retained only for the unauthenticated local/test mode.  It
    is never consulted when first-party authentication is required, so a browser
    cannot grant itself a higher role through a spoofed request header.
    """

    from .auth import current_principal

    principal = getattr(request.state, "principal", None) or current_principal()
    if principal:
        return principal.email, principal.role

    actor_header = request.headers.get("x-buili-actor")
    role_header = request.headers.get("x-buili-role")
    if get_settings().auth_required:
        raise HTTPException(status_code=401, detail="authentication required")
    if get_settings().require_auth_headers and (not actor_header or not role_header):
        raise HTTPException(status_code=401, detail="trusted identity headers are required")
    actor = (actor_header or "system").strip() or "system"
    role = (role_header or "project_manager").strip().lower()
    return actor, role


def require_role(role: str, allowed: set[str], action: str) -> None:
    if role.lower() not in allowed:
        raise HTTPException(status_code=403, detail=f"role {role!r} cannot {action}")


def issue_action_blockers(
    session: Session,
    issue: Issue,
    workflow: IssueWorkflow,
    *,
    report_type: str | None = None,
    require_approval: bool = False,
) -> list[dict[str, str]]:
    """Return deterministic, user-correctable blockers for an official action."""

    blockers: list[dict[str, str]] = []

    def add(code: str, field: str, message: str) -> None:
        blockers.append({"code": code, "field": field, "message": message})

    requirement = issue.requirement or {}
    observation = issue.observation or {}
    location = issue.plan_location or {}
    if not issue.room.strip() or not (
        location.get("sheet_id")
        or location.get("sheet_number")
        or (location.get("x") is not None and location.get("y") is not None)
    ):
        add("location_required", "plan_location", "Pin the issue to a room and drawing location.")
    if not str(requirement.get("text") or "").strip():
        add("requirement_required", "requirement.text", "Add the expected contract requirement.")
    if not str(observation.get("text") or "").strip():
        add("observation_required", "observation.text", "Describe the observed field condition.")
    if workflow.source_status != "current":
        add(
            "current_source_required",
            "source_status",
            "Link and verify a citation from the current drawing/spec revision.",
        )
    evidence_links = list(
        session.scalars(select(EvidenceLink).where(EvidenceLink.issue_id == issue.issue_id)).all()
    )
    sufficient_evidence = False
    for link in evidence_links:
        field = session.get(FieldEvidence, link.evidence_id)
        if (
            field
            and field.project_id == issue.project_id
            and field.sufficiency != "insufficient"
            and bool(field.hash)
            and bool(field.location_json)
        ):
            sufficient_evidence = True
            break
    if not sufficient_evidence:
        add(
            "field_evidence_required",
            "evidence",
            "Link location-confirmed field evidence with an integrity hash.",
        )

    if report_type == "rfi":
        if not issue.rfi_draft.strip() or "?" not in issue.rfi_draft:
            add("rfi_question_required", "rfi_draft", "Write one explicit design question.")
    elif report_type == "punch":
        if not (issue.subcontractor.strip() or issue.assignee.strip()):
            add("responsible_party_required", "assignee", "Assign the responsible party or trade.")
        if not issue.due_date.strip():
            add("due_date_required", "due_date", "Set a punch completion due date.")
        if not (workflow.expected_condition.strip() or str(requirement.get("text") or "").strip()):
            add(
                "expected_condition_required",
                "expected_condition",
                "State the expected completed condition.",
            )
    elif report_type == "co_evidence":
        impact = workflow.impact_json or {}
        if not any(impact.get(key) for key in ("cost", "schedule", "scope", "safety")):
            add(
                "impact_basis_required",
                "impact",
                "Record a scope, cost, schedule, or safety impact basis.",
            )
    if require_approval and workflow.review_status != "approved":
        add("human_approval_required", "review_status", "An authorized reviewer must approve it.")
    return blockers


def record_audit(
    session: Session,
    *,
    project: Project | None,
    actor: str,
    action: str,
    entity_type: str,
    entity_id: str,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditEvent:
    event = AuditEvent(
        org_id=project.org_id if project else "",
        project_id=project.project_id if project else "",
        actor=actor,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        before_json=jsonable_encoder(before or {}),
        after_json=jsonable_encoder(after or {}),
        metadata_json=jsonable_encoder(metadata or {}),
    )
    session.add(event)
    return event


def project_snapshot(project: Project, profile: ProjectProfile | None = None) -> dict[str, Any]:
    return {
        "project_id": project.project_id,
        "org_id": project.org_id,
        "name": project.name,
        "address": project.address,
        "project_type": project.project_type,
        "status": project.status,
        "client": profile.client if profile else "",
        "timezone": profile.timezone if profile else "UTC",
        "unit_system": profile.unit_system if profile else "imperial",
        "settings": dict(profile.settings_json or {}) if profile else {},
        "workflow": dict(profile.workflow_json or {}) if profile else {},
    }


def get_or_create_project_profile(session: Session, project_id: str) -> ProjectProfile:
    profile = session.scalar(select(ProjectProfile).where(ProjectProfile.project_id == project_id))
    if profile:
        return profile
    profile = ProjectProfile(
        project_id=project_id,
        settings_json={
            "floor_naming": "level_name",
            "disciplines": ["architectural", "electrical", "mechanical", "plumbing"],
            "issue_types": ["rfi", "punch", "change_evidence", "observation"],
            "retention_days": 2555,
        },
        workflow_json={
            "review_steps": ["project_manager"],
            "sla_hours": 48,
            "high_risk_second_reviewer": True,
        },
    )
    session.add(profile)
    session.flush()
    return profile


def issue_snapshot(issue: Issue, workflow: IssueWorkflow | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "issue_id": issue.issue_id,
        "project_id": issue.project_id,
        "type": issue.type,
        "discipline": issue.discipline,
        "severity": issue.severity,
        "room": issue.room,
        "status": issue.status,
        "confidence": issue.confidence,
        "title": issue.title,
        "description": issue.description,
        "recommended_action": issue.recommended_action,
        "assignee": issue.assignee,
        "due_date": issue.due_date,
        "subcontractor": issue.subcontractor,
        "requirement": dict(issue.requirement or {}),
        "observation": dict(issue.observation or {}),
        "plan_location": dict(issue.plan_location or {}),
        "rfi_draft": issue.rfi_draft,
    }
    if workflow:
        payload["workflow"] = {
            "priority": workflow.priority,
            "expected_condition": workflow.expected_condition,
            "difference": workflow.difference,
            "recommended_route": workflow.recommended_route,
            "evidence_gaps": list(workflow.evidence_gaps_json or []),
            "source_status": workflow.source_status,
            "source_snapshot": list(workflow.source_snapshot_json or []),
            "impact": dict(workflow.impact_json or {}),
            "review_status": workflow.review_status,
            "reviewer": workflow.reviewer,
            "version": workflow.version,
        }
    return payload


def _route_for_issue(issue: Issue) -> str:
    if issue.type == "potential_change_order":
        return "pce"
    if issue.type in {"location_mismatch", "spec_mismatch"}:
        return "rfi"
    if issue.type in {"missing_item", "count_mismatch", "coverage_check"}:
        return "punch"
    return "more_evidence" if issue.type == "unverified" else "observation"


def get_or_create_issue_workflow(session: Session, issue: Issue) -> IssueWorkflow:
    workflow = session.scalar(select(IssueWorkflow).where(IssueWorkflow.issue_id == issue.issue_id))
    if workflow:
        return workflow
    gaps: list[dict[str, Any]] = []
    observation = issue.observation or {}
    if not observation.get("media_id") or observation.get("media_id") == "field_verification_pending":
        gaps.append({"type": "field_evidence", "message": "Upload location-confirmed field evidence."})
    if not issue.requirement or not issue.requirement.get("source"):
        gaps.append({"type": "source", "message": "Link a current drawing or specification citation."})
    workflow = IssueWorkflow(
        issue_id=issue.issue_id,
        priority="high" if issue.severity in {"blocker", "major"} else "medium",
        expected_condition=str((issue.requirement or {}).get("text", "")),
        difference=issue.description,
        recommended_route=_route_for_issue(issue),
        evidence_gaps_json=gaps,
        source_status="unresolved",
        review_status=issue.status if issue.status else "review_ready",
    )
    session.add(workflow)
    session.flush()
    return workflow


def _normalized_logical_key(document: Document, session: Session) -> tuple[str, str, str]:
    metadata = document.metadata_json or {}
    sheet_number = str(metadata.get("sheet_number") or "").strip()
    discipline = str(metadata.get("discipline") or "").strip()
    if not sheet_number:
        sheet = session.scalar(
            select(Sheet).where(Sheet.doc_id == document.doc_id).order_by(Sheet.page_no.asc())
        )
        if sheet:
            sheet_number = sheet.sheet_number
            discipline = discipline or sheet.discipline
    explicit = str(metadata.get("logical_key") or "").strip()
    if explicit:
        return explicit.lower(), sheet_number, discipline
    if sheet_number:
        return f"{document.type}:{sheet_number}".lower(), sheet_number, discipline
    stem = Path(document.filename).stem.lower()
    revision = re.escape(str(document.revision or "").lower())
    stem = re.sub(r"(?:^|[-_\s])rev(?:ision)?[-_\s]*[a-z0-9.]+(?:$|[-_\s])", "-", stem)
    if revision:
        stem = re.sub(rf"(?:^|[-_\s]){revision}(?:$|[-_\s])", "-", stem)
    stem = re.sub(r"[-_\s]+", "-", stem).strip("-") or Path(document.filename).stem.lower()
    return f"{document.type}:{stem}", sheet_number, discipline


def get_or_create_revision(
    session: Session,
    document: Document,
    *,
    actor: str = "system",
    state: str = "unclassified",
) -> DocumentRevision:
    record = session.scalar(
        select(DocumentRevision).where(DocumentRevision.document_id == document.doc_id)
    )
    if record:
        return record
    logical_key, sheet_number, discipline = _normalized_logical_key(document, session)
    record = DocumentRevision(
        document_id=document.doc_id,
        project_id=document.project_id,
        logical_key=logical_key,
        sheet_number=sheet_number,
        revision=document.revision,
        issue_date=str((document.metadata_json or {}).get("issue_date") or ""),
        discipline=discipline,
        state=state,
        source_hash=document.hash,
        upload_actor=actor,
    )
    session.add(record)
    session.flush()
    return record


def source_snapshot_for_issue(session: Session, issue: Issue) -> list[dict[str, Any]]:
    document_ids: dict[str, str] = {}
    for item in list(issue.evidence or []):
        document_id = ""
        if item.evidence_type == "spec_chunk":
            chunk = session.get(SpecChunk, item.ref_id)
            document_id = chunk.doc_id if chunk else ""
        elif item.evidence_type in {"sheet", "plan_entity"}:
            entity = session.get(PlanEntity, item.ref_id)
            if entity:
                sheet = session.get(Sheet, entity.sheet_id)
                document_id = sheet.doc_id if sheet else ""
            else:
                sheet = session.get(Sheet, item.ref_id)
                document_id = sheet.doc_id if sheet else ""
        elif item.evidence_type in {"document", "source"}:
            document_id = item.ref_id
        if document_id:
            document_ids[document_id] = item.evidence_type

    source_name = str((issue.requirement or {}).get("source") or "")
    if source_name:
        matches = session.scalars(
            select(Sheet)
            .join(Document, Document.doc_id == Sheet.doc_id)
            .where(Document.project_id == issue.project_id, Sheet.sheet_number == source_name)
        ).all()
        for sheet in matches:
            document_ids[sheet.doc_id] = "requirement"

    snapshot: list[dict[str, Any]] = []
    for document_id, relation in document_ids.items():
        document = session.get(Document, document_id)
        if not document or document.project_id != issue.project_id:
            continue
        revision = get_or_create_revision(session, document)
        snapshot.append(
            {
                "document_id": document.doc_id,
                "filename": document.filename,
                "type": document.type,
                "revision": document.revision,
                "issue_date": revision.issue_date,
                "state": revision.state,
                "source_hash": document.hash,
                "logical_key": revision.logical_key,
                "sheet_number": revision.sheet_number,
                "relation": relation,
            }
        )
    return sorted(snapshot, key=lambda item: (str(item["logical_key"]), str(item["document_id"])))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
