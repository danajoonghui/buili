from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class Organization(TimestampMixin, Base):
    __tablename__ = "organizations"

    org_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("org"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    billing_status: Mapped[str] = mapped_column(String, default="pilot")

    projects: Mapped[list["Project"]] = relationship(back_populates="organization")


class User(TimestampMixin, Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("usr"))
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)


class Membership(TimestampMixin, Base):
    __tablename__ = "memberships"

    membership_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("mem"))
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.org_id"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"))
    role: Mapped[str] = mapped_column(String, default="owner")


class UserCredential(TimestampMixin, Base):
    """Additive credential record for legacy databases that already contain users.

    Password material is never stored on ``users``.  Keeping credentials in a
    separate one-to-one table lets deployed pilot databases adopt authentication
    through ``create_all`` without an unsafe in-place column migration.
    """

    __tablename__ = "user_credentials"

    credential_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("cred")
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.user_id"), unique=True, index=True
    )
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    password_changed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class LoginSession(TimestampMixin, Base):
    """Revocable server-side browser session.

    Only a keyed hash of the bearer secret is persisted.  A database disclosure
    therefore does not expose live session cookies.
    """

    __tablename__ = "login_sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("ses"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.org_id"), index=True)
    role: Mapped[str] = mapped_column(String, default="project_manager")
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    user_agent_hash: Mapped[str] = mapped_column(String(64), default="")
    ip_hash: Mapped[str] = mapped_column(String(64), default="")


class Project(TimestampMixin, Base):
    __tablename__ = "projects"

    project_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("prj"))
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.org_id"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    address: Mapped[str] = mapped_column(String, default="")
    project_type: Mapped[str] = mapped_column(String, default="tenant_improvement")
    status: Mapped[str] = mapped_column(String, default="active")

    organization: Mapped[Organization] = relationship(back_populates="projects")
    documents: Mapped[list["Document"]] = relationship(back_populates="project")
    issues: Mapped[list["Issue"]] = relationship(back_populates="project")


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("doc"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"))
    type: Mapped[str] = mapped_column(String, default="plan")
    filename: Mapped[str] = mapped_column(String, default="")
    mime: Mapped[str] = mapped_column(String, default="application/pdf")
    r2_key: Mapped[str] = mapped_column(String, nullable=False)
    hash: Mapped[str] = mapped_column(String, default="")
    revision: Mapped[str] = mapped_column(String, default="A")
    parsed_status: Mapped[str] = mapped_column(String, default="uploaded")
    size: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    project: Mapped[Project] = relationship(back_populates="documents")
    sheets: Mapped[list["Sheet"]] = relationship(back_populates="document")
    chunks: Mapped[list["SpecChunk"]] = relationship(back_populates="document")


class Sheet(TimestampMixin, Base):
    __tablename__ = "sheets"

    sheet_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("sht"))
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id"))
    sheet_number: Mapped[str] = mapped_column(String, default="A-000")
    discipline: Mapped[str] = mapped_column(String, default="architectural")
    page_no: Mapped[int] = mapped_column(Integer, default=1)
    image_key: Mapped[str] = mapped_column(String, default="")
    title: Mapped[str] = mapped_column(String, default="")

    document: Mapped[Document] = relationship(back_populates="sheets")
    entities: Mapped[list["PlanEntity"]] = relationship(back_populates="sheet")


class PlanEntity(TimestampMixin, Base):
    __tablename__ = "plan_entities"

    entity_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("ent"))
    sheet_id: Mapped[str] = mapped_column(ForeignKey("sheets.sheet_id"))
    type: Mapped[str] = mapped_column(String, nullable=False)
    room_id: Mapped[str] = mapped_column(String, default="")
    room: Mapped[str] = mapped_column(String, default="")
    bbox: Mapped[list[float]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String, default="document_text+geometric_locator")
    linked_requirement_ids: Mapped[list[str]] = mapped_column(JSON, default=list)

    sheet: Mapped[Sheet] = relationship(back_populates="entities")


class SpecChunk(TimestampMixin, Base):
    __tablename__ = "spec_chunks"

    chunk_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("chk"))
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id"))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    embedding: Mapped[list[float]] = mapped_column(JSON, default=list)
    page: Mapped[int] = mapped_column(Integer, default=1)
    bbox: Mapped[list[float]] = mapped_column(JSON, default=list)

    document: Mapped[Document] = relationship(back_populates="chunks")


class SiteMedia(TimestampMixin, Base):
    __tablename__ = "site_media"

    media_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("med"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"))
    filename: Mapped[str] = mapped_column(String, default="")
    mime: Mapped[str] = mapped_column(String, default="")
    r2_key: Mapped[str] = mapped_column(String, nullable=False)
    hash: Mapped[str] = mapped_column(String, default="")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Frame(TimestampMixin, Base):
    __tablename__ = "frames"

    frame_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("frm"))
    media_id: Mapped[str] = mapped_column(ForeignKey("site_media.media_id"))
    timestamp: Mapped[float] = mapped_column(Float, default=0.0)
    r2_key: Mapped[str] = mapped_column(String, default="")
    blur_score: Mapped[float] = mapped_column(Float, default=0.0)
    room_hint: Mapped[str] = mapped_column(String, default="")


class Observation(TimestampMixin, Base):
    __tablename__ = "observations"

    observation_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("obs")
    )
    media_id: Mapped[str] = mapped_column(String, default="")
    frame_id: Mapped[str] = mapped_column(String, default="")
    object_type: Mapped[str] = mapped_column(String, nullable=False)
    bbox: Mapped[list[float]] = mapped_column(JSON, default=list)
    text: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)


class Issue(TimestampMixin, Base):
    __tablename__ = "issues"

    issue_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("iss"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"))
    type: Mapped[str] = mapped_column(String, nullable=False)
    discipline: Mapped[str] = mapped_column(String, default="architectural")
    severity: Mapped[str] = mapped_column(String, default="minor")
    room: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="review_ready")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    recommended_action: Mapped[str] = mapped_column(Text, default="")
    assignee: Mapped[str] = mapped_column(String, default="")
    due_date: Mapped[str] = mapped_column(String, default="")
    subcontractor: Mapped[str] = mapped_column(String, default="")
    requirement: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    observation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    plan_location: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    rfi_draft: Mapped[str] = mapped_column(Text, default="")

    project: Mapped[Project] = relationship(back_populates="issues")
    evidence: Mapped[list["IssueEvidence"]] = relationship(back_populates="issue")
    spatial_evidence: Mapped[list["SpatialEvidence"]] = relationship(back_populates="issue")

    @property
    def spatial_context(self) -> dict[str, Any]:
        if not self.spatial_evidence:
            return {}
        latest = sorted(self.spatial_evidence, key=lambda item: item.created_at)[-1]
        features = latest.geometry_features_json or {}
        return {
            "spatial_evidence_id": latest.id,
            "room_graph_id": latest.room_graph_id,
            "design_asset_id": latest.design_asset_id,
            "field_asset_id": latest.field_asset_id,
            "snapshot_uri": latest.snapshot_uri,
            "spatial_note": latest.spatial_note,
            "alignment_confidence": features.get("room_alignment_confidence", 0.0),
            "geometry_confidence": features.get("geometry_confidence", 0.0),
            "geometry_features": features,
        }


class IssueEvidence(TimestampMixin, Base):
    __tablename__ = "issue_evidence"

    evidence_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("evd"))
    issue_id: Mapped[str] = mapped_column(ForeignKey("issues.issue_id"))
    evidence_type: Mapped[str] = mapped_column(String, nullable=False)
    ref_id: Mapped[str] = mapped_column(String, default="")
    r2_key: Mapped[str] = mapped_column(String, default="")
    page: Mapped[int] = mapped_column(Integer, default=1)
    bbox: Mapped[list[float]] = mapped_column(JSON, default=list)
    frame_ts: Mapped[float] = mapped_column(Float, default=0.0)
    label: Mapped[str] = mapped_column(String, default="")

    issue: Mapped[Issue] = relationship(back_populates="evidence")


class PlanGraph(TimestampMixin, Base):
    __tablename__ = "plan_graphs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("pg"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"))
    sheet_id: Mapped[str] = mapped_column(String, default="")
    graph_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    scale_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    source_doc_id: Mapped[str] = mapped_column(String, default="")
    version: Mapped[int] = mapped_column(Integer, default=1)


class SpatialAsset(TimestampMixin, Base):
    __tablename__ = "spatial_assets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("spa"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"))
    type: Mapped[str] = mapped_column(String, default="design_glb")
    uri: Mapped[str] = mapped_column(String, default="")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class FieldPoseFrame(TimestampMixin, Base):
    __tablename__ = "field_pose_frames"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("fpf"))
    media_id: Mapped[str] = mapped_column(String, default="")
    timestamp: Mapped[float] = mapped_column(Float, default=0.0)
    rgb_uri: Mapped[str] = mapped_column(String, default="")
    depth_uri: Mapped[str] = mapped_column(String, default="")
    intrinsics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    pose_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    blur_score: Mapped[float] = mapped_column(Float, default=0.0)
    room_hint: Mapped[str] = mapped_column(String, default="")


class SpatialAlignment(TimestampMixin, Base):
    __tablename__ = "spatial_alignments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("aln"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"))
    plan_graph_id: Mapped[str] = mapped_column(String, default="")
    field_asset_id: Mapped[str] = mapped_column(String, default="")
    transform_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    anchor_pairs_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)


class SpatialEvidence(TimestampMixin, Base):
    __tablename__ = "spatial_evidence"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("spe"))
    issue_id: Mapped[str] = mapped_column(ForeignKey("issues.issue_id"))
    room_graph_id: Mapped[str] = mapped_column(String, default="")
    design_asset_id: Mapped[str] = mapped_column(String, default="")
    field_asset_id: Mapped[str] = mapped_column(String, default="")
    geometry_features_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    snapshot_uri: Mapped[str] = mapped_column(String, default="")
    spatial_note: Mapped[str] = mapped_column(Text, default="")

    issue: Mapped[Issue] = relationship(back_populates="spatial_evidence")


class Job(TimestampMixin, Base):
    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("job"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"))
    state: Mapped[str] = mapped_column(String, default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    input_hash: Mapped[str] = mapped_column(String, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    events: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)


class ModelRun(TimestampMixin, Base):
    __tablename__ = "model_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("run"))
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.job_id"))
    model_name: Mapped[str] = mapped_column(String, default="buili-rule-vlm")
    prompt_hash: Mapped[str] = mapped_column(String, default="")
    input_hash: Mapped[str] = mapped_column(String, default="")
    output_hash: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="completed")
    latency: Mapped[float] = mapped_column(Float, default=0.0)
    cost_estimate: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class UploadIntent(TimestampMixin, Base):
    __tablename__ = "upload_intents"

    upload_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("upl"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"))
    kind: Mapped[str] = mapped_column(String, default="document")
    filename: Mapped[str] = mapped_column(String, default="")
    mime: Mapped[str] = mapped_column(String, default="")
    size: Mapped[int] = mapped_column(Integer, default=0)
    r2_key: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="presigned")


# Workflow/control-plane records intentionally live in additive tables.  The pilot
# database is already deployed as SQLite in a few environments and create_all does
# not add columns to existing tables.  Keeping these records separate lets older
# databases upgrade safely while preserving immutable review, revision and report
# history.


class ProjectProfile(TimestampMixin, Base):
    __tablename__ = "project_profiles"

    profile_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("prf")
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.project_id"), unique=True, index=True
    )
    client: Mapped[str] = mapped_column(String, default="")
    timezone: Mapped[str] = mapped_column(String, default="UTC")
    unit_system: Mapped[str] = mapped_column(String, default="imperial")
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    workflow_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class DirectoryMember(TimestampMixin, Base):
    __tablename__ = "directory_members"

    directory_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("dir")
    )
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), index=True)
    person_name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, default="")
    company: Mapped[str] = mapped_column(String, default="")
    role: Mapped[str] = mapped_column(String, default="field_user")
    trade: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="active")
    notification_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    access_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class DocumentRevision(TimestampMixin, Base):
    __tablename__ = "document_revisions"

    revision_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("rev")
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id"), unique=True, index=True
    )
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), index=True)
    logical_key: Mapped[str] = mapped_column(String, index=True)
    sheet_number: Mapped[str] = mapped_column(String, default="")
    revision: Mapped[str] = mapped_column(String, default="")
    issue_date: Mapped[str] = mapped_column(String, default="")
    discipline: Mapped[str] = mapped_column(String, default="")
    state: Mapped[str] = mapped_column(String, default="unclassified", index=True)
    supersedes_document_id: Mapped[str] = mapped_column(String, default="")
    activated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    source_hash: Mapped[str] = mapped_column(String, default="")
    upload_actor: Mapped[str] = mapped_column(String, default="system")
    parse_version: Mapped[str] = mapped_column(String, default="buili-parser-v1")


class FieldEvidence(TimestampMixin, Base):
    __tablename__ = "field_evidence"

    evidence_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("fld")
    )
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), index=True)
    client_capture_id: Mapped[str] = mapped_column(String, default="", index=True)
    media_id: Mapped[str] = mapped_column(String, default="")
    media_type: Mapped[str] = mapped_column(String, default="photo")
    filename: Mapped[str] = mapped_column(String, default="")
    mime: Mapped[str] = mapped_column(String, default="")
    uri: Mapped[str] = mapped_column(String, default="")
    hash: Mapped[str] = mapped_column(String, default="")
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    author: Mapped[str] = mapped_column(String, default="")
    location_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    location_method: Mapped[str] = mapped_column(String, default="manual")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    quality_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    sufficiency: Mapped[str] = mapped_column(String, default="unreviewed")
    status: Mapped[str] = mapped_column(String, default="unlinked", index=True)


class EvidenceLink(TimestampMixin, Base):
    __tablename__ = "evidence_links"

    link_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("lnk")
    )
    evidence_id: Mapped[str] = mapped_column(ForeignKey("field_evidence.evidence_id"), index=True)
    issue_id: Mapped[str] = mapped_column(ForeignKey("issues.issue_id"), index=True)
    relevance: Mapped[str] = mapped_column(String, default="supports")
    annotation: Mapped[str] = mapped_column(Text, default="")
    linked_by: Mapped[str] = mapped_column(String, default="system")


class IssueWorkflow(TimestampMixin, Base):
    __tablename__ = "issue_workflows"

    workflow_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("iwf")
    )
    issue_id: Mapped[str] = mapped_column(
        ForeignKey("issues.issue_id"), unique=True, index=True
    )
    priority: Mapped[str] = mapped_column(String, default="medium")
    expected_condition: Mapped[str] = mapped_column(Text, default="")
    difference: Mapped[str] = mapped_column(Text, default="")
    recommended_route: Mapped[str] = mapped_column(String, default="observation")
    evidence_gaps_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    source_status: Mapped[str] = mapped_column(String, default="unresolved")
    source_snapshot_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    impact_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    review_status: Mapped[str] = mapped_column(String, default="review_ready", index=True)
    reviewer: Mapped[str] = mapped_column(String, default="")
    version: Mapped[int] = mapped_column(Integer, default=1)


class ReviewRecord(TimestampMixin, Base):
    __tablename__ = "review_records"

    review_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("rvw")
    )
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), index=True)
    issue_id: Mapped[str] = mapped_column(ForeignKey("issues.issue_id"), index=True)
    reviewer: Mapped[str] = mapped_column(String, nullable=False)
    decision: Mapped[str] = mapped_column(String, nullable=False)
    reason_code: Mapped[str] = mapped_column(String, default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    issue_version: Mapped[int] = mapped_column(Integer, default=1)
    snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ReportRecord(TimestampMixin, Base):
    __tablename__ = "report_records"

    report_id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), index=True)
    report_type: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="draft", index=True)
    created_by: Mapped[str] = mapped_column(String, default="system")


class ReportVersion(TimestampMixin, Base):
    __tablename__ = "report_versions"

    version_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("rpv")
    )
    report_id: Mapped[str] = mapped_column(ForeignKey("report_records.report_id"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    format: Mapped[str] = mapped_column(String, default="pdf")
    path: Mapped[str] = mapped_column(String, default="")
    checksum: Mapped[str] = mapped_column(String, nullable=False)
    source_snapshot_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    issue_snapshot_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String, default="draft")
    created_by: Mapped[str] = mapped_column(String, default="system")
    reviewer: Mapped[str] = mapped_column(String, default="")
    issued_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ReportScope(TimestampMixin, Base):
    """Immutable draft selection intent used by the issuance gate.

    Legacy project-wide drafts may issue the approved subset.  A user-curated
    selection is strict: every selected issue must pass the human and evidence
    gates, so the official artifact never silently changes scope.
    """

    __tablename__ = "report_scopes"

    scope_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("rps")
    )
    report_id: Mapped[str] = mapped_column(
        ForeignKey("report_records.report_id"), unique=True, index=True
    )
    issue_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    explicit_selection: Mapped[int] = mapped_column(Integer, default=0)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    audit_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("aud")
    )
    org_id: Mapped[str] = mapped_column(String, default="", index=True)
    project_id: Mapped[str] = mapped_column(String, default="", index=True)
    actor: Mapped[str] = mapped_column(String, default="system")
    action: Mapped[str] = mapped_column(String, index=True)
    entity_type: Mapped[str] = mapped_column(String, index=True)
    entity_id: Mapped[str] = mapped_column(String, index=True)
    before_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    after_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class Notification(TimestampMixin, Base):
    __tablename__ = "notifications"

    notification_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("ntf")
    )
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), index=True)
    recipient: Mapped[str] = mapped_column(String, default="")
    event_type: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(Text, default="")
    entity_type: Mapped[str] = mapped_column(String, default="")
    entity_id: Mapped[str] = mapped_column(String, default="")
    channel_json: Mapped[list[str]] = mapped_column(JSON, default=lambda: ["in_app"])
    read_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
