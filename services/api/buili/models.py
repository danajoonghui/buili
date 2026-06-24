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
