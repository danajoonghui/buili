from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1)
    address: str = ""
    project_type: str = "tenant_improvement"


class ProjectOut(BaseModel):
    project_id: str
    org_id: str
    name: str
    address: str
    project_type: str
    status: str

    model_config = {"from_attributes": True}


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)
    remember_me: bool = False


class UploadPresignRequest(BaseModel):
    project_id: str
    filename: str
    mime: str
    size: int = 0
    kind: Literal["document", "media", "submittal"] = "document"


class UploadPresignResponse(BaseModel):
    upload_id: str
    method: str
    upload_url: str
    complete_url: str
    r2_key: str
    headers: dict[str, str] = Field(default_factory=dict)


class UploadCompleteRequest(BaseModel):
    document_type: str = "plan"
    revision: str = "A"


class AnalyzeRequest(BaseModel):
    priority: Literal["normal", "high"] = "normal"
    force: bool = False
    spatial: bool = True


class JobOut(BaseModel):
    job_id: str
    project_id: str
    state: str
    progress: int
    retry_count: int
    input_hash: str
    error: str
    events: list[dict[str, Any]]

    model_config = {"from_attributes": True}


class DocumentOut(BaseModel):
    doc_id: str
    project_id: str
    type: str
    filename: str
    mime: str
    r2_key: str
    hash: str
    revision: str
    parsed_status: str
    size: int
    metadata_json: dict[str, Any]
    is_current: bool = False
    revision_state: str = "unclassified"
    issue_date: str = ""

    model_config = {"from_attributes": True}


class SiteMediaOut(BaseModel):
    media_id: str
    project_id: str
    filename: str
    mime: str
    r2_key: str
    hash: str
    metadata_json: dict[str, Any]
    download_url: str = ""

    model_config = {"from_attributes": True}


class ObservationOut(BaseModel):
    observation_id: str
    media_id: str
    frame_id: str
    object_type: str
    bbox: list[float]
    text: str
    confidence: float

    model_config = {"from_attributes": True}


class TechnologyStatusOut(BaseModel):
    key: str
    label: str
    status: str
    evidence_count: int
    summary: str


class IssuePatch(BaseModel):
    title: str | None = Field(default=None, min_length=1)
    type: str | None = None
    discipline: str | None = None
    status: str | None = None
    severity: str | None = None
    room: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    assignee: str | None = None
    due_date: str | None = None
    subcontractor: str | None = None
    description: str | None = None
    recommended_action: str | None = None
    requirement: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    plan_location: dict[str, Any] | None = None
    rfi_draft: str | None = None
    priority: Literal["low", "medium", "high", "critical"] | None = None
    expected_condition: str | None = None
    difference: str | None = None
    recommended_route: Literal["rfi", "punch", "pce", "observation", "more_evidence"] | None = None
    evidence_gaps: list[dict[str, Any]] | None = None
    source_status: Literal["current", "stale", "unresolved", "conflicting"] | None = None
    impact: dict[str, Any] | None = None


class EvidenceOut(BaseModel):
    evidence_id: str
    evidence_type: str
    ref_id: str
    r2_key: str
    page: int
    bbox: list[float]
    frame_ts: float
    label: str
    download_url: str = ""

    model_config = {"from_attributes": True}


class IssueOut(BaseModel):
    issue_id: str
    project_id: str
    type: str
    discipline: str
    severity: str
    room: str
    status: str
    confidence: float
    title: str
    description: str
    recommended_action: str
    assignee: str
    due_date: str
    subcontractor: str
    requirement: dict[str, Any]
    observation: dict[str, Any]
    plan_location: dict[str, Any]
    rfi_draft: str
    evidence: list[EvidenceOut] = Field(default_factory=list)
    spatial_context: dict[str, Any] = Field(default_factory=dict)
    priority: str = "medium"
    expected_condition: str = ""
    difference: str = ""
    recommended_route: str = "observation"
    evidence_gaps: list[dict[str, Any]] = Field(default_factory=list)
    source_status: str = "unresolved"
    review_status: str = "review_ready"
    issue_version: int = 1
    risk_flags: list[str] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class RfiOut(BaseModel):
    issue_id: str
    title: str
    markdown: str
    readiness: dict[str, Any] = Field(default_factory=dict)


class ReportRequest(BaseModel):
    report_type: Literal["punch", "co_evidence", "rfi"] = "punch"
    format: Literal["pdf", "csv", "xlsx", "md"] = "pdf"
    issue_ids: list[str] | None = Field(default=None, min_length=1, max_length=250)


class ReportOut(BaseModel):
    report_id: str
    report_type: str
    format: str
    path: str
    download_url: str
    issue_ids: list[str] = Field(default_factory=list)
    readiness: list[dict[str, Any]] = Field(default_factory=list)
    can_issue: bool = False


class OverlayOut(BaseModel):
    project_id: str
    sheets: list[dict[str, Any]]
    pins: list[dict[str, Any]]
    regions: list[dict[str, Any]]


class ProjectPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    address: str | None = None
    project_type: str | None = None
    status: Literal["setup", "active", "on_hold", "archived"] | None = None
    client: str | None = None
    timezone: str | None = None
    unit_system: Literal["imperial", "metric"] | None = None


class ProjectSettingsPatch(BaseModel):
    timezone: str | None = None
    unit_system: Literal["imperial", "metric"] | None = None
    settings: dict[str, Any] | None = None
    workflow: dict[str, Any] | None = None


class DirectoryCreate(BaseModel):
    person_name: str = Field(min_length=1)
    email: str = ""
    company: str = ""
    role: str = "field_user"
    trade: str = ""
    status: Literal["invited", "active", "disabled"] = "active"
    notification: dict[str, Any] = Field(default_factory=dict)
    access_expires_at: datetime | None = None


class DirectoryPatch(BaseModel):
    person_name: str | None = Field(default=None, min_length=1)
    email: str | None = None
    company: str | None = None
    role: str | None = None
    trade: str | None = None
    status: Literal["invited", "active", "disabled"] | None = None
    notification: dict[str, Any] | None = None
    access_expires_at: datetime | None = None


class RevisionActivateRequest(BaseModel):
    logical_key: str | None = None
    sheet_number: str | None = None
    issue_date: str | None = None
    discipline: str | None = None


class EvidenceSyncRequest(BaseModel):
    project_id: str = ""
    client_capture_id: str = ""
    client_id: str = ""
    media_id: str = ""
    media_type: Literal["photo", "video", "audio", "measurement"] = "photo"
    evidence_type: Literal["photo", "video", "audio", "measurement"] | None = None
    filename: str = ""
    mime: str = ""
    uri: str = ""
    hash: str = ""
    sha256: str = ""
    content_base64: str = ""
    captured_at: datetime | None = None
    author: str = ""
    location: dict[str, Any] = Field(default_factory=dict)
    location_method: str = "manual"
    metadata: dict[str, Any] = Field(default_factory=dict)
    observation: Any = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)
    sufficiency: Literal["unreviewed", "sufficient", "insufficient"] = "unreviewed"


class EvidencePatch(BaseModel):
    author: str | None = None
    location: dict[str, Any] | None = None
    location_method: str | None = None
    metadata: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    sufficiency: Literal["unreviewed", "sufficient", "insufficient"] | None = None


class EvidenceLocationPatch(BaseModel):
    location: dict[str, Any]
    location_method: str = "manual"


class EvidenceLinkRequest(BaseModel):
    issue_id: str
    relevance: Literal["supports", "contradicts", "context", "completion"] = "supports"
    annotation: str = ""


class IssueCreate(BaseModel):
    project_id: str
    title: str = Field(min_length=1)
    type: str = "observation"
    discipline: str = "architectural"
    severity: Literal["blocker", "major", "minor", "informational"] = "minor"
    room: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    description: str = ""
    recommended_action: str = ""
    assignee: str = ""
    due_date: str = ""
    subcontractor: str = ""
    requirement: dict[str, Any] = Field(default_factory=dict)
    observation: dict[str, Any] = Field(default_factory=dict)
    plan_location: dict[str, Any] = Field(default_factory=dict)
    rfi_draft: str = ""
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    expected_condition: str = ""
    difference: str = ""
    recommended_route: Literal["rfi", "punch", "pce", "observation", "more_evidence"] = "observation"
    evidence_gaps: list[dict[str, Any]] = Field(default_factory=list)
    source_references: list[dict[str, Any]] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    impact: dict[str, Any] = Field(default_factory=dict)


class ReviewCreate(BaseModel):
    decision: Literal["approve", "reject", "request_evidence"]
    reviewer: str = ""
    reason: str = ""
    reason_code: str = ""
    evidence_gaps: list[dict[str, Any]] = Field(default_factory=list)


class RequestEvidenceCreate(BaseModel):
    requested_by: str = ""
    reason: str = Field(min_length=1)
    evidence_gaps: list[dict[str, Any]] = Field(default_factory=list)
    recipient: str = ""


class ReportExportRequest(BaseModel):
    recipients: list[str] = Field(default_factory=list)
    external_id: str = ""
