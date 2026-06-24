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

    model_config = {"from_attributes": True}


class SiteMediaOut(BaseModel):
    media_id: str
    project_id: str
    filename: str
    mime: str
    r2_key: str
    hash: str
    metadata_json: dict[str, Any]

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
    status: str | None = None
    severity: str | None = None
    room: str | None = None
    assignee: str | None = None
    due_date: str | None = None
    subcontractor: str | None = None
    description: str | None = None
    recommended_action: str | None = None


class EvidenceOut(BaseModel):
    evidence_id: str
    evidence_type: str
    ref_id: str
    r2_key: str
    page: int
    bbox: list[float]
    frame_ts: float
    label: str

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

    model_config = {"from_attributes": True}


class RfiOut(BaseModel):
    issue_id: str
    title: str
    markdown: str


class ReportRequest(BaseModel):
    report_type: Literal["punch", "co_evidence", "rfi"] = "punch"
    format: Literal["pdf", "csv", "xlsx", "md"] = "pdf"


class ReportOut(BaseModel):
    report_id: str
    report_type: str
    format: str
    path: str
    download_url: str


class OverlayOut(BaseModel):
    project_id: str
    sheets: list[dict[str, Any]]
    pins: list[dict[str, Any]]
    regions: list[dict[str, Any]]
