from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import AsyncIterator

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .config import get_settings
from .database import get_session, init_db
from .database import SessionLocal
from .models import (
    Document,
    Frame,
    Issue,
    Job,
    Observation,
    Project,
    SiteMedia,
    SpecChunk,
    UploadIntent,
    new_id,
)
from .pipeline import (
    cosine_search,
    create_job_for_project,
    ensure_demo_project,
    overlay_for_project,
    run_analysis_job,
)
from .reports import build_markdown_rfi, build_report
from .schemas import (
    AnalyzeRequest,
    DocumentOut,
    ObservationOut,
    IssueOut,
    IssuePatch,
    JobOut,
    OverlayOut,
    ProjectCreate,
    ProjectOut,
    ReportOut,
    ReportRequest,
    RfiOut,
    SiteMediaOut,
    TechnologyStatusOut,
    UploadCompleteRequest,
    UploadPresignRequest,
    UploadPresignResponse,
)
from .storage import file_sha256, object_path, save_upload

settings = get_settings()

DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".docx", ".xlsx"}
MEDIA_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".mp4",
    ".mov",
    ".m4v",
    ".mp3",
    ".m4a",
    ".wav",
}
DOCUMENT_TYPES = {"plan", "spec", "submittal", "rfi", "change_order", "other"}


def _clean_filename(filename: str) -> str:
    clean = Path(filename).name.strip().replace("\x00", "")
    if not clean or clean in {".", ".."}:
        raise HTTPException(status_code=400, detail="filename is required")
    return clean


def _validate_upload_request(filename: str, mime: str, size: int, kind: str) -> str:
    clean = _clean_filename(filename)
    suffix = Path(clean).suffix.lower()
    if size <= 0:
        raise HTTPException(status_code=400, detail="upload size must be greater than zero")
    if size > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="upload exceeds maximum allowed size")
    if kind in {"document", "submittal"} and suffix not in DOCUMENT_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"unsupported document type: {suffix or 'none'}")
    if kind == "media" and suffix not in MEDIA_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"unsupported media type: {suffix or 'none'}")
    if mime and "/" not in mime:
        raise HTTPException(status_code=400, detail="invalid mime type")
    return clean


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    with SessionLocal() as session:
        ensure_demo_project(session)
    yield


app = FastAPI(title="Buili API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "buili-api"}


@app.get("/v1/projects", response_model=list[ProjectOut])
def list_projects(session: Session = Depends(get_session)) -> list[Project]:
    return list(session.scalars(select(Project).order_by(Project.created_at.desc())).all())


@app.post("/v1/projects", response_model=ProjectOut)
def create_project(payload: ProjectCreate, session: Session = Depends(get_session)) -> Project:
    demo = ensure_demo_project(session)
    project = Project(
        org_id=demo.org_id,
        name=payload.name,
        address=payload.address,
        project_type=payload.project_type,
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@app.post("/v1/uploads/presign", response_model=UploadPresignResponse)
def presign_upload(
    payload: UploadPresignRequest, session: Session = Depends(get_session)
) -> UploadPresignResponse:
    project = session.get(Project, payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    filename = _validate_upload_request(payload.filename, payload.mime, payload.size, payload.kind)
    upload_id = new_id("upl")
    r2_key = (
        f"org/{project.org_id}/project/{project.project_id}/raw/{upload_id}_"
        f"{filename}"
    )
    intent = UploadIntent(
        upload_id=upload_id,
        project_id=project.project_id,
        kind=payload.kind,
        filename=filename,
        mime=payload.mime,
        size=payload.size,
        r2_key=r2_key,
    )
    session.add(intent)
    session.commit()
    return UploadPresignResponse(
        upload_id=upload_id,
        method="POST",
        upload_url=f"{settings.public_base_url}/v1/uploads/{upload_id}",
        complete_url=f"{settings.public_base_url}/v1/uploads/{upload_id}/complete",
        r2_key=r2_key,
        headers={},
    )


@app.post("/v1/uploads/{upload_id}")
async def upload_file(
    upload_id: str,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> dict[str, object]:
    intent = session.get(UploadIntent, upload_id)
    if not intent:
        raise HTTPException(status_code=404, detail="upload intent not found")
    if intent.status != "presigned":
        raise HTTPException(status_code=409, detail=f"upload is already {intent.status}")
    try:
        size, digest = await save_upload(file, intent.r2_key, max_bytes=settings.max_upload_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    if intent.size and size != intent.size:
        object_path(intent.r2_key).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="received size does not match presigned size")
    intent.status = "uploaded"
    intent.size = size
    session.commit()
    return {"upload_id": upload_id, "size": size, "sha256": digest, "r2_key": intent.r2_key}


@app.post("/v1/uploads/{upload_id}/complete")
def complete_upload(
    upload_id: str,
    payload: UploadCompleteRequest,
    session: Session = Depends(get_session),
) -> dict[str, str]:
    intent = session.get(UploadIntent, upload_id)
    if not intent:
        raise HTTPException(status_code=404, detail="upload intent not found")
    if intent.status != "uploaded":
        raise HTTPException(status_code=409, detail="upload has not been received")
    if intent.kind in {"document", "submittal"} and payload.document_type not in DOCUMENT_TYPES:
        raise HTTPException(status_code=400, detail="unsupported document type")
    digest = file_sha256(object_path(intent.r2_key))

    if intent.kind in {"document", "submittal"}:
        doc = Document(
            project_id=intent.project_id,
            type=payload.document_type if intent.kind == "document" else "submittal",
            filename=intent.filename,
            mime=intent.mime,
            r2_key=intent.r2_key,
            hash=digest,
            revision=payload.revision,
            parsed_status="uploaded",
            size=intent.size,
        )
        session.add(doc)
        intent.status = "completed"
        session.commit()
        return {"status": "completed", "document_id": doc.doc_id}

    media = SiteMedia(
        project_id=intent.project_id,
        filename=intent.filename,
        mime=intent.mime,
        r2_key=intent.r2_key,
        hash=digest,
        metadata_json={"source": "user_upload", "upload_id": intent.upload_id, "size": intent.size},
    )
    session.add(media)
    intent.status = "completed"
    session.commit()
    return {"status": "completed", "media_id": media.media_id}


@app.post("/v1/projects/{project_id}/analyze", response_model=JobOut)
def analyze_project(
    project_id: str,
    _: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
) -> Job:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    job = create_job_for_project(project, session)
    background_tasks.add_task(_run_job_task, job.job_id)
    return job


def _run_job_task(job_id: str) -> None:
    from .database import SessionLocal

    with SessionLocal() as session:
        run_analysis_job(job_id, session)


@app.get("/v1/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, session: Session = Depends(get_session)) -> Job:
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/v1/projects/{project_id}/jobs/latest", response_model=JobOut | None)
def latest_project_job(project_id: str, session: Session = Depends(get_session)) -> Job | None:
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="project not found")
    return session.scalar(
        select(Job).where(Job.project_id == project_id).order_by(Job.created_at.desc())
    )


@app.get("/v1/projects/{project_id}/documents", response_model=list[DocumentOut])
def list_documents(project_id: str, session: Session = Depends(get_session)) -> list[Document]:
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="project not found")
    return list(
        session.scalars(select(Document).where(Document.project_id == project_id).order_by(Document.created_at.desc())).all()
    )


@app.get("/v1/projects/{project_id}/media", response_model=list[SiteMediaOut])
def list_media(project_id: str, session: Session = Depends(get_session)) -> list[SiteMedia]:
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="project not found")
    return list(
        session.scalars(select(SiteMedia).where(SiteMedia.project_id == project_id).order_by(SiteMedia.created_at.desc())).all()
    )


@app.get("/v1/projects/{project_id}/observations", response_model=list[ObservationOut])
def list_observations(project_id: str, session: Session = Depends(get_session)) -> list[Observation]:
    media_ids = select(SiteMedia.media_id).where(SiteMedia.project_id == project_id)
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="project not found")
    return list(
        session.scalars(
            select(Observation)
            .where(Observation.media_id.in_(media_ids))
            .order_by(Observation.confidence.desc())
        ).all()
    )


@app.get("/v1/projects/{project_id}/technology-status", response_model=list[TechnologyStatusOut])
def technology_status(project_id: str, session: Session = Depends(get_session)) -> list[TechnologyStatusOut]:
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="project not found")
    docs = session.scalars(select(Document).where(Document.project_id == project_id)).all()
    chunks = session.scalars(select(SpecChunk).join(Document).where(Document.project_id == project_id)).all()
    media = session.scalars(select(SiteMedia).where(SiteMedia.project_id == project_id)).all()
    media_ids = select(SiteMedia.media_id).where(SiteMedia.project_id == project_id)
    observations = session.scalars(select(Observation).where(Observation.media_id.in_(media_ids))).all()
    issues = session.scalars(select(Issue).where(Issue.project_id == project_id)).all()
    jobs = session.scalars(select(Job).where(Job.project_id == project_id)).all()
    frames = session.scalars(select(Frame).where(Frame.media_id.in_(media_ids))).all()
    return [
        TechnologyStatusOut(
            key="pdf_rag",
            label="PDF drawing/spec RAG analysis",
            status="ready" if docs and chunks else "needs_input",
            evidence_count=len(chunks),
            summary=f"{len(docs)} documents parsed into {len(chunks)} searchable citation chunks.",
        ),
        TechnologyStatusOut(
            key="media_recognition",
            label="Field photo/video construction element recognition",
            status="ready" if media and observations else "needs_media",
            evidence_count=len(observations),
            summary=f"{len(media)} media files, {len(frames)} derived frames, {len(observations)} recognized observations.",
        ),
        TechnologyStatusOut(
            key="mismatch_candidates",
            label="Drawing-field mismatch candidate generation",
            status="ready" if issues else "needs_review",
            evidence_count=len(issues),
            summary=f"{len(issues)} review candidates generated with requirement, observation, and plan-location links.",
        ),
        TechnologyStatusOut(
            key="reports",
            label="Punch list, RFI, and change order report generation",
            status="ready" if issues else "needs_issues",
            evidence_count=len([issue for issue in issues if issue.rfi_draft]),
            summary="Punch PDF/CSV, RFI draft, and CO evidence PDF generation are available from issue data.",
        ),
        TechnologyStatusOut(
            key="web_review",
            label="Web-based field issue review and management",
            status="ready" if jobs else "needs_job",
            evidence_count=len(jobs),
            summary=f"{len(jobs)} pipeline runs available for browser-based issue review, approval, RFI, and reporting.",
        ),
    ]


@app.get("/v1/projects/{project_id}/issues", response_model=list[IssueOut])
def list_issues(project_id: str, session: Session = Depends(get_session)) -> list[Issue]:
    issues = session.scalars(
        select(Issue)
        .options(selectinload(Issue.evidence))
        .where(Issue.project_id == project_id)
        .order_by(Issue.confidence.desc())
    ).all()
    return list(issues)


@app.patch("/v1/issues/{issue_id}", response_model=IssueOut)
def update_issue(
    issue_id: str, payload: IssuePatch, session: Session = Depends(get_session)
) -> Issue:
    issue = session.scalar(
        select(Issue).options(selectinload(Issue.evidence)).where(Issue.issue_id == issue_id)
    )
    if not issue:
        raise HTTPException(status_code=404, detail="issue not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(issue, key, value)
    session.commit()
    session.refresh(issue)
    return issue


@app.post("/v1/issues/{issue_id}/rfi", response_model=RfiOut)
def create_rfi(issue_id: str, session: Session = Depends(get_session)) -> RfiOut:
    issue = session.get(Issue, issue_id)
    if not issue:
        raise HTTPException(status_code=404, detail="issue not found")
    return RfiOut(issue_id=issue.issue_id, title=issue.title, markdown=build_markdown_rfi(issue))


@app.post("/v1/projects/{project_id}/reports", response_model=ReportOut)
def create_report(
    project_id: str,
    payload: ReportRequest,
    session: Session = Depends(get_session),
) -> ReportOut:
    try:
        report_id, path = build_report(session, project_id, payload.report_type, payload.format)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    rel = path.relative_to(settings.storage_root / "reports")
    return ReportOut(
        report_id=report_id,
        report_type=payload.report_type,
        format=payload.format,
        path=str(path),
        download_url=f"{settings.public_base_url}/v1/reports/{rel.as_posix()}",
    )


@app.get("/v1/reports/{path:path}")
def download_report(path: str) -> FileResponse:
    report_path = settings.storage_root / "reports" / path
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="report not found")
    return FileResponse(report_path)


@app.get("/v1/projects/{project_id}/plan-overlay", response_model=OverlayOut)
def get_overlay(project_id: str, session: Session = Depends(get_session)) -> dict:
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="project not found")
    return overlay_for_project(project_id, session)


@app.get("/v1/projects/{project_id}/rag/search")
def rag_search(project_id: str, q: str, session: Session = Depends(get_session)) -> dict[str, object]:
    chunks = session.scalars(
        select(SpecChunk).join(Document).where(Document.project_id == project_id)
    ).all()
    return {
        "query": q,
        "filters": {"project_id": project_id},
        "retrieval": {"bm25_top_k": 50, "vector_top_k": 50, "rerank_top_k": 8},
        "returned_context": cosine_search(q, list(chunks), top_k=8),
    }


@app.get("/v1/metrics")
def metrics(session: Session = Depends(get_session)) -> dict[str, int]:
    return {
        "projects": len(session.scalars(select(Project)).all()),
        "jobs": len(session.scalars(select(Job)).all()),
        "issues": len(session.scalars(select(Issue)).all()),
    }


if __name__ == "__main__":
    uvicorn.run("services.api.buili.main:app", host=settings.api_host, port=settings.api_port, reload=False)
