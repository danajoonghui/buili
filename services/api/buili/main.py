from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, text
from sqlalchemy.orm import Session, selectinload

from .auth import (
    bind_principal,
    ensure_pilot_identity,
    request_path_requires_auth,
    reset_principal,
    resolve_request_principal,
    validate_production_auth_settings,
)
from .auth import (
    router as auth_router,
)
from .config import get_settings
from .database import SessionLocal, get_session, init_db
from .gpu import force_gpu_7, gpu_policy
from .models import (
    Document,
    DocumentRevision,
    Frame,
    Issue,
    Job,
    Observation,
    PlanGraph,
    Project,
    ReportRecord,
    ReportScope,
    ReportVersion,
    SiteMedia,
    SpatialAsset,
    SpatialEvidence,
    SpecChunk,
    UploadIntent,
    new_id,
)
from .pipeline import (
    create_job_for_project,
    ensure_demo_project,
    overlay_for_project,
    rag_answer,
    run_analysis_job,
)
from .reports import build_markdown_rfi, build_report
from .schemas import (
    AnalyzeRequest,
    DocumentOut,
    IssueOut,
    IssuePatch,
    JobOut,
    ObservationOut,
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
from .spatial.router import router as spatial_router
from .storage import file_sha256, object_path, save_upload
from .workflow_router import ensure_project_revisions, issue_detail, refresh_review_readiness
from .workflow_router import router as workflow_router
from .workflows import (
    EDIT_ROLES,
    actor_context,
    get_or_create_issue_workflow,
    get_or_create_project_profile,
    issue_action_blockers,
    issue_snapshot,
    project_snapshot,
    record_audit,
    require_role,
    sha256_path,
    source_snapshot_for_issue,
)

force_gpu_7()
settings = get_settings()
REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_PUBLIC_ROOT = REPO_ROOT / "apps" / "web" / "public"
API_STATIC_ROOT = Path(__file__).resolve().parent / "static"
VLM_ARTIFACT_CANDIDATES = [
    "buili_internvl35_14b_plus_open_lora",
    "buili_internvl35_14b_lora",
]

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
        raise HTTPException(
            status_code=415, detail=f"unsupported document type: {suffix or 'none'}"
        )
    if kind == "media" and suffix not in MEDIA_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"unsupported media type: {suffix or 'none'}")
    if mime and "/" not in mime:
        raise HTTPException(status_code=400, detail="invalid mime type")
    return clean


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    validate_production_auth_settings()
    init_db()
    with SessionLocal() as session:
        pilot_project = ensure_demo_project(session)
        ensure_pilot_identity(session, pilot_project)
        # Backfill additive revision records once at startup. This keeps legacy
        # databases compatible and prevents concurrent first-page reads from
        # racing to create the same revision row.
        for project_id in session.scalars(select(Project.project_id)).all():
            ensure_project_revisions(session, project_id)
        session.commit()
    yield


app = FastAPI(title="Buili API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (WEB_PUBLIC_ROOT / "plans").exists():
    app.mount("/plans", StaticFiles(directory=WEB_PUBLIC_ROOT / "plans"), name="plans")
if (WEB_PUBLIC_ROOT / "site-media").exists():
    app.mount(
        "/site-media", StaticFiles(directory=WEB_PUBLIC_ROOT / "site-media"), name="site-media"
    )
if (WEB_PUBLIC_ROOT / "plan2field3d").exists():
    app.mount(
        "/plan2field3d",
        StaticFiles(directory=WEB_PUBLIC_ROOT / "plan2field3d"),
        name="plan2field3d",
    )

app.include_router(auth_router)
app.include_router(spatial_router)
app.include_router(workflow_router)
init_db()


@app.middleware("http")
async def accept_api_prefix(request: Request, call_next):
    if request.scope["path"].startswith("/api/"):
        request.scope["path"] = request.scope["path"][4:]
    return await call_next(request)


@app.middleware("http")
async def authenticate_and_scope_request(request: Request, call_next):
    principal = resolve_request_principal(request)
    if settings.auth_required and request_path_requires_auth(request.scope["path"]) and not principal:
        return JSONResponse(
            status_code=401,
            content={"detail": "authentication required"},
            headers={"Cache-Control": "no-store"},
        )
    if settings.auth_required and request.method.upper() in {
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
    }:
        origin = request.headers.get("origin", "").rstrip("/")
        allowed = {item.rstrip("/") for item in settings.cors_allow_origins}
        if origin and "*" not in allowed and origin not in allowed:
            return JSONResponse(
                status_code=403,
                content={"detail": "request origin is not allowed"},
                headers={"Cache-Control": "no-store"},
            )
    request.state.principal = principal
    token = bind_principal(principal)
    try:
        response = await call_next(request)
        if request_path_requires_auth(request.scope["path"]):
            response.headers.setdefault("Cache-Control", "private, no-store")
        return response
    finally:
        reset_principal(token)


@app.get("/", include_in_schema=False)
def web_root() -> FileResponse:
    return FileResponse(API_STATIC_ROOT / "index.html")


@app.get("/manifest.webmanifest", include_in_schema=False)
def web_manifest() -> FileResponse:
    return FileResponse(
        WEB_PUBLIC_ROOT / "manifest.webmanifest", media_type="application/manifest+json"
    )


@app.get("/sw.js", include_in_schema=False)
def service_worker() -> FileResponse:
    return FileResponse(WEB_PUBLIC_ROOT / "sw.js", media_type="application/javascript")


@app.get("/buili_favicon_transparent.png", include_in_schema=False)
def favicon_png() -> FileResponse:
    return FileResponse(WEB_PUBLIC_ROOT / "buili_favicon_transparent.png", media_type="image/png")


@app.get("/icon.svg", include_in_schema=False)
def icon_svg() -> FileResponse:
    return FileResponse(WEB_PUBLIC_ROOT / "icon.svg", media_type="image/svg+xml")


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {"status": "ok", "service": "buili-api", "gpu": gpu_policy()}


@app.get("/readyz")
def readyz(session: Session = Depends(get_session)) -> dict[str, object]:
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database is not ready") from exc
    if not settings.storage_root.exists() or not settings.storage_root.is_dir():
        raise HTTPException(status_code=503, detail="object storage is not ready")
    return {"status": "ready", "service": "buili-api", "database": "ok", "storage": "ok"}


def _read_artifact_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _vlm_training_status() -> dict[str, object]:
    artifacts_root = REPO_ROOT / "data" / "artifacts"
    artifact_name = VLM_ARTIFACT_CANDIDATES[0]
    artifact_dir = artifacts_root / artifact_name
    for candidate in VLM_ARTIFACT_CANDIDATES:
        candidate_dir = artifacts_root / candidate
        candidate_summary = _read_artifact_json(candidate_dir / "training_summary.json")
        if candidate_summary:
            artifact_name = candidate
            artifact_dir = candidate_dir
            summary = candidate_summary
            break

    summary = _read_artifact_json(artifact_dir / "training_summary.json")
    manifest = _read_artifact_json(artifact_dir / "adapter_manifest.json")
    generation_qa = _read_artifact_json(artifact_dir / "generation_qa.json")
    dataset_manifest = summary.get("dataset_manifest") if summary else None
    open_corpus_manifest = summary.get("open_corpus_manifest") if summary else None
    if not summary:
        return {
            "status": "not_trained",
            "model_family": "InternVL3.5",
            "base_model_id": "OpenGVLab/InternVL3_5-14B-HF",
            "teacher_model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "preferred_artifact_path": str(artifact_dir),
            "gpu_policy": gpu_policy(),
            "detail": (
                "Run HF_HOME=/SSD/guest/chojoonghui/hf_cache CUDA_VISIBLE_DEVICES=7 "
                "conda run -n cjh_buili python ml/train_internvl35_lora.py "
                "--dataset data/processed/buili_vlm_plus_open/sft_dataset.jsonl "
                "--out-dir data/artifacts/buili_internvl35_14b_plus_open_lora"
            ),
        }

    qa_rate = None
    qa_samples = 0
    if generation_qa:
        qa_rate = generation_qa.get("json_valid_rate")
        qa_samples = int(generation_qa.get("max_eval_samples") or 0)

    dataset_sha256 = None
    dataset_rows = None
    if isinstance(dataset_manifest, dict):
        dataset_sha256 = dataset_manifest.get("sha256")
        dataset_rows = dataset_manifest.get("rows")
    elif manifest:
        dataset_sha256 = manifest.get("dataset_sha256")

    open_corpus_version = None
    open_corpus_records = None
    if isinstance(open_corpus_manifest, dict):
        open_corpus_version = open_corpus_manifest.get("corpus_version")
        open_corpus_records = open_corpus_manifest.get("records")
    elif manifest:
        open_corpus_version = manifest.get("open_corpus_version")

    return {
        "status": summary.get("status", "trained"),
        "artifact_name": artifact_name,
        "model_family": summary.get("model_family"),
        "base_model_id": summary.get("base_model_id"),
        "teacher_model_id": summary.get("teacher_model_id"),
        "adapter_path": summary.get("adapter_path"),
        "adapter_files": manifest.get("adapter_files") if manifest else [],
        "quantization": summary.get("quantization"),
        "train_rows": summary.get("train_rows"),
        "eval_rows": summary.get("eval_rows"),
        "global_steps": summary.get("global_steps"),
        "eval_loss": summary.get("eval_loss"),
        "dataset": summary.get("dataset"),
        "dataset_sha256": dataset_sha256,
        "dataset_rows": dataset_rows,
        "open_corpus_version": open_corpus_version,
        "open_corpus_records": open_corpus_records,
        "data_governance": summary.get("data_governance"),
        "raw_generation_json_valid_rate": summary.get("json_valid_rate"),
        "production_prompt_json_valid_rate": qa_rate,
        "production_prompt_eval_samples": qa_samples,
        "gpu_policy": summary.get("gpu") or gpu_policy(),
        "scope_note": summary.get("scope_note"),
    }


@app.get("/v1/projects", response_model=list[ProjectOut])
def list_projects(session: Session = Depends(get_session)) -> list[Project]:
    return list(session.scalars(select(Project).order_by(Project.created_at.desc())).all())


@app.post("/v1/projects", response_model=ProjectOut)
def create_project(payload: ProjectCreate, session: Session = Depends(get_session)) -> Project:
    from .auth import current_principal

    principal = current_principal()
    demo = ensure_demo_project(session)
    project = Project(
        org_id=principal.org_id if principal else demo.org_id,
        name=payload.name,
        address=payload.address,
        project_type=payload.project_type,
    )
    session.add(project)
    session.flush()
    get_or_create_project_profile(session, project.project_id)
    record_audit(
        session,
        project=project,
        actor="system",
        action="PROJECT_CREATED",
        entity_type="project",
        entity_id=project.project_id,
        after=project_snapshot(project, get_or_create_project_profile(session, project.project_id)),
    )
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
    r2_key = f"org/{project.org_id}/project/{project.project_id}/raw/{upload_id}_{filename}"
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
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, str]:
    actor, role = actor_context(request)
    require_role(role, EDIT_ROLES, "complete uploads")
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
        session.flush()
        revision_state = DocumentRevision(
            document_id=doc.doc_id,
            project_id=doc.project_id,
            logical_key=f"{doc.type}:{Path(doc.filename).stem.lower()}",
            revision=doc.revision,
            state="unclassified",
            source_hash=doc.hash,
            upload_actor=actor,
        )
        session.add(revision_state)
        intent.status = "completed"
        project = session.get(Project, doc.project_id)
        record_audit(
            session,
            project=project,
            actor=actor,
            action="FILE_UPLOADED",
            entity_type="document",
            entity_id=doc.doc_id,
            after={
                "filename": doc.filename,
                "hash": doc.hash,
                "document_type": doc.type,
                "revision": doc.revision,
                "revision_state": "unclassified",
            },
        )
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
    session.flush()
    project = session.get(Project, media.project_id)
    record_audit(
        session,
        project=project,
        actor=actor,
        action="FILE_UPLOADED",
        entity_type="site_media",
        entity_id=media.media_id,
        after={"filename": media.filename, "hash": media.hash, "mime": media.mime},
    )
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
    ensure_project_revisions(session, project_id)
    session.commit()
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
def list_documents(project_id: str, session: Session = Depends(get_session)) -> list[dict[str, object]]:
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="project not found")
    documents = list(
        session.scalars(
            select(Document)
            .where(Document.project_id == project_id)
            .order_by(Document.created_at.desc())
        ).all()
    )
    revision_by_document = {
        revision.document_id: revision
        for _, revision in ensure_project_revisions(session, project_id)
    }
    session.commit()
    return [
        {
            "doc_id": item.doc_id,
            "project_id": item.project_id,
            "type": item.type,
            "filename": item.filename,
            "mime": item.mime,
            "r2_key": item.r2_key,
            "hash": item.hash,
            "revision": item.revision,
            "parsed_status": item.parsed_status,
            "size": item.size,
            "metadata_json": dict(item.metadata_json or {}),
            "is_current": revision_by_document[item.doc_id].state == "current",
            "revision_state": revision_by_document[item.doc_id].state,
            "issue_date": revision_by_document[item.doc_id].issue_date,
        }
        for item in documents
    ]


@app.get("/v1/projects/{project_id}/media", response_model=list[SiteMediaOut])
def list_media(project_id: str, session: Session = Depends(get_session)) -> list[dict[str, object]]:
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="project not found")
    items = list(
        session.scalars(
            select(SiteMedia)
            .where(SiteMedia.project_id == project_id)
            .order_by(SiteMedia.created_at.desc())
        ).all()
    )
    return [
        {
            "media_id": item.media_id,
            "project_id": item.project_id,
            "filename": item.filename,
            "mime": item.mime,
            "r2_key": item.r2_key,
            "hash": item.hash,
            "metadata_json": dict(item.metadata_json or {}),
            "download_url": f"{settings.public_base_url}/v1/media/{item.media_id}/download",
        }
        for item in items
    ]


@app.get("/v1/media/{media_id}/download")
def download_media(media_id: str, session: Session = Depends(get_session)) -> FileResponse:
    media = session.get(SiteMedia, media_id)
    if not media:
        raise HTTPException(status_code=404, detail="media not found")
    if media.r2_key.startswith("asset://site-media/"):
        relative_name = Path(media.r2_key.removeprefix("asset://site-media/")).name
        path = (WEB_PUBLIC_ROOT / "site-media" / relative_name).resolve()
        allowed_root = (WEB_PUBLIC_ROOT / "site-media").resolve()
    else:
        path = object_path(media.r2_key).resolve()
        allowed_root = settings.storage_root.resolve()
    try:
        path.relative_to(allowed_root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="media not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="media file is missing")
    response = FileResponse(
        path,
        media_type=media.mime or "application/octet-stream",
        filename=Path(media.filename).name or media.media_id,
        content_disposition_type="inline",
    )
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/v1/projects/{project_id}/observations", response_model=list[ObservationOut])
def list_observations(
    project_id: str, session: Session = Depends(get_session)
) -> list[Observation]:
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
def technology_status(
    project_id: str, session: Session = Depends(get_session)
) -> list[TechnologyStatusOut]:
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="project not found")
    docs = session.scalars(select(Document).where(Document.project_id == project_id)).all()
    chunks = session.scalars(
        select(SpecChunk).join(Document).where(Document.project_id == project_id)
    ).all()
    media = session.scalars(select(SiteMedia).where(SiteMedia.project_id == project_id)).all()
    media_ids = select(SiteMedia.media_id).where(SiteMedia.project_id == project_id)
    observations = session.scalars(
        select(Observation).where(Observation.media_id.in_(media_ids))
    ).all()
    issues = session.scalars(select(Issue).where(Issue.project_id == project_id)).all()
    jobs = session.scalars(select(Job).where(Job.project_id == project_id)).all()
    frames = session.scalars(select(Frame).where(Frame.media_id.in_(media_ids))).all()
    plan_graphs = session.scalars(select(PlanGraph).where(PlanGraph.project_id == project_id)).all()
    spatial_assets = session.scalars(
        select(SpatialAsset).where(SpatialAsset.project_id == project_id)
    ).all()
    spatial_evidence = session.scalars(
        select(SpatialEvidence)
        .join(Issue, Issue.issue_id == SpatialEvidence.issue_id)
        .where(Issue.project_id == project_id)
    ).all()
    vlm = _vlm_training_status()
    vlm_trained = vlm.get("status") == "trained"
    vlm_qa_rate = vlm.get("production_prompt_json_valid_rate")
    statuses = [
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
            summary=(
                f"{len(media)} media files, {len(frames)} derived frames, "
                f"{len(observations)} recognized observations."
            ),
        ),
        TechnologyStatusOut(
            key="mismatch_candidates",
            label="Drawing-field mismatch candidate generation",
            status="ready" if issues else "needs_review",
            evidence_count=len(issues),
            summary=(
                f"{len(issues)} review candidates generated with requirement, "
                "observation, and plan-location links."
            ),
        ),
        TechnologyStatusOut(
            key="reports",
            label="Punch list, RFI, and change order report generation",
            status="ready" if issues else "needs_issues",
            evidence_count=len([issue for issue in issues if issue.rfi_draft]),
            summary=(
                "Punch PDF/CSV, RFI draft, and CO evidence PDF generation are "
                "available from issue data."
            ),
        ),
        TechnologyStatusOut(
            key="web_review",
            label="Web-based field issue review and management",
            status="ready" if jobs else "needs_job",
            evidence_count=len(jobs),
            summary=(
                f"{len(jobs)} pipeline runs available for browser-based issue review, "
                "approval, RFI, and reporting."
            ),
        ),
        TechnologyStatusOut(
            key="plan2field_3d",
            label="Plan2Field 2D drawing to lightweight 3D spatial pipeline",
            status="ready" if plan_graphs and spatial_assets else "needs_plan_graph",
            evidence_count=len(spatial_evidence) or len(plan_graphs),
            summary=(
                f"{len(plan_graphs)} plan graphs, {len(spatial_assets)} spatial assets, "
                f"and {len(spatial_evidence)} issue-level spatial evidence links."
            ),
        ),
    ]
    statuses.append(
        TechnologyStatusOut(
            key="vlm_14b_adapter",
            label="14B VLM field-to-report domain adapter",
            status="ready_with_guardrail" if vlm_trained and vlm_qa_rate else "needs_training",
            evidence_count=int(vlm.get("global_steps") or 0),
            summary=(
                f"{vlm.get('base_model_id', 'InternVL3.5-14B')} LoRA adapter trained on "
                f"{vlm.get('train_rows', 0)} rows; production prompt JSON QA "
                f"{int(float(vlm_qa_rate or 0) * 100)}% over "
                f"{vlm.get('production_prompt_eval_samples', 0)} samples. Raw long-form "
                "generation still uses a guardrail/compositor."
            ),
        )
    )
    return statuses


@app.get("/v1/projects/{project_id}/issues", response_model=list[IssueOut])
def list_issues(project_id: str, session: Session = Depends(get_session)) -> list[dict[str, object]]:
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="project not found")
    issues = session.scalars(
        select(Issue)
        .options(selectinload(Issue.evidence))
        .where(Issue.project_id == project_id)
        .order_by(Issue.confidence.desc())
    ).all()
    payload = [issue_detail(session, issue) for issue in issues]
    session.commit()
    return payload


@app.patch("/v1/issues/{issue_id}", response_model=IssueOut)
def update_issue(
    issue_id: str,
    payload: IssuePatch,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, object]:
    actor, role = actor_context(request)
    require_role(role, EDIT_ROLES, "edit issues")
    issue = session.scalar(
        select(Issue).options(selectinload(Issue.evidence)).where(Issue.issue_id == issue_id)
    )
    if not issue:
        raise HTTPException(status_code=404, detail="issue not found")
    workflow = get_or_create_issue_workflow(session, issue)
    before = issue_snapshot(issue, workflow)
    values = payload.model_dump(exclude_unset=True)
    if values.get("status") in {"approved", "issued", "closed"}:
        raise HTTPException(
            status_code=409,
            detail="approved, issued, and closed states require the review/action workflow",
        )
    workflow_fields = {
        "priority": "priority",
        "expected_condition": "expected_condition",
        "difference": "difference",
        "recommended_route": "recommended_route",
        "evidence_gaps": "evidence_gaps_json",
        "source_status": "source_status",
        "impact": "impact_json",
    }
    for key, value in values.items():
        if value is not None:
            if key in workflow_fields:
                setattr(workflow, workflow_fields[key], value)
            else:
                setattr(issue, key, value)
    workflow.version += 1
    if workflow.review_status == "approved":
        workflow.review_status = "review_ready"
        workflow.reviewer = ""
        issue.status = "review_ready"
    project = session.get(Project, issue.project_id)
    record_audit(
        session,
        project=project,
        actor=actor,
        action="ISSUE_EDITED",
        entity_type="issue",
        entity_id=issue.issue_id,
        before=before,
        after=issue_snapshot(issue, workflow),
    )
    session.commit()
    session.refresh(issue)
    return issue_detail(session, issue)


@app.post("/v1/issues/{issue_id}/rfi", response_model=RfiOut)
def create_rfi(issue_id: str, session: Session = Depends(get_session)) -> RfiOut:
    issue = session.get(Issue, issue_id)
    if not issue:
        raise HTTPException(status_code=404, detail="issue not found")
    workflow = get_or_create_issue_workflow(session, issue)
    refresh_review_readiness(session, issue, workflow)
    blockers = issue_action_blockers(
        session,
        issue,
        workflow,
        report_type="rfi",
        require_approval=True,
    )
    session.commit()
    return RfiOut(
        issue_id=issue.issue_id,
        title=issue.title,
        markdown=build_markdown_rfi(
            issue,
            sources=list(workflow.source_snapshot_json or []),
            impact=dict(workflow.impact_json or {}),
        ),
        readiness={"can_issue": not blockers, "blockers": blockers},
    )


@app.post("/v1/projects/{project_id}/reports", response_model=ReportOut)
def create_report(
    project_id: str,
    payload: ReportRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> ReportOut:
    actor, role = actor_context(request)
    require_role(role, EDIT_ROLES, "create report drafts")
    selected_issue_ids = list(dict.fromkeys(payload.issue_ids or [])) if payload.issue_ids else None
    try:
        report_id, path = build_report(
            session,
            project_id,
            payload.report_type,
            payload.format,
            issue_ids=selected_issue_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    reports_root = (settings.storage_root / "reports").resolve()
    try:
        rel = path.resolve().relative_to(reports_root)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="report path escaped storage root") from exc
    issue_statement = (
        select(Issue)
        .options(selectinload(Issue.evidence), selectinload(Issue.spatial_evidence))
        .where(Issue.project_id == project_id)
    )
    if selected_issue_ids is not None:
        issue_statement = issue_statement.where(Issue.issue_id.in_(selected_issue_ids))
    issues = list(session.scalars(issue_statement.order_by(Issue.confidence.desc())).all())
    if selected_issue_ids is not None:
        issue_by_id = {issue.issue_id: issue for issue in issues}
        issues = [issue_by_id[issue_id] for issue_id in selected_issue_ids]
    snapshots = []
    readiness: list[dict[str, object]] = []
    source_snapshots: dict[tuple[str, str], dict[str, object]] = {}
    for issue in issues:
        workflow = get_or_create_issue_workflow(session, issue)
        refresh_review_readiness(session, issue, workflow)
        current_sources = source_snapshot_for_issue(session, issue)
        workflow.source_snapshot_json = current_sources
        snapshots.append(issue_snapshot(issue, workflow))
        blockers = issue_action_blockers(
            session,
            issue,
            workflow,
            report_type=payload.report_type,
            require_approval=True,
        )
        readiness.append(
            {
                "issue_id": issue.issue_id,
                "title": issue.title,
                "can_issue": not blockers,
                "blockers": blockers,
            }
        )
        for source in current_sources:
            source_snapshots[(str(source.get("document_id")), str(source.get("revision")))] = source
    report = ReportRecord(
        report_id=report_id,
        project_id=project_id,
        report_type=payload.report_type,
        title=f"{payload.report_type.replace('_', ' ').title()} draft",
        status="draft",
        created_by=actor,
    )
    version = ReportVersion(
        report_id=report_id,
        version=1,
        format=payload.format,
        path=str(path.resolve()),
        checksum=sha256_path(path),
        source_snapshot_json=list(source_snapshots.values()),
        issue_snapshot_json=snapshots,
        status="draft",
        created_by=actor,
    )
    scope = ReportScope(
        report_id=report_id,
        issue_ids_json=[issue.issue_id for issue in issues],
        explicit_selection=1 if selected_issue_ids is not None else 0,
    )
    session.add_all([report, version, scope])
    project = session.get(Project, project_id)
    record_audit(
        session,
        project=project,
        actor=actor,
        action="REPORT_DRAFT_CREATED",
        entity_type="report",
        entity_id=report_id,
        after={
            "version": 1,
            "format": payload.format,
            "checksum": version.checksum,
            "source_snapshot": version.source_snapshot_json,
        },
    )
    session.commit()
    return ReportOut(
        report_id=report_id,
        report_type=payload.report_type,
        format=payload.format,
        path=str(path),
        download_url=f"{settings.public_base_url}/v1/reports/{rel.as_posix()}",
        issue_ids=[issue.issue_id for issue in issues],
        readiness=readiness,
        can_issue=bool(readiness) and all(bool(item["can_issue"]) for item in readiness),
    )


@app.get("/v1/reports/{path:path}")
def download_report(path: str, session: Session = Depends(get_session)) -> FileResponse:
    reports_root = (settings.storage_root / "reports").resolve()
    report_path = (reports_root / path).resolve()
    try:
        report_path.relative_to(reports_root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="report not found") from exc
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="report not found")
    # ReportRecord is tenant-scoped by the authenticated-session query policy.
    # Match the artifact through an accessible report rather than trusting an
    # otherwise guessable storage path.
    accessible_report_ids = select(ReportRecord.report_id)
    version = session.scalar(
        select(ReportVersion).where(
            ReportVersion.report_id.in_(accessible_report_ids),
            ReportVersion.path == str(report_path),
        )
    )
    if not version:
        raise HTTPException(status_code=404, detail="report not found")
    return FileResponse(report_path)


@app.get("/v1/projects/{project_id}/plan-overlay", response_model=OverlayOut)
def get_overlay(project_id: str, session: Session = Depends(get_session)) -> dict:
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="project not found")
    return overlay_for_project(project_id, session)


@app.get("/v1/projects/{project_id}/rag/search")
def rag_search(
    project_id: str, q: str, session: Session = Depends(get_session)
) -> dict[str, object]:
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="project not found")
    revision_rows = ensure_project_revisions(session, project_id)
    current_doc_ids = {
        document.doc_id for document, revision in revision_rows if revision.state == "current"
    }
    chunks = session.scalars(
        select(SpecChunk)
        .join(Document)
        .where(Document.project_id == project_id, SpecChunk.doc_id.in_(current_doc_ids))
    ).all()
    result = rag_answer(q, list(chunks), top_k=8)
    result["filters"] = {"project_id": project_id, "revision_scope": "current"}
    session.commit()
    return result


@app.get("/v1/metrics")
def metrics(session: Session = Depends(get_session)) -> dict[str, int]:
    return {
        "projects": len(session.scalars(select(Project)).all()),
        "jobs": len(session.scalars(select(Job)).all()),
        "issues": len(session.scalars(select(Issue)).all()),
    }


@app.get("/v1/training/status")
def training_status() -> dict[str, object]:
    path = REPO_ROOT / "data" / "artifacts" / "buili_ai_stack" / "training_progress.json"
    if not path.exists():
        return {
            "overall_training_progress_percent": 0,
            "status": "not_trained",
            "gpu_policy": gpu_policy(),
            "vlm_domain_adapter": _vlm_training_status(),
            "detail": (
                "Run CUDA_VISIBLE_DEVICES=7 conda run -n cjh_buili "
                "python ml/train_full_ai_stack.py"
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["gpu_policy"] = gpu_policy()
    payload["vlm_domain_adapter"] = _vlm_training_status()
    return payload


if __name__ == "__main__":
    uvicorn.run(
        "services.api.buili.main:app", host=settings.api_host, port=settings.api_port, reload=False
    )
