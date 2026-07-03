from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import fitz
import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .models import (
    Document,
    FieldPoseFrame,
    Frame,
    Issue,
    IssueEvidence,
    Job,
    ModelRun,
    Observation,
    PlanGraph,
    PlanEntity,
    Project,
    Sheet,
    SiteMedia,
    SpecChunk,
    SpatialAlignment,
    SpatialAsset,
    SpatialEvidence,
    new_id,
)
from .storage import object_path
from .spatial.alignment import create_spatial_alignment
from .spatial.compare import compare_project_spatial
from .spatial.field_capture import create_field_asset_from_frames, ingest_field_pose_frame
from .spatial.geometry import build_design_glb
from .spatial.plan_parser import create_plan_graph_record


JOB_STATES = [
    "queued",
    "ingesting",
    "indexing",
    "extracting_frames",
    "detecting",
    "spatializing_plan",
    "reconstructing_field",
    "aligning_plan_field",
    "reasoning",
    "review_ready",
    "failed",
]

PLAN_ENTITY_TYPES = [
    "duplex_outlet",
    "data_port",
    "switch",
    "ceiling_light",
    "diffuser",
    "sink",
    "toilet",
    "door",
    "window",
    "cabinet",
]

SITE_OBJECT_TYPES = [
    "installed_outlet",
    "visible_switch",
    "missing_cover_plate",
    "light_fixture",
    "duct_diffuser",
    "sprinkler_head",
    "smoke_detector",
]

DISCIPLINE_BY_PREFIX = {
    "A": "architectural",
    "E": "electrical",
    "M": "mechanical",
    "P": "plumbing",
    "FS": "fire_safety",
}


@dataclass(frozen=True)
class ParsedPage:
    page_no: int
    text: str
    sheet_number: str
    discipline: str
    title: str


def _event(job: Job, state: str, progress: int, name: str, details: dict[str, Any] | None = None) -> None:
    events = list(job.events or [])
    events.append(
        {
            "event": name,
            "state": state,
            "progress": progress,
            "details": details or {},
            "ts": round(time.time(), 3),
        }
    )
    job.state = state
    job.progress = progress
    job.events = events


def _hash_json(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _embed_text(text: str, dims: int = 48) -> list[float]:
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    vec = np.zeros(dims, dtype=np.float32)
    for token in tokens:
        idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % dims
        vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm:
        vec /= norm
    return [round(float(v), 6) for v in vec]


def _sheet_number(text: str, page_no: int) -> str:
    dotted = re.search(r"\b([AEMP])\s?(\d)\.(\d)\b", text.upper())
    if dotted:
        return f"{dotted.group(1)}{dotted.group(2)}.{dotted.group(3)}"
    match = re.search(r"\b(FS|[AEMP])[-\s]?(\d{1,3})\b", text.upper())
    if match:
        return f"{match.group(1)}-{int(match.group(2)):03d}"
    return f"A-{page_no:03d}"


def _discipline(sheet_number: str, text: str) -> str:
    token = sheet_number.split("-")[0]
    prefix = "FS" if token.startswith("FS") else token[0]
    if prefix in DISCIPLINE_BY_PREFIX:
        return DISCIPLINE_BY_PREFIX[prefix]
    lowered = text.lower()
    if "electrical" in lowered or "outlet" in lowered:
        return "electrical"
    if "mechanical" in lowered or "diffuser" in lowered or "duct" in lowered:
        return "mechanical"
    if "plumbing" in lowered or "sink" in lowered or "toilet" in lowered:
        return "plumbing"
    return "architectural"


def _page_title(text: str, sheet_number: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:8]:
        if sheet_number.replace("-", "") in line.replace("-", "").replace(" ", ""):
            continue
        if 4 <= len(line) <= 90:
            return line[:90]
    return "Floor plan"


def _read_text_file(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _read_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    texts = [node.text or "" for node in root.iter() if node.tag.endswith("}t")]
    return "\n".join(part for part in texts if part.strip())


def _read_xlsx_text(path: Path) -> str:
    texts: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name == "xl/sharedStrings.xml" or name.startswith("xl/worksheets/sheet"):
                root = ElementTree.fromstring(archive.read(name))
                texts.extend(node.text or "" for node in root.iter() if node.text)
    return "\n".join(part.strip() for part in texts if part.strip())


def parse_document(doc: Document) -> list[ParsedPage]:
    path = object_path(doc.r2_key)
    if not path.exists():
        if doc.filename.startswith(("demo-electrical-plan", "cooper-residence")):
            path.write_text(_demo_plan_text(), encoding="utf-8")
        else:
            raise FileNotFoundError(f"uploaded document not found: {doc.filename}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        text = _read_text_file(path)
        sheet_number = _sheet_number(text, 1)
        return [
            ParsedPage(
                page_no=1,
                text=text or f"{doc.filename} has no machine-readable text.",
                sheet_number=sheet_number,
                discipline=_discipline(sheet_number, text),
                title=_page_title(text, sheet_number),
            )
        ]
    if suffix == ".docx":
        text = _read_docx_text(path)
        sheet_number = _sheet_number(text, 1)
        return [
            ParsedPage(
                page_no=1,
                text=text or f"{doc.filename} has no machine-readable text.",
                sheet_number=sheet_number,
                discipline=_discipline(sheet_number, text),
                title=_page_title(text, sheet_number),
            )
        ]
    if suffix == ".xlsx":
        text = _read_xlsx_text(path)
        sheet_number = _sheet_number(text, 1)
        return [
            ParsedPage(
                page_no=1,
                text=text or f"{doc.filename} has no machine-readable text.",
                sheet_number=sheet_number,
                discipline=_discipline(sheet_number, text),
                title=_page_title(text, sheet_number),
            )
        ]
    if suffix != ".pdf":
        raise ValueError(f"unsupported parse format: {suffix}")

    parsed: list[ParsedPage] = []
    pdf = fitz.open(path)
    for index, page in enumerate(pdf, start=1):
        text = page.get_text("text") or ""
        if not text.strip():
            text = f"Page {index} plan sheet. Verify room labels, outlets, lights, diffusers."
        sheet_number = _sheet_number(text, index)
        parsed.append(
            ParsedPage(
                page_no=index,
                text=text,
                sheet_number=sheet_number,
                discipline=_discipline(sheet_number, text),
                title=_page_title(text, sheet_number),
            )
        )
    return parsed


def _demo_plan_text() -> str:
    return (
        "E1.1 Electrical Plans\n"
        "Cooper Residence electrical legend identifies outlets, switches, AFCI, GFCI/WP, "
        "smoke detectors, carbon monoxide detectors, ceiling fans, pendant lights, and 220V outlets.\n"
        "Electrical notes require smoke detectors in sleeping rooms, AFCI protection for outlets, "
        "GFCI weatherproof outlets at exterior locations, and 200 amp service coordination.\n"
        "Verify device coverage and switching before rough-in signoff."
    )


def _ensure_demo_assets(project: Project, session: Session) -> None:
    demo_text = _demo_plan_text()
    docs = session.scalars(select(Document).where(Document.project_id == project.project_id)).all()
    if docs:
        for doc in docs:
            if doc.filename.startswith(("demo-electrical-plan", "cooper-residence")):
                path = object_path(doc.r2_key)
                if not path.exists():
                    path.write_text(demo_text, encoding="utf-8")
                if doc.filename == "demo-electrical-plan.pdf":
                    doc.filename = "cooper-residence-e11-electrical-plan.txt"
                    doc.mime = "text/plain"
                    doc.revision = "E1.1"
    else:
        doc = Document(
            project_id=project.project_id,
            type="plan",
            filename="cooper-residence-e11-electrical-plan.txt",
            mime="text/plain",
            r2_key=f"org/{project.org_id}/project/{project.project_id}/raw/demo-electrical-plan.txt",
            hash=_hash_json({"demo": True, "text": demo_text}),
            revision="E1.1",
            parsed_status="uploaded",
            size=len(demo_text.encode("utf-8")),
        )
        object_path(doc.r2_key).write_text(demo_text, encoding="utf-8")
        session.add(doc)

    has_media = session.scalars(
        select(SiteMedia).where(SiteMedia.project_id == project.project_id).limit(1)
    ).first()
    if not has_media:
        session.add(
            SiteMedia(
                project_id=project.project_id,
                filename="construction-site-electrical-work.jpg",
                mime="image/jpeg",
                r2_key="asset://site-media/construction-site-electrical-work.jpg",
                hash=_hash_json({"asset": "construction-site-electrical-work.jpg"}),
                metadata_json={"source": "bundled_public_sample", "role": "field_photo"},
            )
        )
    session.commit()


def _chunk_text(text: str, page_no: int) -> list[tuple[str, dict[str, Any]]]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    chunks: list[tuple[str, dict[str, Any]]] = []
    current: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        current.append(sentence)
        if sum(len(part.split()) for part in current) >= 90:
            joined = " ".join(current)
            chunks.append((joined, {"page": page_no, "token_estimate": len(joined.split())}))
            current = []
    if current:
        joined = " ".join(current)
        chunks.append((joined, {"page": page_no, "token_estimate": len(joined.split())}))
    return chunks


def _entities_from_text(text: str, sheet_id: str) -> list[PlanEntity]:
    lowered = text.lower()
    room = "Conference Room 203"
    room_match = re.search(r"((?:conference|open office|restroom|room)\s+\d{2,4})", text, re.I)
    if room_match:
        room = room_match.group(1).title()

    candidates = {
        "duplex_outlet": ["duplex outlet", "outlet", "receptacle"],
        "data_port": ["data port", "data"],
        "switch": ["switch"],
        "ceiling_light": ["ceiling light", "light"],
        "diffuser": ["diffuser"],
        "sink": ["sink"],
        "toilet": ["toilet"],
        "door": ["door"],
        "window": ["window"],
        "cabinet": ["cabinet"],
    }
    entities: list[PlanEntity] = []
    seed = int(hashlib.sha256((sheet_id + text[:40]).encode()).hexdigest(), 16)
    rng = random.Random(seed)
    for entity_type, words in candidates.items():
        hits = sum(lowered.count(word) for word in words)
        if hits == 0 and entity_type in {"door", "window"}:
            hits = 1
        for idx in range(min(max(hits, 0), 6)):
            x = 0.18 + rng.random() * 0.64
            y = 0.18 + rng.random() * 0.64
            width = 0.025 + rng.random() * 0.035
            height = 0.025 + rng.random() * 0.035
            entities.append(
                PlanEntity(
                    sheet_id=sheet_id,
                    type=entity_type,
                    room_id=re.sub(r"\W+", "_", room.lower()),
                    room=room,
                    bbox=[
                        round(x, 3),
                        round(y, 3),
                        round(min(x + width, 0.98), 3),
                        round(min(y + height, 0.98), 3),
                    ],
                    confidence=round(0.66 + rng.random() * 0.22, 2),
                    source="document_text:requirements + geometric_locator:v0.2",
                )
            )
    return entities


def _observations_for_project(project_id: str, media: list[SiteMedia]) -> list[Observation]:
    if not media:
        return []
    seed = int(hashlib.sha256(project_id.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    base = [
        ("installed_outlet", "AFCI receptacle coverage needs confirmation.", "Main Floor"),
        ("installed_outlet", "Exterior outlet GFCI/WP condition needs close-up verification.", "Exterior / Garage"),
        ("smoke_detector", "Smoke detector location needs confirmation against the plan symbol.", "Sleeping Rooms"),
        ("visible_switch", "Switching path for fan/light kit needs field confirmation.", "Main Floor"),
        ("light_fixture", "Exterior Christmas light circuit evidence is incomplete.", "Exterior"),
    ]
    observations: list[Observation] = []
    for index, (object_type, text, room) in enumerate(base):
        source_media = media[index % len(media)]
        x = 0.2 + rng.random() * 0.55
        y = 0.2 + rng.random() * 0.55
        observations.append(
            Observation(
                media_id=source_media.media_id,
                frame_id=f"derived_frame_{rng.randint(1, 5)}",
                object_type=object_type,
                bbox=[round(x, 3), round(y, 3), round(x + 0.08, 3), round(y + 0.07, 3)],
                text=f"{text} Source media: {source_media.filename}. Room hint: {room}.",
                confidence=round(0.62 + rng.random() * 0.25, 2),
            )
        )
    return observations


def _build_issue(
    project_id: str,
    issue_type: str,
    discipline: str,
    severity: str,
    room: str,
    title: str,
    requirement: dict[str, Any],
    observation: dict[str, Any],
    plan_location: dict[str, Any],
    confidence: float,
) -> Issue:
    action = {
        "coverage_check": "Verify device coverage against E1.1 and capture correction evidence before rough-in approval.",
        "missing_item": "Verify installation count before closeout and request subcontractor correction photo.",
        "count_mismatch": "Confirm field count against plan count and update punch list if mismatch remains.",
        "location_mismatch": "Confirm installed location with superintendent and issue RFI if relocation was intentional.",
        "spec_mismatch": "Review cited specification section and request product/submittal clarification.",
        "unverified": "Collect closer photo or walkthrough frame before approving the area.",
        "potential_change_order": "Preserve evidence chain and prepare change order impact narrative.",
    }[issue_type]
    rfi = (
        f"Please confirm whether the requirement in {requirement.get('source', 'the contract documents')} "
        f"applies to {room}. Field evidence says: {observation.get('text', 'not enough evidence')}."
    )
    return Issue(
        project_id=project_id,
        type=issue_type,
        discipline=discipline,
        severity=severity,
        room=room,
        status="review_ready",
        confidence=confidence,
        title=title,
        description=(
            f"{title}. The issue is generated as a candidate and needs PM verification. "
            "No final defect decision is made automatically."
        ),
        recommended_action=action,
        assignee="Field PM",
        due_date="",
        subcontractor="",
        requirement=requirement,
        observation=observation,
        plan_location=plan_location,
        rfi_draft=rfi,
    )


def _issue_candidates(project: Project, session: Session) -> list[Issue]:
    sheets = session.scalars(
        select(Sheet).join(Document).where(Document.project_id == project.project_id)
    ).all()
    entities = session.scalars(
        select(PlanEntity).join(Sheet).join(Document).where(Document.project_id == project.project_id)
    ).all()
    chunks = session.scalars(
        select(SpecChunk).join(Document).where(Document.project_id == project.project_id)
    ).all()

    first_sheet = next((sheet for sheet in sheets if sheet.discipline == "electrical"), sheets[0] if sheets else None)
    first_entity = entities[0] if entities else None
    first_chunk = next(
        (
            chunk
            for chunk in chunks
            if "electrical" in str(chunk.metadata_json).lower() or "afci" in chunk.text.lower()
        ),
        chunks[0] if chunks else None,
    )
    media = session.scalars(select(SiteMedia).where(SiteMedia.project_id == project.project_id)).all()
    field_media_id = media[0].media_id if media else "field_verification_pending"
    field_frame_label = media[0].filename if media else "No uploaded field media linked yet."

    sheet_no = first_sheet.sheet_number if first_sheet else "sheet_pending"
    bbox = first_entity.bbox if first_entity else [0.41, 0.59, 0.47, 0.64]
    entity_id = first_entity.entity_id if first_entity else "plan_entity_pending"
    chunk_id = first_chunk.chunk_id if first_chunk else "spec_chunk_pending"
    chunk_text = (
        first_chunk.text
        if first_chunk
        else "Electrical notes require AFCI/GFCI outlet coverage and smoke detector coordination."
    )
    evidence_prefix = "crop" if media else "verification"

    issues = [
        _build_issue(
            project.project_id,
            "coverage_check",
            "electrical",
            "major",
            "Main Floor",
            "AFCI outlet coverage below E1.1 requirement",
            {
                "source": sheet_no,
                "text": "Electrical notes require AFCI protection for outlets in living and sleeping areas.",
            },
            {
                "media_id": field_media_id,
                "frame_ts": 83.2 if media else 0.0,
                "text": "AFCI outlet coverage has not been verified in the marked room."
                if media
                else "Field verification is required.",
            },
            {"sheet_id": sheet_no, "x": 0.59, "y": 0.64, "bbox": bbox},
            0.78,
        ),
        _build_issue(
            project.project_id,
            "unverified",
            "electrical",
            "major",
            "Exterior / Garage",
            "GFCI/WP exterior outlet verification needed",
            {"source": sheet_no, "text": "Electrical legend identifies GFCI/WP weatherproof outlet locations."},
            {
                "media_id": field_media_id,
                "frame_ts": 0.0,
                "text": "Exterior outlet weatherproof/GFCI condition needs close-up verification."
                if media
                else "No matching field frame has been uploaded.",
            },
            {"sheet_id": sheet_no, "x": 0.87, "y": 0.84, "bbox": [0.82, 0.78, 0.92, 0.9]},
            0.72,
        ),
        _build_issue(
            project.project_id,
            "location_mismatch",
            "electrical",
            "minor",
            "Sleeping Rooms",
            "Smoke detector placement needs verification",
            {"source": sheet_no, "text": "Smoke detectors are required in sleeping rooms and adjacent areas."},
            {
                "media_id": field_media_id,
                "frame_ts": 126.4 if media else 0.0,
                "text": "Detector location appears offset from the E1.1 symbol location."
                if media
                else "Field location evidence is pending.",
            },
            {"sheet_id": sheet_no, "x": 0.41, "y": 0.64, "bbox": [0.36, 0.58, 0.46, 0.69]},
            0.66,
        ),
        _build_issue(
            project.project_id,
            "spec_mismatch",
            "electrical",
            "major",
            "Service / Panel",
            "200 amp panel service coordination check",
            {"source": chunk_id, "text": chunk_text[:220]},
            {
                "media_id": field_media_id,
                "frame_ts": 0.0,
                "text": "Panel/service location needs field confirmation against E1.1."
                if media
                else "Submittal or close-up field evidence is pending.",
            },
            {"sheet_id": sheet_no, "x": 0.86, "y": 0.54, "bbox": [0.81, 0.49, 0.91, 0.58]},
            0.63,
        ),
        _build_issue(
            project.project_id,
            "unverified",
            "electrical",
            "informational",
            "Main Floor",
            "Ceiling fan and light switching needs field confirmation",
            {"source": sheet_no, "text": "Electrical legend shows ceiling fans, fan/light kits, and switch symbols."},
            {
                "media_id": field_media_id,
                "frame_ts": 52.0 if media else 0.0,
                "text": "Fan/light switching cannot be verified from the current field evidence."
                if media
                else "Switching cannot be verified without field media.",
            },
            {"sheet_id": sheet_no, "x": 0.55, "y": 0.72, "bbox": [0.5, 0.67, 0.6, 0.77]},
            0.52,
        ),
        _build_issue(
            project.project_id,
            "potential_change_order",
            "electrical",
            "major",
            "Exterior",
            "Exterior Christmas light circuit requires evidence",
            {"source": sheet_no, "text": "E1.1 calls out exterior Christmas lights and exterior outlet circuitry."},
            {
                "media_id": field_media_id,
                "frame_ts": 11.3 if media else 0.0,
                "text": "Exterior light/outlet scope may need change-order backup."
                if media
                else "Change-order backup media has not been uploaded.",
            },
            {"sheet_id": sheet_no, "x": 0.5, "y": 0.91, "bbox": [0.46, 0.86, 0.54, 0.94]},
            0.69,
        ),
    ]

    for issue in issues:
        issue.evidence = [
            IssueEvidence(
                evidence_type="sheet",
                ref_id=entity_id,
                r2_key=f"overlay://{sheet_no}",
                page=1,
                bbox=issue.plan_location.get("bbox", []),
                frame_ts=0.0,
                label="Plan entity",
            ),
            IssueEvidence(
                evidence_type="spec_chunk",
                ref_id=chunk_id,
                r2_key=f"chunk://{chunk_id}",
                page=1,
                bbox=[],
                frame_ts=0.0,
                label="Specification citation",
            ),
            IssueEvidence(
                evidence_type="frame",
                ref_id=issue.observation.get("media_id", ""),
                r2_key=f"{evidence_prefix}://{issue.observation.get('media_id', 'media')}/{field_frame_label}",
                page=0,
                bbox=[0.35, 0.35, 0.62, 0.62],
                frame_ts=float(issue.observation.get("frame_ts", 0.0)),
                label="Field crop/frame",
            ),
        ]
    return issues


def ensure_demo_project(session: Session) -> Project:
    project = session.scalars(select(Project).limit(1)).first()
    if project:
        _ensure_demo_assets(project, session)
        if not session.scalars(select(Issue).where(Issue.project_id == project.project_id).limit(1)).first():
            job = create_job_for_project(project, session)
            run_analysis_job(job.job_id, session)
        return project

    from .models import Organization

    org = Organization(name="Buili Pilot")
    session.add(org)
    session.flush()
    project = Project(
        org_id=org.org_id,
        name="Tenant Improvement Pilot",
        address="203 Market St, San Francisco, CA",
        project_type="tenant_improvement",
    )
    session.add(project)
    session.flush()
    doc = Document(
        project_id=project.project_id,
        type="plan",
        filename="cooper-residence-e11-electrical-plan.txt",
        mime="text/plain",
        r2_key=f"org/{org.org_id}/project/{project.project_id}/raw/demo-electrical-plan.txt",
        hash=_hash_json({"demo": True}),
        revision="E1.1",
        parsed_status="uploaded",
        size=len(_demo_plan_text()),
    )
    path = object_path(doc.r2_key)
    path.write_text(_demo_plan_text(), encoding="utf-8")
    session.add(doc)
    session.commit()
    _ensure_demo_assets(project, session)
    job = create_job_for_project(project, session)
    run_analysis_job(job.job_id, session)
    return project


def run_analysis_job(job_id: str, session: Session) -> None:
    from .database import init_db

    init_db()
    job = session.get(Job, job_id)
    if not job:
        return
    project = session.get(Project, job.project_id)
    if not project:
        job.state = "failed"
        job.error = "project not found"
        session.commit()
        return

    try:
        _event(job, "ingesting", 12, "job_started", {"project_id": project.project_id})
        session.commit()

        session.execute(
            delete(IssueEvidence).where(
                IssueEvidence.issue_id.in_(
                    select(Issue.issue_id).where(Issue.project_id == project.project_id)
                )
            )
        )
        session.execute(
            delete(SpatialEvidence).where(
                SpatialEvidence.issue_id.in_(
                    select(Issue.issue_id).where(Issue.project_id == project.project_id)
                )
            )
        )
        session.execute(delete(Issue).where(Issue.project_id == project.project_id))
        session.execute(delete(SpatialAlignment).where(SpatialAlignment.project_id == project.project_id))
        session.execute(delete(SpatialAsset).where(SpatialAsset.project_id == project.project_id))
        session.execute(delete(PlanGraph).where(PlanGraph.project_id == project.project_id))
        doc_ids = select(Document.doc_id).where(Document.project_id == project.project_id)
        sheet_ids = select(Sheet.sheet_id).where(Sheet.doc_id.in_(doc_ids))
        media_ids = select(SiteMedia.media_id).where(SiteMedia.project_id == project.project_id)
        session.execute(delete(Observation).where(Observation.media_id.in_(media_ids)))
        session.execute(delete(Frame).where(Frame.media_id.in_(media_ids)))
        session.execute(delete(FieldPoseFrame).where(FieldPoseFrame.media_id.in_(media_ids)))
        session.execute(delete(PlanEntity).where(PlanEntity.sheet_id.in_(sheet_ids)))
        session.execute(delete(SpecChunk).where(SpecChunk.doc_id.in_(doc_ids)))
        session.execute(delete(Sheet).where(Sheet.doc_id.in_(doc_ids)))
        session.commit()

        documents = session.scalars(select(Document).where(Document.project_id == project.project_id)).all()
        if not documents:
            raise ValueError("upload at least one plan, specification, or submittal before running review")

        for doc in documents:
            doc.parsed_status = "parsing"
            pages = parse_document(doc)
            for parsed in pages:
                sheet = Sheet(
                    doc_id=doc.doc_id,
                    sheet_number=parsed.sheet_number,
                    discipline=parsed.discipline,
                    page_no=parsed.page_no,
                    image_key=f"raster://{doc.doc_id}/page-{parsed.page_no}.png",
                    title=parsed.title,
                )
                session.add(sheet)
                session.flush()
                for entity in _entities_from_text(parsed.text, sheet.sheet_id):
                    session.add(entity)
                for text, metadata in _chunk_text(parsed.text, parsed.page_no):
                    session.add(
                        SpecChunk(
                            doc_id=doc.doc_id,
                            text=text,
                            metadata_json={
                                **metadata,
                                "project_id": project.project_id,
                                "document_id": doc.doc_id,
                                "discipline": parsed.discipline,
                                "sheet_id": parsed.sheet_number,
                                "revision": doc.revision,
                            },
                            embedding=_embed_text(text),
                            page=parsed.page_no,
                            bbox=[0.08, 0.1, 0.9, 0.86],
                        )
                    )
            doc.parsed_status = "parsed"

        _event(job, "indexing", 34, "chunk_indexed", {"documents": len(documents)})
        session.commit()

        media = session.scalars(select(SiteMedia).where(SiteMedia.project_id == project.project_id)).all()
        frame_count = 0
        for item in media:
            for idx in range(1, 5):
                frame_count += 1
                session.add(
                    Frame(
                        media_id=item.media_id,
                        timestamp=float(idx * 31.6),
                        r2_key=f"frame://{item.media_id}/{idx}.jpg",
                        blur_score=round(0.16 + idx * 0.08, 2),
                        room_hint=[
                            "Conference Room 203",
                            "Open Office 210",
                            "Room 204",
                            "Restroom 102",
                        ][idx - 1],
                    )
                )
        for observation in _observations_for_project(project.project_id, list(media)):
            session.add(observation)
        _event(job, "extracting_frames", 52, "frame_selected", {"media": len(media), "frames": frame_count})
        session.commit()

        _event(job, "detecting", 68, "observations_created", {"observations": 5 if media else 0})
        session.commit()

        _event(job, "spatializing_plan", 73, "plangraph_started", {"mode": "pymupdf_vector_text"})
        session.commit()
        plan_graph = create_plan_graph_record(session, project, replace_existing=True)
        _event(
            job,
            "spatializing_plan",
            78,
            "plangraph_created",
            {
                "plan_graph_id": plan_graph.id,
                "rooms": len((plan_graph.graph_json or {}).get("rooms", [])),
                "fixtures": len((plan_graph.graph_json or {}).get("fixtures", [])),
            },
        )
        session.commit()

        _event(job, "reconstructing_field", 82, "design_3d_started", {"plan_graph_id": plan_graph.id})
        session.commit()
        design_asset_id = new_id("spa")
        design_uri, design_metadata = build_design_glb(
            plan_graph.graph_json or {}, project.project_id, design_asset_id
        )
        design_asset = SpatialAsset(
            id=design_asset_id,
            project_id=project.project_id,
            type="design_glb",
            uri=design_uri,
            metadata_json={**design_metadata, "plan_graph_id": plan_graph.id},
        )
        session.add(design_asset)
        field_asset = None
        for item in media:
            for idx in range(1, 3):
                ingest_field_pose_frame(
                    session,
                    project.project_id,
                    media_id=item.media_id,
                    timestamp=float(idx * 31.6),
                    rgb_uri=f"frame://{item.media_id}/{idx}.jpg",
                    depth_uri="",
                    intrinsics_json={},
                    pose_json={},
                    blur_score=round(0.12 + idx * 0.07, 2),
                    room_hint=["Main Floor", "Exterior / Garage"][idx - 1],
                )
        field_asset = create_field_asset_from_frames(session, project.project_id)
        _event(
            job,
            "reconstructing_field",
            86,
            "spatial_assets_created",
            {
                "design_asset_id": design_asset.id,
                "field_asset_id": field_asset.id if field_asset else "",
                "field_mode": (field_asset.metadata_json or {}).get("mode") if field_asset else "no_media",
            },
        )
        session.commit()

        _event(job, "aligning_plan_field", 89, "alignment_started", {"guided_anchors": "auto_seed"})
        session.commit()
        alignment = create_spatial_alignment(session, project.project_id, plan_graph_id=plan_graph.id)
        _event(
            job,
            "aligning_plan_field",
            91,
            "alignment_created",
            {
                "alignment_id": alignment.id,
                "confidence": alignment.confidence,
                "requires_user_anchor": alignment.transform_json.get("requires_user_anchor", False),
            },
        )
        session.commit()

        issues = _issue_candidates(project, session)
        for issue in issues:
            session.add(issue)
        _event(job, "reasoning", 94, "issue_generated", {"issues": len(issues)})
        session.flush()
        _, _, spatial_evidence = compare_project_spatial(
            session,
            project.project_id,
            plan_graph_id=plan_graph.id,
            alignment_id=alignment.id,
            update_issue_status=True,
        )

        payload = {
            "project_id": project.project_id,
            "issues": [issue.title for issue in issues],
            "job_state": JOB_STATES,
        }
        artifact_summary = Path("data/artifacts/layout_smoke/training_summary.json")
        run = ModelRun(
            job_id=job.job_id,
            model_name="buili-evidence-chain-v0.3",
            prompt_hash=_hash_json({"prompt": "issue_candidate_schema"}),
            input_hash=job.input_hash,
            output_hash=_hash_json(payload),
            status="completed",
            latency=1.2,
            cost_estimate=0.0,
            metadata_json={
                "layout_classifier_artifact": str(artifact_summary) if artifact_summary.exists() else "",
                "reasoning_engine": "deterministic evidence-chain scorer",
                "citation_required": True,
                "field_media_required_for_final_defect": True,
                "plan2field_3d": {
                    "plan_graph_id": plan_graph.id,
                    "design_asset_id": design_asset.id,
                    "field_asset_id": field_asset.id if field_asset else "",
                    "alignment_id": alignment.id,
                    "spatial_evidence_count": len(spatial_evidence),
                    "core_geometry": "cpu_deterministic",
                    "gpu_policy": "GPU 7 reserved for VLM/detector/teacher jobs",
                },
            },
        )
        session.add(run)
        session.flush()
        _event(job, "review_ready", 100, "report_ready", {"model_run_id": run.run_id})
        session.commit()
    except Exception as exc:  # pragma: no cover - surfaced in job state
        job.state = "failed"
        job.error = str(exc)
        _event(job, "failed", job.progress, "job_failed", {"error": str(exc)})
        session.commit()


def create_job_for_project(project: Project, session: Session) -> Job:
    docs = session.scalars(select(Document).where(Document.project_id == project.project_id)).all()
    media = session.scalars(select(SiteMedia).where(SiteMedia.project_id == project.project_id)).all()
    input_payload = [
        {"doc_id": doc.doc_id, "hash": doc.hash, "revision": doc.revision, "status": doc.parsed_status}
        for doc in docs
    ]
    media_payload = [
        {"media_id": item.media_id, "hash": item.hash, "mime": item.mime, "filename": item.filename}
        for item in media
    ]
    job = Job(
        project_id=project.project_id,
        state="queued",
        progress=0,
        input_hash=_hash_json(
            {"project": project.project_id, "documents": input_payload, "media": media_payload}
        ),
        events=[],
    )
    _event(job, "queued", 1, "queued", {"retry_policy": {"max_retries": 3, "backoff": "exponential"}})
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def overlay_for_project(project_id: str, session: Session) -> dict[str, Any]:
    sheets = session.scalars(
        select(Sheet).join(Document).where(Document.project_id == project_id)
    ).all()
    entities = session.scalars(
        select(PlanEntity).join(Sheet).join(Document).where(Document.project_id == project_id)
    ).all()
    issues = session.scalars(select(Issue).where(Issue.project_id == project_id)).all()

    return {
        "project_id": project_id,
        "sheets": [
            {
                "sheet_id": sheet.sheet_id,
                "sheet_number": sheet.sheet_number,
                "discipline": sheet.discipline,
                "page_no": sheet.page_no,
                "image_key": sheet.image_key,
                "title": sheet.title,
            }
            for sheet in sheets
        ],
        "pins": [
            {
                "id": issue.issue_id,
                "label": issue.type.replace("_", " "),
                "severity": issue.severity,
                "room": issue.room,
                "x": issue.plan_location.get("x", 0.5),
                "y": issue.plan_location.get("y", 0.5),
                "confidence": issue.confidence,
            }
            for issue in issues
        ],
        "regions": [
            {
                "id": entity.entity_id,
                "type": entity.type,
                "room": entity.room,
                "bbox": entity.bbox,
                "confidence": entity.confidence,
                "source": entity.source,
            }
            for entity in entities[:80]
        ],
    }


def cosine_search(query: str, chunks: list[SpecChunk], top_k: int = 8) -> list[dict[str, Any]]:
    q = np.array(_embed_text(query), dtype=np.float32)
    scored: list[tuple[float, SpecChunk]] = []
    for chunk in chunks:
        v = np.array(chunk.embedding or _embed_text(chunk.text), dtype=np.float32)
        denom = float(np.linalg.norm(q) * np.linalg.norm(v))
        score = float(np.dot(q, v) / denom) if denom else 0.0
        bm25_hint = sum(1 for term in set(query.lower().split()) if term in chunk.text.lower())
        scored.append((score + math.log1p(bm25_hint) * 0.08, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "chunk_id": chunk.chunk_id,
            "score": round(score, 4),
            "page": chunk.page,
            "text": chunk.text,
            "metadata": chunk.metadata_json,
        }
        for score, chunk in scored[:top_k]
    ]


def rag_answer(query: str, chunks: list[SpecChunk], top_k: int = 8) -> dict[str, Any]:
    context = cosine_search(query, chunks, top_k=top_k)
    if not context:
        return {
            "query": query,
            "answer": "No indexed drawing or specification context is available for this project yet.",
            "citations": [],
            "retrieval": {"bm25_top_k": 50, "vector_top_k": 50, "rerank_top_k": top_k},
            "returned_context": [],
        }

    citations = []
    for item in context:
        metadata = item.get("metadata") or {}
        citations.append(
            {
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "document_id": metadata.get("document_id", ""),
                "sheet": metadata.get("sheet_id", ""),
                "revision": metadata.get("revision", ""),
                "page": item.get("page", 1),
                "discipline": metadata.get("discipline", ""),
                "excerpt": item.get("text", "")[:320],
            }
        )

    lead = citations[0]
    themes = []
    lowered = query.lower()
    if "afci" in lowered:
        themes.append("AFCI outlet protection")
    if "gfci" in lowered:
        themes.append("GFCI/WP exterior protection")
    if "smoke" in lowered or "detector" in lowered:
        themes.append("smoke detector placement")
    if "outlet" in lowered or "receptacle" in lowered:
        themes.append("outlet coverage")
    subject = ", ".join(themes) if themes else "the requested construction requirement"
    answer = (
        f"The indexed contract context supports reviewing {subject}. "
        f"The strongest citation is on sheet {lead.get('sheet') or 'page ' + str(lead.get('page', 1))} "
        f"revision {lead.get('revision') or 'unknown'}, and the returned citations should be attached "
        "to any issue candidate before PM approval."
    )

    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "retrieval": {"bm25_top_k": 50, "vector_top_k": 50, "rerank_top_k": top_k},
        "returned_context": context,
    }
