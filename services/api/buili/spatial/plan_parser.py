from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import fitz
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import (
    Document,
    DocumentRevision,
    PlanEntity,
    PlanGraph,
    Project,
    Sheet,
    SpecChunk,
    SpatialAsset,
)
from ..storage import object_path
from .floorplan_extractor import extract_floorplan_payload_from_pdf
from .semantic_auto import build_semantic_scene_from_pdf, semantic_scene_to_plan_graph_payload

FIXTURE_TYPE_MAP = {
    "duplex_outlet": "duplex outlet",
    "data_port": "data port",
    "switch": "switch",
    "ceiling_light": "ceiling light",
    "diffuser": "diffuser",
    "sink": "sink",
    "toilet": "toilet",
    "door": "door",
    "window": "window",
    "cabinet": "cabinet",
}


def document_spatial_provenance(session: Session, document: Document) -> dict[str, Any]:
    revision = session.scalar(
        select(DocumentRevision).where(DocumentRevision.document_id == document.doc_id)
    )
    return {
        "source_doc_id": document.doc_id,
        "source_hash": document.hash,
        "source_revision": document.revision,
        "source_issue_date": revision.issue_date if revision else "",
        "source_revision_id": revision.revision_id if revision else "",
        "source_revision_state": revision.state if revision else "unclassified",
        "source_filename": document.filename,
    }


def plan_graph_provenance(graph: PlanGraph) -> dict[str, Any]:
    payload = graph.graph_json or {}
    provenance = dict(payload.get("provenance") or {})
    provenance.setdefault("source_doc_id", graph.source_doc_id)
    return provenance


def plan_graph_is_current(session: Session, graph: PlanGraph) -> bool:
    provenance = plan_graph_provenance(graph)
    document = session.get(Document, graph.source_doc_id)
    if not document or document.project_id != graph.project_id:
        return False
    revision = session.scalar(
        select(DocumentRevision).where(
            DocumentRevision.document_id == document.doc_id,
            DocumentRevision.project_id == graph.project_id,
            DocumentRevision.state == "current",
        )
    )
    return bool(
        revision
        and provenance.get("source_doc_id") == document.doc_id
        and provenance.get("source_hash") == document.hash
        and provenance.get("source_revision") == document.revision
    )


def spatial_asset_is_current(session: Session, asset: SpatialAsset) -> bool:
    if asset.type != "design_glb":
        return True
    metadata = asset.metadata_json or {}
    graph = session.get(PlanGraph, str(metadata.get("plan_graph_id") or ""))
    if not graph or not plan_graph_is_current(session, graph):
        return False
    provenance = plan_graph_provenance(graph)
    return all(
        metadata.get(key) == provenance.get(key)
        for key in ("source_doc_id", "source_hash", "source_revision")
    )


def _slug(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or fallback


def _room_polygon(seed: int, bbox: list[float] | None = None) -> list[list[float]]:
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[:4]
        width = max((x2 - x1) * 10.0, 2.8)
        depth = max((y2 - y1) * 8.0, 2.4)
        offset_x = max(x1 * 7.0, 0.0)
        offset_y = max(y1 * 6.0, 0.0)
    else:
        width = 3.6 + (seed % 5) * 0.45
        depth = 2.8 + (seed % 7) * 0.32
        offset_x = (seed % 3) * 4.4
        offset_y = (seed % 4) * 3.4
    return [
        [round(offset_x, 3), round(offset_y, 3)],
        [round(offset_x + width, 3), round(offset_y, 3)],
        [round(offset_x + width, 3), round(offset_y + depth, 3)],
        [round(offset_x, 3), round(offset_y + depth, 3)],
    ]


def _walls_for_room(
    room_id: str, polygon: list[list[float]], height_m: float = 2.7
) -> list[dict[str, Any]]:
    walls: list[dict[str, Any]] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        walls.append(
            {
                "id": f"{room_id}_W{index + 1}",
                "room_id": room_id,
                "from": start,
                "to": end,
                "height_m": height_m,
            }
        )
    return walls


def _nearest_wall(room_walls: list[dict[str, Any]], bbox: list[float]) -> str:
    if not room_walls:
        return ""
    if len(bbox) < 4:
        return room_walls[0]["id"]
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    best_id = room_walls[0]["id"]
    best_dist = float("inf")
    for wall in room_walls:
        start = wall["from"]
        end = wall["to"]
        vx = end[0] - start[0]
        vy = end[1] - start[1]
        denom = vx * vx + vy * vy
        t = (
            0.0
            if denom == 0
            else max(
                0.0,
                min(1.0, ((cx - start[0]) * vx + (cy - start[1]) * vy) / denom),
            )
        )
        px = start[0] + t * vx
        py = start[1] + t * vy
        dist = math.hypot(cx - px, cy - py)
        if dist < best_dist:
            best_dist = dist
            best_id = wall["id"]
    return best_id


def _required_count(entity_type: str, chunks: list[SpecChunk]) -> int:
    haystack = "\n".join(chunk.text.lower() for chunk in chunks)
    if entity_type in {"duplex_outlet", "switch"}:
        if "two" in haystack or "2" in haystack:
            return 2
    if entity_type in {"smoke_detector", "ceiling_light", "diffuser"}:
        return 1
    return 1


def _parse_scale(
    chunks: list[SpecChunk], calibration_px: float | None, calibration_m: float | None
) -> dict[str, Any]:
    if calibration_px and calibration_m and calibration_m > 0:
        return {
            "px_per_meter": round(calibration_px / calibration_m, 4),
            "source": "user_calibration",
            "confidence": 0.95,
        }
    text = "\n".join(chunk.text for chunk in chunks)
    scale_match = re.search(
        r"scale\s*[:=]?\s*([0-9/ .]+)[\"']?\s*=\s*([0-9/ .]+)[\"']?",
        text,
        re.I,
    )
    if scale_match:
        return {"px_per_meter": 126.4, "source": "dimension_text_detected", "confidence": 0.62}
    if "e1.1" in text.lower() or "electrical" in text.lower():
        return {"px_per_meter": 126.4, "source": "sheet_default_electrical", "confidence": 0.5}
    return {
        "px_per_meter": 100.0,
        "source": "default_estimate_user_calibration_required",
        "confidence": 0.32,
    }


def _pdf_vector_seed(doc: Document) -> dict[str, Any]:
    path = object_path(doc.r2_key)
    if not path.exists() or path.suffix.lower() != ".pdf":
        return {"available": False, "reason": "not_pdf_or_missing"}
    try:
        pdf = fitz.open(path)
        pages: list[dict[str, Any]] = []
        for page_index, page in enumerate(pdf, start=1):
            drawings = page.get_drawings()
            blocks = page.get_text("blocks")
            pages.append(
                {
                    "page": page_index,
                    "drawing_count": len(drawings),
                    "text_block_count": len(blocks),
                    "rect": [round(float(v), 3) for v in page.rect],
                }
            )
        return {"available": True, "pages": pages}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def build_plan_graph_payload(
    session: Session,
    project: Project,
    *,
    source_doc_id: str | None = None,
    preferred_sheet_id: str | None = None,
    calibration_px: float | None = None,
    calibration_m: float | None = None,
) -> dict[str, Any]:
    current_doc_ids = list(
        session.scalars(
            select(DocumentRevision.document_id).where(
                DocumentRevision.project_id == project.project_id,
                DocumentRevision.state == "current",
            )
        ).all()
    )
    doc_query = select(Document).where(
        Document.project_id == project.project_id,
        Document.type == "plan",
    )
    if source_doc_id:
        if source_doc_id not in current_doc_ids:
            raise ValueError("source document is not the current activated revision")
        doc_query = doc_query.where(Document.doc_id == source_doc_id)
    else:
        if not current_doc_ids:
            raise ValueError("activate a current drawing revision before spatial extraction")
        doc_query = doc_query.where(Document.doc_id.in_(current_doc_ids))
    documents = list(session.scalars(doc_query.order_by(Document.created_at.asc())).all())
    if not documents:
        raise ValueError("no parsed document is available for PlanGraph extraction")

    document_ids = [document.doc_id for document in documents]
    sheet_query = select(Sheet).where(Sheet.doc_id.in_(document_ids))
    if preferred_sheet_id:
        sheet_query = sheet_query.where(
            (Sheet.sheet_id == preferred_sheet_id) | (Sheet.sheet_number == preferred_sheet_id)
        )
    sheets = list(session.scalars(sheet_query.order_by(Sheet.page_no.asc())).all())
    if not sheets:
        raise ValueError("run analysis before extracting PlanGraph; no sheets found")

    selected_sheet = next(
        (sheet for sheet in sheets if sheet.discipline == "electrical"), sheets[0]
    )
    entities = list(
        session.scalars(
            select(PlanEntity).where(PlanEntity.sheet_id == selected_sheet.sheet_id)
        ).all()
    )
    source_doc = next(
        (doc for doc in documents if doc.doc_id == selected_sheet.doc_id), documents[0]
    )
    chunks = list(
        session.scalars(
            select(SpecChunk)
            .where(SpecChunk.doc_id == source_doc.doc_id)
            .order_by(SpecChunk.page.asc())
        ).all()
    )
    provenance = document_spatial_provenance(session, source_doc)
    scale = _parse_scale(chunks, calibration_px, calibration_m)
    source_path = object_path(source_doc.r2_key)

    if source_path.exists() and source_path.suffix.lower() == ".pdf":
        try:
            output_dir = (
                get_settings().storage_root
                / "spatial"
                / project.project_id
                / "automatic_semantic_scene"
            )
            scene, scene_metadata = build_semantic_scene_from_pdf(
                source_path,
                output_dir=output_dir,
                page_no=selected_sheet.page_no,
                use_ocr=True,
            )
            if len(scene.walls) >= 6:
                payload = semantic_scene_to_plan_graph_payload(
                    scene,
                    project_id=project.project_id,
                    sheet_id=selected_sheet.sheet_number,
                    scale=scale,
                    source_doc_id=source_doc.doc_id,
                    source_filename=source_doc.filename,
                )
                payload["extraction"]["scene_build"] = scene_metadata
                payload["extraction"].update(provenance)
                payload["provenance"] = provenance
                for chunk in chunks[:12]:
                    payload["sources"].append(
                        {
                            "citation_chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "sheet_id": str(
                                (chunk.metadata_json or {}).get("sheet_id")
                                or selected_sheet.sheet_number
                            ),
                            "bbox": chunk.bbox or [],
                            "source_type": "citation_chunk",
                            "source_strength": "strong",
                        }
                    )
                return payload
        except Exception as exc:
            auto_semantic_error = str(exc)
    else:
        auto_semantic_error = ""

    if selected_sheet.discipline != "electrical":
        extracted_payload = extract_floorplan_payload_from_pdf(
            source_doc,
            project_id=project.project_id,
            sheet_id=selected_sheet.sheet_number,
            page_no=selected_sheet.page_no,
            scale=scale,
        )
        if extracted_payload and len(extracted_payload.get("walls", [])) >= 6:
            extracted_payload["extraction"]["automatic_semantic_error"] = auto_semantic_error
            extracted_payload["extraction"].update(provenance)
            extracted_payload["provenance"] = provenance
            for chunk in chunks[:12]:
                extracted_payload["sources"].append(
                    {
                        "citation_chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "sheet_id": str(
                            (chunk.metadata_json or {}).get("sheet_id")
                            or selected_sheet.sheet_number
                        ),
                        "bbox": chunk.bbox or [],
                        "source_type": "citation_chunk",
                        "source_strength": "strong",
                    }
                )
            return extracted_payload

    room_entities: dict[str, list[PlanEntity]] = defaultdict(list)
    for entity in entities:
        room_name = entity.room or "Room Pending"
        room_entities[room_name].append(entity)
    if not room_entities:
        room_entities["Room Pending"] = []

    rooms: list[dict[str, Any]] = []
    walls: list[dict[str, Any]] = []
    room_wall_map: dict[str, list[dict[str, Any]]] = {}
    for index, (room_name, grouped) in enumerate(room_entities.items(), start=1):
        room_id = (
            grouped[0].room_id
            if grouped and grouped[0].room_id
            else f"R{index:03d}_{_slug(room_name, 'room')}"
        )
        bbox = grouped[0].bbox if grouped else None
        polygon = _room_polygon(index, bbox)
        room = {"id": room_id, "name": room_name, "polygon": polygon}
        room_walls = _walls_for_room(room_id, polygon)
        rooms.append(room)
        walls.extend(room_walls)
        room_wall_map[room_id] = room_walls

    fixtures: list[dict[str, Any]] = []
    openings: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    room_ids_by_name = {room["name"]: room["id"] for room in rooms}
    for entity in entities:
        room_id = entity.room_id or room_ids_by_name.get(entity.room) or rooms[0]["id"]
        room_walls = room_wall_map.get(room_id, [])
        wall_id = _nearest_wall(room_walls, entity.bbox)
        if entity.type in {"door", "window"}:
            openings.append(
                {
                    "type": entity.type,
                    "wall_id": wall_id,
                    "x_m": round(((entity.bbox[0] + entity.bbox[2]) / 2) * 10.0, 3)
                    if len(entity.bbox) >= 4
                    else 0.0,
                    "width_m": 0.9 if entity.type == "door" else 1.2,
                    "source_entity_id": entity.entity_id,
                }
            )
        else:
            fixtures.append(
                {
                    "type": FIXTURE_TYPE_MAP.get(entity.type, entity.type),
                    "room_id": room_id,
                    "wall_id": wall_id,
                    "required_count": _required_count(entity.type, chunks),
                    "observed_count": 0,
                    "bbox": entity.bbox,
                    "source_entity_id": entity.entity_id,
                }
            )
        sources.append(
            {
                "citation_chunk_id": "",
                "doc_id": source_doc.doc_id,
                "sheet_id": selected_sheet.sheet_number,
                "bbox": entity.bbox,
                "source_type": "plan_entity",
                "source_strength": "strong" if entity.source else "display_only",
            }
        )

    for chunk in chunks[:12]:
        sources.append(
            {
                "citation_chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "sheet_id": str(
                    (chunk.metadata_json or {}).get("sheet_id") or selected_sheet.sheet_number
                ),
                "bbox": chunk.bbox or [],
                "source_type": "citation_chunk",
                "source_strength": "strong",
            }
        )

    return {
        "project_id": project.project_id,
        "sheet_id": selected_sheet.sheet_number,
        "scale": scale,
        "rooms": rooms,
        "walls": walls,
        "openings": openings,
        "fixtures": fixtures,
        "sources": sources,
        "provenance": provenance,
        "extraction": {
            "method": "pymupdf_vector_text_plus_existing_plan_entities",
            "source_doc_id": source_doc.doc_id,
            "source_filename": source_doc.filename,
            "sheet_db_id": selected_sheet.sheet_id,
            "sheet_title": selected_sheet.title,
            "pdf_vector_seed": _pdf_vector_seed(source_doc),
            "automatic_semantic_error": auto_semantic_error,
            "source_required_for_strong_evidence": True,
            **provenance,
        },
    }


def create_plan_graph_record(
    session: Session,
    project: Project,
    *,
    source_doc_id: str | None = None,
    preferred_sheet_id: str | None = None,
    calibration_px: float | None = None,
    calibration_m: float | None = None,
    replace_existing: bool = True,
) -> PlanGraph:
    payload = build_plan_graph_payload(
        session,
        project,
        source_doc_id=source_doc_id,
        preferred_sheet_id=preferred_sheet_id,
        calibration_px=calibration_px,
        calibration_m=calibration_m,
    )
    previous_version = session.scalar(
        select(PlanGraph.version)
        .where(PlanGraph.project_id == project.project_id)
        .order_by(PlanGraph.version.desc())
    ) or 0
    if replace_existing:
        session.execute(delete(PlanGraph).where(PlanGraph.project_id == project.project_id))
        session.flush()
    record = PlanGraph(
        project_id=project.project_id,
        sheet_id=payload["sheet_id"],
        graph_json=payload,
        scale_json=payload["scale"],
        source_doc_id=payload["extraction"].get("source_doc_id", ""),
        version=previous_version + 1,
    )
    session.add(record)
    session.flush()
    return record
