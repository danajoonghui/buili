from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from uuid import uuid4
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .config import get_settings
from .models import EvidenceLink, FieldEvidence, Issue, Project
from .workflows import get_or_create_issue_workflow, source_snapshot_for_issue

REPORT_TITLES = {
    "punch": "Punch List",
    "co_evidence": "Change Order Evidence Package",
    "rfi": "RFI Draft Package",
}


def _report_dir(project_id: str) -> Path:
    settings = get_settings()
    path = settings.storage_root / "reports" / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _text(value: object) -> str:
    return str(value or "").replace("\r", " ").strip()


def _report_title(report_type: str) -> str:
    return REPORT_TITLES.get(report_type, report_type.replace("_", " ").title())


def _evidence_refs(issue: Issue) -> str:
    refs = []
    for item in issue.evidence or []:
        parts = [item.evidence_type, item.label or item.ref_id]
        if item.page:
            parts.append(f"p.{item.page}")
        if item.frame_ts:
            parts.append(f"{item.frame_ts:.1f}s")
        refs.append(" / ".join(_text(part) for part in parts if _text(part)))
    return "; ".join(refs)


def _location_label(issue: Issue) -> str:
    location = issue.plan_location or {}
    parts = [
        issue.room,
        str(location.get("floor") or ""),
        str(location.get("wall") or ""),
        f"Grid {location.get('grid')}" if location.get("grid") else "",
        f"Sheet {location.get('sheet_number') or location.get('sheet_id')}"
        if location.get("sheet_number") or location.get("sheet_id")
        else "",
    ]
    return " · ".join(dict.fromkeys(part for part in parts if part))


def _report_context(session: Session, issue: Issue) -> dict[str, object]:
    workflow = get_or_create_issue_workflow(session, issue)
    sources = source_snapshot_for_issue(session, issue)
    evidence: list[dict[str, object]] = []
    links = list(
        session.scalars(select(EvidenceLink).where(EvidenceLink.issue_id == issue.issue_id)).all()
    )
    for link in links:
        item = session.get(FieldEvidence, link.evidence_id)
        if not item:
            continue
        location = item.location_json or {}
        location_text = " · ".join(
            str(location.get(key))
            for key in ("floor", "room", "wall", "grid")
            if location.get(key)
        )
        evidence.append(
            {
                "evidence_id": item.evidence_id,
                "filename": item.filename,
                "media_type": item.media_type,
                "author": item.author,
                "captured_at": item.captured_at.isoformat() + "Z" if item.captured_at else "",
                "location": location_text,
                "hash": item.hash,
                "relevance": link.relevance,
                "annotation": link.annotation,
                "sufficiency": item.sufficiency,
            }
        )
    return {
        "workflow": workflow,
        "sources": sources,
        "evidence": evidence,
        "location": _location_label(issue),
    }


def _source_label(source: dict[str, object]) -> str:
    label = str(source.get("sheet_number") or source.get("filename") or "Contract source")
    revision = str(source.get("revision") or "")
    state = str(source.get("state") or "unverified")
    issue_date = str(source.get("issue_date") or "")
    details = [label, f"Rev {revision}" if revision else "", state, issue_date]
    return " · ".join(part for part in details if part)


def _evidence_caption(item: dict[str, object]) -> str:
    digest = str(item.get("hash") or "")
    parts = [
        str(item.get("filename") or item.get("evidence_id") or "Evidence"),
        str(item.get("media_type") or ""),
        str(item.get("author") or "unknown author"),
        str(item.get("captured_at") or "capture time unavailable"),
        str(item.get("location") or "location unavailable"),
        f"SHA-256 {digest}" if digest else "hash unavailable",
    ]
    return " · ".join(part for part in parts if part)


def _impact_label(workflow) -> str:  # type: ignore[no-untyped-def]
    impact = workflow.impact_json or {}
    if not impact:
        return "No impact basis recorded."
    return "; ".join(f"{key.title()}: {_text(value)}" for key, value in impact.items() if value)


def _issue_row(issue: Issue, context: dict[str, object] | None = None) -> dict[str, object]:
    requirement = issue.requirement or {}
    observation = issue.observation or {}
    spatial = issue.spatial_context or {}
    geometry = spatial.get("geometry_features") or {}
    confidence = float(issue.confidence or 0.0)
    return {
        "issue_id": issue.issue_id,
        "type": issue.type,
        "discipline": issue.discipline,
        "severity": issue.severity,
        "room": str((context or {}).get("location") or issue.room),
        "confidence": f"{confidence:.2f}",
        "status": issue.status,
        "title": issue.title,
        "requirement_source": requirement.get("source", ""),
        "requirement": requirement.get("text", ""),
        "observation_media": observation.get("media_id", ""),
        "observation": observation.get("text", ""),
        "recommended_action": issue.recommended_action,
        "rfi_question": issue.rfi_draft,
        "evidence_count": len(issue.evidence or []),
        "evidence_refs": _evidence_refs(issue),
        "spatial_evidence_id": spatial.get("spatial_evidence_id", ""),
        "spatial_note": spatial.get("spatial_note", ""),
        "alignment_confidence": geometry.get("room_alignment_confidence", ""),
        "geometry_confidence": geometry.get("geometry_confidence", ""),
        "needs_more_evidence": geometry.get("needs_more_evidence", ""),
        "responsible_trade": issue.subcontractor or issue.assignee,
        "due_date": issue.due_date,
        "current_sources": "; ".join(
            _source_label(source) for source in (context or {}).get("sources", [])
        ),
        "field_evidence": "; ".join(
            _evidence_caption(item) for item in (context or {}).get("evidence", [])
        ),
    }


REPORT_FIELDS = [
    "issue_id",
    "type",
    "discipline",
    "severity",
    "room",
    "confidence",
    "status",
    "title",
    "requirement_source",
    "requirement",
    "observation_media",
    "observation",
    "recommended_action",
    "rfi_question",
    "evidence_count",
    "evidence_refs",
    "spatial_evidence_id",
    "spatial_note",
    "alignment_confidence",
    "geometry_confidence",
    "needs_more_evidence",
    "responsible_trade",
    "due_date",
    "current_sources",
    "field_evidence",
]


def _report_rows(
    issues: list[Issue], contexts: dict[str, dict[str, object]] | None = None
) -> list[dict[str, object]]:
    return [_issue_row(issue, (contexts or {}).get(issue.issue_id)) for issue in issues]


def build_csv_report(
    project: Project,
    issues: list[Issue],
    report_type: str,
    contexts: dict[str, dict[str, object]] | None = None,
) -> Path:
    path = _report_dir(project.project_id) / f"{report_type}_{uuid4().hex[:8]}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(_report_rows(issues, contexts))
    return path


def _xml_cell(value: object, row_idx: int, col_idx: int) -> str:
    col = ""
    n = col_idx
    while n:
        n, rem = divmod(n - 1, 26)
        col = chr(65 + rem) + col
    text = xml_escape(_text(value))
    return f'<c r="{col}{row_idx}" t="inlineStr"><is><t>{text}</t></is></c>'


def build_xlsx_report(
    project: Project,
    issues: list[Issue],
    report_type: str,
    contexts: dict[str, dict[str, object]] | None = None,
) -> Path:
    path = _report_dir(project.project_id) / f"{report_type}_{uuid4().hex[:8]}.xlsx"
    rows = _report_rows(issues, contexts)
    sheet_rows = [
        REPORT_FIELDS,
        *[[row.get(header, "") for header in REPORT_FIELDS] for row in rows],
    ]
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        + "".join(
            f'<row r="{row_idx}">'
            + "".join(
                _xml_cell(value, row_idx, col_idx) for col_idx, value in enumerate(row, start=1)
            )
            + "</row>"
            for row_idx, row in enumerate(sheet_rows, start=1)
        )
        + "</sheetData></worksheet>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
            'relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            'officeDocument" '
            'Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Buili Report" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/>'
            "</Relationships>",
        )
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    return path


def _paragraph(text: object, style) -> Paragraph:
    return Paragraph(xml_escape(_text(text)), style)


def _labeled_paragraph(label: str, value: object, style) -> Paragraph:  # type: ignore[no-untyped-def]
    return Paragraph(
        f"<b>{xml_escape(_text(label))}:</b> {xml_escape(_text(value) or 'Not provided')}",
        style,
    )


def _type_specific_fields(
    issue: Issue, report_type: str, context: dict[str, object]
) -> list[tuple[str, str]]:
    workflow = context["workflow"]
    sources = context.get("sources", [])
    source_text = "; ".join(_source_label(source) for source in sources) or "No current source"
    requirement = issue.requirement or {}
    observation = issue.observation or {}
    citation = " · ".join(
        part
        for part in (
            str(requirement.get("source") or ""),
            str(requirement.get("revision") or ""),
            str(requirement.get("citation") or ""),
        )
        if part
    )
    common = [
        ("Issue ID", issue.issue_id),
        ("Exact location", str(context.get("location") or issue.room)),
        ("Current contract source", source_text),
        ("Source citation", citation or str(requirement.get("source") or "Not provided")),
    ]
    if report_type == "rfi":
        return [
            *common,
            ("Subject", issue.title),
            ("Contract requirement", str(requirement.get("text") or "")),
            ("Existing field condition", str(observation.get("text") or "")),
            ("Ambiguity / difference", workflow.difference or issue.description),
            ("Question requiring response", issue.rfi_draft),
            ("Potential impact", _impact_label(workflow)),
            ("Requested action", issue.recommended_action),
        ]
    if report_type == "punch":
        return [
            *common,
            ("Defect / observed condition", str(observation.get("text") or issue.description)),
            (
                "Required completed condition",
                workflow.expected_condition or str(requirement.get("text") or ""),
            ),
            ("Responsible trade", issue.subcontractor or issue.assignee),
            ("Assigned to", issue.assignee),
            ("Due date", issue.due_date),
            ("Corrective action", issue.recommended_action),
            ("Before evidence", "See linked evidence manifest below."),
            ("After evidence", "Completion evidence required before closure."),
        ]
    return [
        *common,
        ("Observed condition", str(observation.get("text") or issue.description)),
        ("Contract baseline", str(requirement.get("text") or "")),
        ("Potential impact basis", _impact_label(workflow)),
        ("Preservation action", issue.recommended_action),
        (
            "Commercial limitation",
            "Evidence package only; no entitlement, responsibility, cost, or schedule determination is made.",
        ),
    ]


def build_pdf_report(
    project: Project,
    issues: list[Issue],
    report_type: str,
    contexts: dict[str, dict[str, object]],
    *,
    artifact_status: str = "draft",
    artifact_version: int = 1,
    artifact_reviewer: str = "",
) -> Path:
    path = _report_dir(project.project_id) / f"{report_type}_{uuid4().hex[:8]}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter, title=f"Buili {report_type} report")
    styles = getSampleStyleSheet()
    story = [
        _paragraph(f"Buili {_report_title(report_type)}", styles["Title"]),
        _paragraph(project.name, styles["Heading2"]),
        _paragraph(project.address or "No address provided", styles["Normal"]),
        _labeled_paragraph(
            "Report status",
            (
                "ISSUED — immutable approved scope"
                if artifact_status == "issued"
                else "DRAFT — human review required"
            ),
            styles["BodyText"],
        ),
        _labeled_paragraph("Version", str(artifact_version), styles["BodyText"]),
        _labeled_paragraph(
            "Reviewer",
            artifact_reviewer or "Pending authorized reviewer",
            styles["BodyText"],
        ),
        _paragraph(
            "Generated from drawing/spec citations, plan pins, field observations, "
            "and review state.",
            styles["BodyText"],
        ),
        Spacer(1, 16),
    ]
    data = [["Issue", "Room", "Severity", "Confidence", "Status"]]
    for issue in issues:
        data.append(
            [
                _paragraph(issue.title, styles["BodyText"]),
                issue.room,
                issue.severity,
                f"{issue.confidence:.2f}",
                issue.status,
            ]
        )
    if len(data) == 1:
        data.append(["No issue candidates", "", "", "", ""])
    table = Table(data, colWidths=[175, 105, 60, 65, 70])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7edf3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d0d8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 18))
    for index, issue in enumerate(issues):
        context = contexts.get(issue.issue_id, {"workflow": None, "sources": [], "evidence": []})
        workflow = context.get("workflow")
        if index:
            story.append(PageBreak())
        story.extend(
            [
                _paragraph(issue.title, styles["Heading2"]),
                _labeled_paragraph("Package type", _report_title(report_type), styles["BodyText"]),
                _labeled_paragraph(
                    "Review state",
                    getattr(workflow, "review_status", issue.status),
                    styles["BodyText"],
                ),
                _labeled_paragraph(
                    "Reviewer",
                    getattr(workflow, "reviewer", "") or "Pending authorized reviewer",
                    styles["BodyText"],
                ),
                Spacer(1, 8),
            ]
        )
        for label, value in _type_specific_fields(issue, report_type, context):
            story.append(_labeled_paragraph(label, value, styles["BodyText"]))
            story.append(Spacer(1, 3))
        story.append(Spacer(1, 7))
        story.append(_paragraph("Evidence / attachment manifest", styles["Heading3"]))
        evidence = context.get("evidence", [])
        if evidence:
            for item in evidence:
                story.append(_paragraph(f"• {_evidence_caption(item)}", styles["BodyText"]))
        else:
            story.append(_paragraph("No location-confirmed field evidence linked.", styles["BodyText"]))
        story.append(
            _labeled_paragraph(
                "Plan/source links",
                _evidence_refs(issue) or "No source attachments linked",
                styles["BodyText"],
            )
        )
    doc.build(story)
    return path


def build_markdown_rfi(
    issue: Issue,
    *,
    sources: list[dict[str, object]] | None = None,
    impact: dict[str, object] | None = None,
) -> str:
    source_text = "; ".join(_source_label(source) for source in (sources or []))
    citation = " · ".join(
        str((issue.requirement or {}).get(key) or "")
        for key in ("source", "revision", "citation")
        if (issue.requirement or {}).get(key)
    )
    impact_text = "; ".join(
        f"{key.title()}: {_text(value)}" for key, value in (impact or {}).items() if value
    )
    return (
        f"# RFI Draft: {issue.title}\n\n"
        "**Status:** DRAFT — authorized review required\n\n"
        f"**Exact location:** {_location_label(issue)}\n\n"
        f"**Current source:** {source_text or 'No current source verified'}\n\n"
        f"**Source citation:** {citation or 'No citation provided'}\n\n"
        f"**Contract requirement:** {issue.requirement.get('text', 'No requirement text')}\n\n"
        f"**Existing field condition:** {issue.observation.get('text', 'No field observation')}\n\n"
        f"**Question requiring response:** {issue.rfi_draft}\n\n"
        f"**Potential impact:** {impact_text or 'No impact basis recorded'}\n\n"
        f"**Attachments:** {_evidence_refs(issue) or 'No attachments linked'}\n\n"
        "This draft is AI-assisted and requires PM review before sending.\n"
    )


def build_markdown_report(
    project: Project,
    issues: list[Issue],
    report_type: str,
    contexts: dict[str, dict[str, object]],
    *,
    artifact_status: str = "draft",
    artifact_version: int = 1,
    artifact_reviewer: str = "",
) -> Path:
    path = _report_dir(project.project_id) / f"{report_type}_{uuid4().hex[:8]}.md"
    lines = [
        f"# Buili {_report_title(report_type)}",
        "",
        f"Project: {project.name}",
        f"Address: {project.address or 'No address provided'}",
        (
            "Status: ISSUED — immutable approved scope"
            if artifact_status == "issued"
            else "Status: DRAFT — human review required"
        ),
        f"Version: {artifact_version}",
        f"Reviewer: {artifact_reviewer or 'Pending authorized reviewer'}",
        "",
    ]
    if not issues:
        lines.extend(["No issue candidates are available.", ""])
    for index, issue in enumerate(issues, start=1):
        context = contexts.get(issue.issue_id, {"workflow": None, "sources": [], "evidence": []})
        lines.extend(
            [
                f"## {index}. {issue.title}",
                "",
                f"- Location: {context.get('location') or issue.room}",
                f"- Severity: {issue.severity}",
                f"- Confidence: {issue.confidence:.2f}",
                f"- Requirement: {issue.requirement.get('text', '')}",
                f"- Observation: {issue.observation.get('text', '')}",
                f"- Evidence: {_evidence_refs(issue) or 'None linked'}",
                f"- Spatial evidence: {(issue.spatial_context or {}).get('spatial_evidence_id', 'not linked')}",
                f"- Spatial note: {(issue.spatial_context or {}).get('spatial_note', '')}",
                f"- Recommended action: {issue.recommended_action}",
            ]
        )
        lines.extend(
            f"- {label}: {value}" for label, value in _type_specific_fields(issue, report_type, context)
        )
        lines.append("- Evidence / attachment manifest:")
        evidence = context.get("evidence", [])
        lines.extend(
            f"  - {_evidence_caption(item)}" for item in evidence
        )
        if not evidence:
            lines.append("  - No location-confirmed field evidence linked.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_report(
    session: Session,
    project_id: str,
    report_type: str,
    fmt: str,
    *,
    issue_ids: list[str] | None = None,
    artifact_status: str = "draft",
    artifact_version: int = 1,
    artifact_reviewer: str = "",
) -> tuple[str, Path]:
    project = session.get(Project, project_id)
    if not project:
        raise ValueError("project not found")
    statement = (
        select(Issue)
        .options(selectinload(Issue.evidence), selectinload(Issue.spatial_evidence))
        .where(Issue.project_id == project_id)
    )
    if issue_ids is not None:
        statement = statement.where(Issue.issue_id.in_(issue_ids))
    issues = session.scalars(statement.order_by(Issue.confidence.desc())).all()
    if issue_ids is not None:
        by_id = {issue.issue_id: issue for issue in issues}
        missing = [issue_id for issue_id in issue_ids if issue_id not in by_id]
        if missing:
            raise ValueError(f"issues not found in project: {', '.join(missing)}")
        # Preserve the user's explicit builder order across every export format.
        issues = [by_id[issue_id] for issue_id in issue_ids]
    contexts = {issue.issue_id: _report_context(session, issue) for issue in issues}
    if fmt == "csv":
        path = build_csv_report(project, list(issues), report_type, contexts)
    elif fmt == "xlsx":
        path = build_xlsx_report(project, list(issues), report_type, contexts)
    elif fmt == "md":
        path = build_markdown_report(
            project,
            list(issues),
            report_type,
            contexts,
            artifact_status=artifact_status,
            artifact_version=artifact_version,
            artifact_reviewer=artifact_reviewer,
        )
    else:
        path = build_pdf_report(
            project,
            list(issues),
            report_type,
            contexts,
            artifact_status=artifact_status,
            artifact_version=artifact_version,
            artifact_reviewer=artifact_reviewer,
        )
    return uuid4().hex[:12], path
