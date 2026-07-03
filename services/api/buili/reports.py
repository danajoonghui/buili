from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from uuid import uuid4
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .config import get_settings
from .models import Issue, Project

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


def _issue_row(issue: Issue) -> dict[str, object]:
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
        "room": issue.room,
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
]


def _report_rows(issues: list[Issue]) -> list[dict[str, object]]:
    return [_issue_row(issue) for issue in issues]


def build_csv_report(project: Project, issues: list[Issue], report_type: str) -> Path:
    path = _report_dir(project.project_id) / f"{report_type}_{uuid4().hex[:8]}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(_report_rows(issues))
    return path


def _xml_cell(value: object, row_idx: int, col_idx: int) -> str:
    col = ""
    n = col_idx
    while n:
        n, rem = divmod(n - 1, 26)
        col = chr(65 + rem) + col
    text = xml_escape(_text(value))
    return f'<c r="{col}{row_idx}" t="inlineStr"><is><t>{text}</t></is></c>'


def build_xlsx_report(project: Project, issues: list[Issue], report_type: str) -> Path:
    path = _report_dir(project.project_id) / f"{report_type}_{uuid4().hex[:8]}.xlsx"
    rows = _report_rows(issues)
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


def build_pdf_report(project: Project, issues: list[Issue], report_type: str) -> Path:
    path = _report_dir(project.project_id) / f"{report_type}_{uuid4().hex[:8]}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter, title=f"Buili {report_type} report")
    styles = getSampleStyleSheet()
    story = [
        _paragraph(f"Buili {_report_title(report_type)}", styles["Title"]),
        _paragraph(project.name, styles["Heading2"]),
        _paragraph(project.address or "No address provided", styles["Normal"]),
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
    table = Table(data, colWidths=[230, 115, 70, 70, 75])
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
    for issue in issues:
        spatial = issue.spatial_context or {}
        story.extend(
            [
                _paragraph(issue.title, styles["Heading3"]),
                _paragraph(f"Requirement: {issue.requirement.get('text', '')}", styles["BodyText"]),
                _paragraph(f"Observation: {issue.observation.get('text', '')}", styles["BodyText"]),
                _paragraph(
                    f"Evidence links: {_evidence_refs(issue) or 'None linked'}", styles["BodyText"]
                ),
                _paragraph(
                    f"Spatial evidence: {spatial.get('spatial_evidence_id', 'not linked')} "
                    f"{spatial.get('spatial_note', '')}",
                    styles["BodyText"],
                ),
                _paragraph(f"Recommended action: {issue.recommended_action}", styles["BodyText"]),
            ]
        )
        if report_type == "rfi":
            story.append(_paragraph(f"RFI question: {issue.rfi_draft}", styles["BodyText"]))
        if report_type == "co_evidence":
            story.append(
                _paragraph(
                    "CO support: Preserve requirement, observation, plan location, "
                    "and field media before pricing.",
                    styles["BodyText"],
                )
            )
        story.append(Spacer(1, 10))
    doc.build(story)
    return path


def build_markdown_rfi(issue: Issue) -> str:
    return (
        f"# RFI Draft: {issue.title}\n\n"
        f"**Location:** {issue.room}\n\n"
        f"**Contract requirement:** {issue.requirement.get('text', 'No requirement text')}\n\n"
        f"**Field observation:** {issue.observation.get('text', 'No field observation')}\n\n"
        f"**Question:** {issue.rfi_draft}\n\n"
        "This draft is AI-assisted and requires PM review before sending.\n"
    )


def build_markdown_report(project: Project, issues: list[Issue], report_type: str) -> Path:
    path = _report_dir(project.project_id) / f"{report_type}_{uuid4().hex[:8]}.md"
    lines = [
        f"# Buili {_report_title(report_type)}",
        "",
        f"Project: {project.name}",
        f"Address: {project.address or 'No address provided'}",
        "",
    ]
    if not issues:
        lines.extend(["No issue candidates are available.", ""])
    for index, issue in enumerate(issues, start=1):
        lines.extend(
            [
                f"## {index}. {issue.title}",
                "",
                f"- Location: {issue.room}",
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
        if report_type == "rfi":
            lines.append(f"- RFI question: {issue.rfi_draft}")
        if report_type == "co_evidence":
            lines.append("- CO support: Bundle citation, plan pin, field media, and PM decision.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_report(session: Session, project_id: str, report_type: str, fmt: str) -> tuple[str, Path]:
    project = session.get(Project, project_id)
    if not project:
        raise ValueError("project not found")
    issues = session.scalars(
        select(Issue)
        .options(selectinload(Issue.evidence), selectinload(Issue.spatial_evidence))
        .where(Issue.project_id == project_id)
        .order_by(Issue.confidence.desc())
    ).all()
    if fmt == "csv":
        path = build_csv_report(project, list(issues), report_type)
    elif fmt == "xlsx":
        path = build_xlsx_report(project, list(issues), report_type)
    elif fmt == "md":
        path = build_markdown_report(project, list(issues), report_type)
    else:
        path = build_pdf_report(project, list(issues), report_type)
    return uuid4().hex[:12], path
