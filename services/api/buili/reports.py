from __future__ import annotations

import csv
from pathlib import Path
from uuid import uuid4

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import Issue, Project


def _report_dir(project_id: str) -> Path:
    settings = get_settings()
    path = settings.storage_root / "reports" / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_csv_report(project: Project, issues: list[Issue], report_type: str) -> Path:
    path = _report_dir(project.project_id) / f"{report_type}_{uuid4().hex[:8]}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "issue_id",
                "type",
                "severity",
                "room",
                "confidence",
                "status",
                "title",
                "recommended_action",
            ],
        )
        writer.writeheader()
        for issue in issues:
            writer.writerow(
                {
                    "issue_id": issue.issue_id,
                    "type": issue.type,
                    "severity": issue.severity,
                    "room": issue.room,
                    "confidence": issue.confidence,
                    "status": issue.status,
                    "title": issue.title,
                    "recommended_action": issue.recommended_action,
                }
            )
    return path


def build_pdf_report(project: Project, issues: list[Issue], report_type: str) -> Path:
    path = _report_dir(project.project_id) / f"{report_type}_{uuid4().hex[:8]}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter, title=f"Buili {report_type} report")
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"Buili {report_type.replace('_', ' ').title()} Report", styles["Title"]),
        Paragraph(project.name, styles["Heading2"]),
        Paragraph(project.address or "No address provided", styles["Normal"]),
        Spacer(1, 16),
    ]
    data = [["Issue", "Room", "Severity", "Confidence", "Status"]]
    for issue in issues:
        data.append(
            [
                Paragraph(issue.title, styles["BodyText"]),
                issue.room,
                issue.severity,
                f"{issue.confidence:.2f}",
                issue.status,
            ]
        )
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
        story.extend(
            [
                Paragraph(issue.title, styles["Heading3"]),
                Paragraph(f"Requirement: {issue.requirement.get('text', '')}", styles["BodyText"]),
                Paragraph(f"Observation: {issue.observation.get('text', '')}", styles["BodyText"]),
                Paragraph(f"Recommended action: {issue.recommended_action}", styles["BodyText"]),
                Spacer(1, 10),
            ]
        )
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


def build_report(session: Session, project_id: str, report_type: str, fmt: str) -> tuple[str, Path]:
    project = session.get(Project, project_id)
    if not project:
        raise ValueError("project not found")
    issues = session.scalars(select(Issue).where(Issue.project_id == project_id)).all()
    if fmt == "csv":
        path = build_csv_report(project, list(issues), report_type)
    else:
        path = build_pdf_report(project, list(issues), report_type)
    return uuid4().hex[:12], path

