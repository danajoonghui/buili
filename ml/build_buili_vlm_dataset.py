from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

PLAN_SOURCE = {
    "name": "Utah.gov Cooper Residence public electrical plan PDF render",
    "url": "https://www.utah.gov/pmn/files/1020117.pdf",
    "license_note": "Publicly accessible permit drawing; use source attribution in demos.",
}

FIELD_SOURCE = {
    "name": "Wikimedia Commons Installing electrical wiring.jpg",
    "url": "https://commons.wikimedia.org/wiki/File:Installing_electrical_wiring.jpg",
    "license_note": "Public-domain field wiring photograph from Wikimedia Commons.",
}


SYSTEM_PROMPT = (
    "You are Buili, a senior construction AI reviewer for MEP/electrical field-to-report "
    "workflows. Produce investor-demo-quality, PM-review-ready outputs. Tie every issue to "
    "a requirement, field evidence, plan location, risk, and next action. Return only valid "
    "JSON. Never make a final defect decision without cited plan/spec evidence and field "
    "verification."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_image(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    with Image.open(dst) as image:
        width, height = image.size
    return {
        "path": str(dst),
        "sha256": sha256_file(dst),
        "width": width,
        "height": height,
    }


def json_answer(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def evidence_chain() -> list[dict[str, Any]]:
    return [
        {
            "step": "requirement",
            "source": "E1.1 electrical notes",
            "finding": (
                "Electrical notes indicate that AFCI/GFCI and smoke detector conditions "
                "must be verified before rough-in approval."
            ),
            "confidence": 0.78,
        },
        {
            "step": "plan_trace",
            "source": "E1.1 plan image",
            "finding": "Device symbols and review pins should be checked against room context.",
            "confidence": 0.74,
        },
        {
            "step": "field_evidence",
            "source": "room-tagged field media required",
            "finding": "Current evidence is not sufficient for a final defect decision.",
            "confidence": 0.68,
        },
    ]


def enterprise_issue(title: str, confidence: float, status: str) -> dict[str, Any]:
    return {
        "title": title,
        "trade": "electrical",
        "discipline": "MEP",
        "severity": "major",
        "status": status,
        "confidence": confidence,
        "executive_summary": (
            f"{title}. Buili should treat this as a review-ready candidate, not a final "
            "defect, until the PM verifies plan citation, room location, and field media."
        ),
        "requirement": {
            "sheet": "E1.1",
            "citation": "Electrical notes and device symbols on E1.1",
            "requirement_type": "rough_in_electrical_verification",
        },
        "observation": {
            "field_state": "verification pending",
            "evidence_quality": "partial",
            "limitation": "The public field photo is not room-tagged to the E1.1 device pin.",
        },
        "evidence_chain": evidence_chain(),
        "risk": {
            "schedule": "inspection delay if discovered after cover",
            "cost": "rework exposure if device protection or placement is wrong",
            "quality": "requires PM and trade foreman verification before closeout",
        },
        "recommended_action": (
            "Verify device coverage against E1.1, capture room-tagged close-up evidence, "
            "and attach the plan pin before rough-in approval."
        ),
        "decision_gate": {
            "approve_allowed": False,
            "allowed_actions": ["more_evidence", "rfi", "reject_candidate"],
            "human_review_required": True,
        },
    }


def report_bundle(report_type: str) -> dict[str, Any]:
    return {
        "report_type": report_type,
        "owner": "Field PM",
        "issue_title": "AFCI/GFCI device verification against E1.1",
        "punch_list_row": {
            "location": "Main Floor / E1.1 verification zone",
            "trade": "electrical",
            "description": (
                "Verify AFCI/GFCI device protection and provide room-tagged close-up "
                "evidence before rough-in approval."
            ),
            "priority": "major",
            "status": "open_pending_verification",
        },
        "rfi_draft": {
            "subject": "Confirm E1.1 AFCI/GFCI verification requirement and acceptable evidence",
            "question": (
                "Please confirm whether AFCI/GFCI device coverage shown on E1.1 requires "
                "field photo evidence at each marked location prior to rough-in approval."
            ),
            "proposed_response": (
                "Contractor to verify device protection and submit room-tagged close-up "
                "evidence tied to marked E1.1 plan pins."
            ),
        },
        "change_order_evidence": {
            "claim_readiness": "not_ready",
            "reason": (
                "Potential cost impact cannot be asserted until plan requirement, field "
                "condition, responsible scope, and corrective labor are verified."
            ),
            "needed_before_pricing": [
                "approved RFI response",
                "before/after field evidence",
                "trade labor/material impact",
                "schedule impact note",
            ],
        },
        "export_targets": ["Procore RFI", "ACC issue", "CSV punch list", "PDF evidence packet"],
    }


def plan_answer(sheet: str, mode: str) -> dict[str, Any]:
    base = {
        "discipline": "electrical",
        "sheet": sheet,
        "source": "public_electrical_plan",
        "human_review_required": True,
        "final_defect_decision": False,
        "product_standard": "silicon_valley_pm_review_ready",
    }
    if mode == "requirements":
        base.update(
            {
                "task": "extract_plan_requirements",
                "executive_summary": (
                    "E1.1 should be treated as an electrical verification source for "
                    "AFCI/GFCI protection, smoke detector placement, and service coordination."
                ),
                "requirements": [
                    {
                        "code": "E1.1-AFCI",
                        "text": (
                            "AFCI protection should be verified for living and sleeping "
                            "area outlets."
                        ),
                        "confidence": 0.78,
                        "business_risk": "failed inspection or rework before cover",
                    },
                    {
                        "code": "E1.1-GFCI-WP",
                        "text": (
                            "Exterior and wet-location receptacles need GFCI/weatherproof "
                            "verification."
                        ),
                        "confidence": 0.74,
                        "business_risk": "weatherproofing deficiency and inspection hold",
                    },
                    {
                        "code": "E1.1-SD",
                        "text": (
                            "Smoke detector locations should be verified against bedroom "
                            "and sleeping area notes."
                        ),
                        "confidence": 0.7,
                        "business_risk": "life-safety review delay",
                    },
                ],
                "evidence_chain": evidence_chain(),
            }
        )
    elif mode == "issue_candidates":
        base.update(
            {
                "task": "generate_issue_candidates",
                "issues": [
                    enterprise_issue(
                        "AFCI outlet coverage below E1.1 requirement",
                        0.78,
                        "awaiting_verification",
                    ),
                    enterprise_issue(
                        "GFCI/WP exterior outlet verification needed",
                        0.72,
                        "needs_more_evidence",
                    ),
                ],
                "portfolio_value": (
                    "The output is suitable for an issue inbox because it separates candidate "
                    "generation from final defect determination."
                ),
            }
        )
    elif mode == "pins":
        base.update(
            {
                "task": "create_plan_trace",
                "pins": [
                    {
                        "label": "AFCI coverage",
                        "x": 0.58,
                        "y": 0.64,
                        "confidence": 0.78,
                        "review_action": "verify room-tagged field evidence",
                    },
                    {
                        "label": "GFCI/WP exterior",
                        "x": 0.86,
                        "y": 0.84,
                        "confidence": 0.72,
                        "review_action": "capture exterior close-up",
                    },
                    {
                        "label": "Smoke detector",
                        "x": 0.41,
                        "y": 0.64,
                        "confidence": 0.66,
                        "review_action": "check sleeping-area placement",
                    },
                ],
                "trace_quality": "candidate_locations_require_pm_confirmation",
            }
        )
    else:
        base.update(
            {
                "task": "classify_sheet",
                "document_type": "electrical_plan",
                "review_focus": ["outlets", "GFCI", "AFCI", "smoke_detectors", "panel"],
                "recommended_workflow": [
                    "extract requirements",
                    "pin candidate locations",
                    "compare field evidence",
                    "generate issue package",
                    "export RFI/punch/CO artifacts",
                ],
            }
        )
    return base


def field_answer(mode: str) -> dict[str, Any]:
    base = {
        "task": mode,
        "source": "public_domain_field_photo",
        "trade": "electrical",
        "human_review_required": True,
        "final_defect_decision": False,
    }
    if mode == "field_observations":
        base.update(
            {
                "observations": [
                    {
                        "element": "rough_opening",
                        "condition": "visible unfinished wall cavity",
                        "confidence": 0.82,
                    },
                    {
                        "element": "cable_bundle",
                        "condition": "visible wiring run in open framing",
                        "confidence": 0.76,
                    },
                    {
                        "element": "junction_box_or_device_location",
                        "condition": "requires close-up verification",
                        "confidence": 0.64,
                    },
                ],
                "evidence_quality": "partial",
                "field_report": {
                    "summary": (
                        "The image supports rough electrical activity detection, but it is "
                        "not sufficient to close E1.1 plan compliance."
                    ),
                    "usable_for": ["evidence intake", "rough-in activity detection"],
                    "not_usable_for": ["final defect decision", "device count verification"],
                },
            }
        )
    else:
        base.update(
            {
                "issue_readiness": "needs_more_evidence",
                "missing_evidence": [
                    "sheet-specific room reference",
                    "close-up of outlet/device label",
                    "before-cover inspection context",
                ],
                "recommended_action": (
                    "Capture room-tagged close-up photos before approving issue closure."
                ),
                "pm_gate": {
                    "status": "blocked_until_more_evidence",
                    "minimum_acceptance": [
                        "room name",
                        "sheet pin",
                        "device close-up",
                        "trade foreman confirmation",
                    ],
                },
            }
        )
    return base


def report_answer(report_type: str) -> dict[str, Any]:
    title = {
        "rfi": "RFI draft for electrical device verification",
        "punch": "Punch list row for rough-in electrical verification",
        "co": "Change order evidence package summary",
    }[report_type]
    return {
        "task": f"generate_{report_type}",
        "title": title,
        "trade": "electrical",
        "sheet": "E1.1",
        "issue": "AFCI/GFCI device verification requires field evidence against E1.1.",
        "executive_summary": (
            "Buili should package this as a human-review item with plan citation, field "
            "evidence gap, risk statement, and export-ready RFI/punch/CO artifacts."
        ),
        "required_evidence": [
            "plan citation",
            "marked plan pin",
            "room-tagged field photo",
            "PM verification decision",
        ],
        "evidence_chain": evidence_chain(),
        "report_bundle": report_bundle(report_type),
        "human_review_required": True,
        "final_defect_decision": False,
    }


def add_row(
    rows: list[dict[str, Any]],
    *,
    sample_id: str,
    images: list[str],
    prompt: str,
    answer: dict[str, Any],
    task: str,
    source_keys: list[str],
) -> None:
    rows.append(
        {
            "id": sample_id,
            "system": SYSTEM_PROMPT,
            "images": images,
            "prompt": prompt,
            "answer": json_answer(answer),
            "task": task,
            "sources": source_keys,
        }
    )


def build_dataset(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_dir = out_dir / "images"
    plan_sources = sorted(Path("apps/web/public/plans").glob("utah-e11-page-*.jpg"))
    full_plan = Path("apps/web/public/plans/utah-e11-electrical-plan.jpg")
    field_src = Path("data/sources/construction-site-electrical-work.jpg")
    if full_plan.exists():
        plan_sources.insert(0, full_plan)
    if not plan_sources:
        raise FileNotFoundError("No rendered Utah E1.1 plan images found.")
    if not field_src.exists():
        raise FileNotFoundError(field_src)

    copied: dict[str, dict[str, Any]] = {}
    for src in plan_sources:
        copied[src.stem] = copy_image(src, image_dir / src.name)
    copied["field_wiring"] = copy_image(field_src, image_dir / "field_wiring.jpg")

    rows: list[dict[str, Any]] = []
    prompt_variants = {
        "requirements": (
            "Inspect this electrical drawing page as a senior PM reviewer. Return an "
            "enterprise-grade JSON requirement extraction with business risk, citation "
            "logic, and evidence chain for Buili."
        ),
        "issue_candidates": (
            "Inspect this plan and produce Silicon-Valley-demo-quality issue packages. "
            "Return JSON with executive summary, severity, confidence, evidence chain, "
            "risk, PM decision gate, and recommended action."
        ),
        "pins": (
            "Create a plan trace suitable for an issue review system. Return JSON with "
            "normalized x/y pins, confidence, review action, and trace quality."
        ),
        "classify": (
            "Classify this sheet for Buili's production pipeline. Return JSON with document "
            "type, discipline, review focus, and recommended workflow."
        ),
    }
    plan_modes = list(prompt_variants)
    for idx, src in enumerate(plan_sources):
        sheet = "E1.1"
        image_path = copied[src.stem]["path"]
        for mode in plan_modes:
            add_row(
                rows,
                sample_id=f"plan-{idx:02d}-{mode}",
                images=[image_path],
                prompt=prompt_variants[mode],
                answer=plan_answer(sheet, mode),
                task=f"plan_{mode}",
                source_keys=["utah_plan"],
            )
        for report_type in ["rfi", "punch", "co"]:
            add_row(
                rows,
                sample_id=f"plan-{idx:02d}-report-{report_type}",
                images=[image_path],
                prompt=(
                    f"Using this plan page, draft the structured {report_type.upper()} support "
                    "Buili should prepare. Return only JSON."
                ),
                answer=report_answer(report_type),
                task=f"report_{report_type}",
                source_keys=["utah_plan"],
            )

    field_image = copied["field_wiring"]["path"]
    for idx in range(24):
        mode = "field_observations" if idx % 2 == 0 else "field_evidence_gate"
        add_row(
            rows,
            sample_id=f"field-{idx:02d}-{mode}",
            images=[field_image],
            prompt=(
                "Inspect this field electrical construction photo for a PM-facing review "
                "workflow. Return JSON observations, evidence limitations, PM gate, and "
                "what cannot be concluded from the image."
            ),
            answer=field_answer(mode),
            task=mode,
            source_keys=["wikimedia_field_photo"],
        )

    for idx, src in enumerate(plan_sources[:6]):
        add_row(
            rows,
            sample_id=f"compare-{idx:02d}",
            images=[copied[src.stem]["path"], field_image],
            prompt=(
                "Compare the plan page and field photo. Decide whether Buili has enough "
                "evidence to close an electrical issue. Return conservative JSON."
            ),
            answer={
                "task": "plan_field_comparison",
                "sheet": "E1.1",
                "decision": "needs_more_evidence",
                "executive_summary": (
                    "The plan and field photo can support a review candidate, but not a "
                    "final defect. Buili should request room-tagged close-ups before issuing "
                    "a correction or change order position."
                ),
                "reason": (
                    "The field photo shows rough electrical work, but it does not prove the "
                    "specific E1.1 outlet/AFCI/GFCI requirement is satisfied."
                ),
                "evidence_chain": evidence_chain(),
                "rfi_readiness": {
                    "ready": True,
                    "question": (
                        "Please confirm the acceptable field evidence required to verify "
                        "E1.1 AFCI/GFCI device protection before rough-in approval."
                    ),
                },
                "punch_readiness": {
                    "ready": True,
                    "row": (
                        "Electrical contractor to provide room-tagged device close-ups tied "
                        "to marked E1.1 plan pins."
                    ),
                },
                "change_order_readiness": {
                    "ready": False,
                    "reason": (
                        "Pricing requires verified scope gap, RFI response, and "
                        "labor/material impact."
                    ),
                },
                "required_next_evidence": [
                    "room-tagged close-up",
                    "marked plan pin",
                    "device count or protection label",
                ],
                "human_review_required": True,
                "final_defect_decision": False,
            },
            task="plan_field_comparison",
            source_keys=["utah_plan", "wikimedia_field_photo"],
        )

    manifest = {
        "dataset": "buili_qwen3_vl_sft",
        "rows": len(rows),
        "sources": {
            "utah_plan": PLAN_SOURCE,
            "wikimedia_field_photo": FIELD_SOURCE,
        },
        "images": copied,
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/buili_vlm"))
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest = build_dataset(args.out_dir)
    split_at = max(1, int(len(rows) * (1 - args.eval_ratio)))
    for idx, row in enumerate(rows):
        row["split"] = "train" if idx < split_at else "eval"
    dataset_path = args.out_dir / "sft_dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest["train_rows"] = sum(row["split"] == "train" for row in rows)
    manifest["eval_rows"] = sum(row["split"] == "eval" for row in rows)
    manifest["sha256"] = sha256_file(dataset_path)
    (args.out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
