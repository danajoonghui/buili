from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

LA_INSPECTIONS_URL = "https://data.lacity.org/resource/9w5z-rg2h.json"
SEATTLE_PERMITS_URL = "https://data.seattle.gov/resource/76t5-zqzr.json"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"

PDF_SOURCES = [
    {
        "key": "utah_cooper_e11",
        "name": "Utah.gov Cooper Residence public electrical plan PDF",
        "url": "https://www.utah.gov/pmn/files/1020117.pdf",
        "license_status": "government_public_access_review_before_commercial_reuse",
        "trade": "electrical",
    },
    {
        "key": "minnesota_electrical_checklist",
        "name": "Minnesota DLI electrical inspection checklist",
        "url": "https://www.dli.mn.gov/sites/default/files/pdf/eli_inspection_checklist2.pdf",
        "license_status": "government_public_access_review_before_commercial_reuse",
        "trade": "electrical",
    },
]

COMMONS_QUERIES = [
    "electrical wiring construction site",
    "construction electrical wiring",
    "rough-in electrical wiring",
    "junction box wiring construction",
    "building electrical conduit",
]

SYSTEM_PROMPT = (
    "You are Buili, a construction AI reviewer. Use only the cited public source and "
    "return compact valid JSON. Never make a final defect decision without human review."
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def request_json(url: str, params: dict[str, Any] | None = None) -> Any:
    full_url = url
    if params:
        full_url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        full_url,
        headers={"User-Agent": "BuiliOpenDataCollector/0.1 contact=research-demo"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BuiliOpenDataCollector/0.1 contact=research-demo"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def normalized_trade(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["electric", "wiring", "power"]):
        return "electrical"
    if any(token in lowered for token in ["vent", "hvac", "mechanical"]):
        return "mechanical"
    if any(token in lowered for token in ["plumb", "sewer", "water"]):
        return "plumbing"
    if "fire" in lowered:
        return "fire_safety"
    return "building"


def normalized_result(result: str) -> str:
    lowered = result.lower()
    if "partial" in lowered:
        return "partial_approval"
    if "approved" in lowered or "approval" in lowered or "pass" in lowered:
        return "approved"
    if "correction" in lowered or "fail" in lowered or "denied" in lowered:
        return "correction_required"
    if "cancel" in lowered:
        return "cancelled"
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_") or "unknown"


def risk_from_result(result: str) -> str:
    normalized = normalized_result(result)
    if normalized in {"correction_required", "partial_approval"}:
        return "major"
    if normalized == "approved":
        return "informational"
    return "needs_review"


def collect_la_inspections(limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = {
        "$limit": limit,
        "$select": (
            "address,permit,permit_status,inspection_date,inspection,"
            "inspection_result,lat_lon"
        ),
        "$order": "inspection_date DESC",
    }
    rows = request_json(LA_INSPECTIONS_URL, params)
    labels: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        inspection = str(row.get("inspection") or "")
        result = str(row.get("inspection_result") or "")
        labels.append(
            {
                "id": f"la-inspection-{index:05d}",
                "source": "Los Angeles DBS open inspections API",
                "source_url": LA_INSPECTIONS_URL,
                "license_status": "government_open_data_terms_apply",
                "address": row.get("address", ""),
                "permit": row.get("permit", ""),
                "permit_status": row.get("permit_status", ""),
                "inspection_date": row.get("inspection_date", ""),
                "inspection": inspection,
                "inspection_result": result,
                "normalized_result": normalized_result(result),
                "trade": normalized_trade(inspection),
                "severity_label": risk_from_result(result),
                "recommended_action": (
                    "Route partial/correction inspections to PM review with permit, trade, "
                    "and follow-up evidence requirements."
                ),
                "lat_lon": row.get("lat_lon", {}),
            }
        )
    return labels, {
        "name": "Los Angeles Department of Building and Safety inspections",
        "url": LA_INSPECTIONS_URL,
        "license_status": "government_open_data_terms_apply",
        "records": len(labels),
        "used_for": ["inspection_result_labels", "risk_routing", "report_generation"],
    }


def collect_seattle_permits(limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = {
        "$limit": limit,
        "$select": (
            "permitnum,permitclassmapped,permittypemapped,permittypedesc,description,"
            "statuscurrent,daysoutcorrections,numberreviewcycles,originaladdress1,"
            "originalcity,originalstate,originalzip,link"
        ),
        "$order": "permitnum DESC",
    }
    rows = request_json(SEATTLE_PERMITS_URL, params)
    labels: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        cycles = int(float(row.get("numberreviewcycles") or 0))
        days_out = int(float(row.get("daysoutcorrections") or 0))
        review_label = "expedite_review" if days_out > 0 or cycles >= 2 else "normal_review"
        labels.append(
            {
                "id": f"seattle-permit-{index:05d}",
                "source": "Seattle building permits open data API",
                "source_url": SEATTLE_PERMITS_URL,
                "license_status": "government_open_data_terms_apply",
                "permit": row.get("permitnum", ""),
                "permit_class": row.get("permitclassmapped", ""),
                "permit_type": row.get("permittypemapped", ""),
                "status": row.get("statuscurrent", ""),
                "description": row.get("description", ""),
                "days_out_corrections": days_out,
                "review_cycles": cycles,
                "review_label": review_label,
                "trade": normalized_trade(
                    f"{row.get('permittypedesc', '')} {row.get('description', '')}"
                ),
                "recommended_action": (
                    "Use correction days and review cycles as weak labels for review "
                    "urgency and report-routing prioritization."
                ),
                "record_link": (row.get("link") or {}).get("url", ""),
            }
        )
    return labels, {
        "name": "Seattle building permits open data",
        "url": SEATTLE_PERMITS_URL,
        "license_status": "government_open_data_terms_apply",
        "records": len(labels),
        "used_for": ["permit_review_labels", "risk_routing", "report_generation"],
    }


def commons_search(query: str, limit: int) -> list[dict[str, Any]]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": 1280,
    }
    payload = request_json(COMMONS_API_URL, params)
    pages = (payload.get("query") or {}).get("pages") or {}
    return list(pages.values())


def collect_commons_images(
    out_dir: Path, max_images: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    per_query = max(3, max_images // max(len(COMMONS_QUERIES), 1) + 2)
    for query in COMMONS_QUERIES:
        for page in commons_search(query, per_query):
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url")
            mime = str(info.get("mime") or "")
            if not url or url in seen_urls or not mime.startswith("image/"):
                continue
            seen_urls.add(url)
            try:
                data = download_bytes(url)
                digest = sha256_bytes(data)
                suffix = ".jpg" if "jpeg" in mime else ".png"
                image_path = out_dir / f"commons-{len(records):04d}-{digest[:10]}{suffix}"
                image_path.write_bytes(data)
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception:
                continue
            metadata = info.get("extmetadata") or {}
            license_short = (metadata.get("LicenseShortName") or {}).get("value", "unknown")
            license_url = (metadata.get("LicenseUrl") or {}).get("value", "")
            records.append(
                {
                    "id": f"commons-field-{len(records):04d}",
                    "source": "Wikimedia Commons API",
                    "source_url": info.get("descriptionurl") or info.get("url") or url,
                    "title": page.get("title", ""),
                    "query": query,
                    "image": str(image_path),
                    "sha256": digest,
                    "width": width,
                    "height": height,
                    "mime": mime,
                    "license": license_short,
                    "license_url": license_url,
                    "license_status": "file_level_license_recorded_review_before_commercial_use",
                    "weak_labels": [
                        normalized_trade(query),
                        "field_photo",
                        "construction_context",
                    ],
                    "human_review_required": True,
                }
            )
            if len(records) >= max_images:
                break
        if len(records) >= max_images:
            break
    return records, {
        "name": "Wikimedia Commons construction field images",
        "url": COMMONS_API_URL,
        "license_status": "file_level_license_recorded_review_before_commercial_use",
        "records": len(records),
        "used_for": ["field_media_recognition", "evidence_gate_vlm_sft"],
    }


def extract_requirements(text: str) -> list[dict[str, Any]]:
    requirements = []
    for idx, line in enumerate(re.split(r"\n+|(?<=[.;])\s+", text)):
        clean = " ".join(line.split())
        lowered = clean.lower()
        if len(clean) < 28:
            continue
        if any(
            token in lowered
            for token in [
                "inspection",
                "shall",
                "required",
                "verify",
                "gfci",
                "afci",
                "smoke",
                "outlet",
                "rough",
                "permit",
            ]
        ):
            requirements.append(
                {
                    "id": f"req-{idx:04d}",
                    "text": clean[:360],
                    "trade": normalized_trade(clean),
                    "confidence": 0.72,
                    "human_review_required": True,
                }
            )
        if len(requirements) >= 24:
            break
    return requirements


def collect_public_pdfs(
    raw_dir: Path, image_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source in PDF_SOURCES:
        try:
            data = download_bytes(source["url"])
        except Exception as exc:
            records.append({**source, "status": "download_failed", "error": str(exc)})
            continue
        pdf_path = raw_dir / f"{source['key']}.pdf"
        pdf_path.write_bytes(data)
        digest = sha256_file(pdf_path)
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") for page in doc[: min(4, doc.page_count)])
        page_images: list[str] = []
        for page_index in range(min(2, doc.page_count)):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            image_path = image_dir / f"{source['key']}-page-{page_index + 1}.jpg"
            pix.save(image_path)
            page_images.append(str(image_path))
        records.append(
            {
                **source,
                "status": "downloaded",
                "path": str(pdf_path),
                "sha256": digest,
                "page_count": doc.page_count,
                "text_sha256": sha256_bytes(text.encode("utf-8")),
                "requirements": extract_requirements(text),
                "rendered_images": page_images,
            }
        )
    return records, {
        "name": "Public plan/checklist PDFs",
        "url": "multiple_government_public_urls",
        "license_status": "government_public_access_review_before_commercial_reuse",
        "records": len([item for item in records if item.get("status") == "downloaded"]),
        "used_for": ["pdf_rag", "requirement_extraction", "plan_trace_vlm_sft"],
    }


def build_sft_rows(
    pdfs: list[dict[str, Any]],
    images: list[dict[str, Any]],
    inspections: list[dict[str, Any]],
    permits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pdf in pdfs:
        if pdf.get("status") != "downloaded":
            continue
        answer = {
            "task": "public_pdf_requirement_extraction",
            "source": pdf["name"],
            "license_status": pdf["license_status"],
            "requirements": pdf.get("requirements", [])[:10],
            "human_review_required": True,
            "final_defect_decision": False,
        }
        rows.append(
            {
                "id": f"pdf-{pdf['key']}-requirements",
                "split": "train",
                "task": "public_pdf_requirement_extraction",
                "system": SYSTEM_PROMPT,
                "images": pdf.get("rendered_images", [])[:1],
                "prompt": (
                    "Extract PM-review-ready construction requirements from this public "
                    "plan/checklist page. Include license_status and human review gate."
                ),
                "answer": json.dumps(answer, ensure_ascii=False, sort_keys=True),
            }
        )
    for record in images:
        answer = {
            "task": "field_media_observation",
            "source": record["source"],
            "source_url": record["source_url"],
            "license_status": record["license_status"],
            "observations": [
                {
                    "element": label,
                    "condition": "weak label from public image search and metadata",
                    "confidence": 0.58,
                }
                for label in record["weak_labels"]
            ],
            "evidence_gate": {
                "status": "needs_room_tagged_closeup_before_defect_decision",
                "human_review_required": True,
            },
            "final_defect_decision": False,
        }
        rows.append(
            {
                "id": record["id"],
                "split": "train" if len(rows) % 5 else "eval",
                "task": "field_media_observation",
                "system": SYSTEM_PROMPT,
                "images": [record["image"]],
                "prompt": (
                    "Create a compact construction field evidence package from this public "
                    "image. Do not assert a defect; list observations and evidence limits."
                ),
                "answer": json.dumps(answer, ensure_ascii=False, sort_keys=True),
            }
        )
    paired = inspections[:80] + permits[:80]
    for index, label in enumerate(paired):
        prompt_text = (
            "Convert this public inspection/permit record into Buili routing labels. "
            f"Record: {json.dumps(label, ensure_ascii=False)[:1200]}"
        )
        answer = {
            "task": "inspection_record_routing",
            "source": label["source"],
            "source_url": label["source_url"],
            "license_status": label["license_status"],
            "trade": label.get("trade", "building"),
            "severity": label.get("severity_label", label.get("review_label", "needs_review")),
            "recommended_action": label["recommended_action"],
            "human_review_required": True,
            "final_defect_decision": False,
        }
        rows.append(
            {
                "id": f"inspection-routing-{index:05d}",
                "split": "train" if index % 8 else "eval",
                "task": "inspection_record_routing",
                "system": SYSTEM_PROMPT,
                "images": [],
                "prompt": prompt_text,
                "answer": json.dumps(answer, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def try_hf_construction_site(limit: int) -> dict[str, Any]:
    if limit <= 0:
        return {
            "name": "LouisChen15/ConstructionSite",
            "status": "not_requested",
            "license_status": "cc-by-nc-4.0_research_only",
        }
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {
            "name": "LouisChen15/ConstructionSite",
            "status": "skipped_missing_hf_token",
            "license_status": "cc-by-nc-4.0_research_only",
        }
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "LouisChen15/ConstructionSite",
            split="train",
            streaming=True,
            token=token,
        )
        count = 0
        for _ in dataset.take(limit):
            count += 1
        return {
            "name": "LouisChen15/ConstructionSite",
            "status": "accessible_research_only",
            "records_seen": count,
            "license_status": "cc-by-nc-4.0_research_only_not_for_commercial_training",
        }
    except Exception as exc:
        return {
            "name": "LouisChen15/ConstructionSite",
            "status": "inaccessible_or_gated",
            "error": str(exc)[:500],
            "license_status": "cc-by-nc-4.0_research_only_not_for_commercial_training",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/processed/open_construction_corpus")
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/open_corpus"))
    parser.add_argument("--la-limit", type=int, default=600)
    parser.add_argument("--seattle-limit", type=int, default=600)
    parser.add_argument("--commons-images", type=int, default=24)
    parser.add_argument("--hf-construction-samples", type=int, default=20)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    inspections, la_source = collect_la_inspections(args.la_limit)
    permits, seattle_source = collect_seattle_permits(args.seattle_limit)
    commons, commons_source = collect_commons_images(args.out_dir / "images", args.commons_images)
    pdfs, pdf_source = collect_public_pdfs(args.raw_dir / "pdfs", args.out_dir / "pdf_pages")
    hf_source = try_hf_construction_site(args.hf_construction_samples)
    sft_rows = build_sft_rows(pdfs, commons, inspections, permits)

    write_jsonl(args.out_dir / "inspection_labels.jsonl", inspections)
    write_jsonl(args.out_dir / "permit_review_labels.jsonl", permits)
    write_jsonl(args.out_dir / "field_media_labels.jsonl", commons)
    write_jsonl(args.out_dir / "public_pdf_requirements.jsonl", pdfs)
    write_jsonl(args.out_dir / "sft_dataset.jsonl", sft_rows)

    manifest = {
        "built_at": int(time.time()),
        "corpus_version": "open_construction_corpus_v0.1",
        "records": {
            "la_inspection_labels": len(inspections),
            "seattle_permit_labels": len(permits),
            "commons_field_images": len(commons),
            "public_pdfs": len([item for item in pdfs if item.get("status") == "downloaded"]),
            "sft_rows": len(sft_rows),
        },
        "sources": [la_source, seattle_source, commons_source, pdf_source, hf_source],
        "commercial_readiness": {
            "usable_now_with_review": [
                "government open data inspection/permit labels",
                "public government PDFs with source attribution and legal review",
                "Wikimedia Commons images after file-level license review",
            ],
            "research_only_or_blocked": [
                "LouisChen15/ConstructionSite is gated and CC-BY-NC-4.0",
                "Roboflow/HF construction safety conversions require license verification",
            ],
            "production_gap": (
                "Silicon-Valley-grade commercial deployment still needs licensed customer "
                "field photos, plan PDFs, and inspector/PM acceptance labels."
            ),
        },
        "sha256": {
            "inspection_labels": sha256_file(args.out_dir / "inspection_labels.jsonl"),
            "permit_review_labels": sha256_file(args.out_dir / "permit_review_labels.jsonl"),
            "field_media_labels": sha256_file(args.out_dir / "field_media_labels.jsonl"),
            "public_pdf_requirements": sha256_file(args.out_dir / "public_pdf_requirements.jsonl"),
            "sft_dataset": sha256_file(args.out_dir / "sft_dataset.jsonl"),
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
