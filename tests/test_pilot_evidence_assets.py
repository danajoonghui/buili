"""Static chain-of-evidence checks for the authenticated Cooper pilot."""

from __future__ import annotations

import re
from pathlib import Path

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "apps" / "web" / "public" / "demo" / "northstar" / "evidence"
SOURCE_PLAN = (
    ROOT
    / "data"
    / "sources"
    / "plan2field3d_public"
    / "utah_cooper_e11_1020117.pdf"
)


def test_voice_note_captions_are_valid_webvtt_and_match_the_pilot_issue() -> None:
    captions = (EVIDENCE / "foreman-voice-note.vtt").read_text(encoding="utf-8")
    timestamps = [line for line in captions.splitlines() if "-->" in line]

    assert captions.startswith("WEBVTT\n\n")
    assert timestamps
    assert all(
        re.fullmatch(
            r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}",
            timestamp,
        )
        for timestamp in timestamps
    )
    assert "," not in "\n".join(timestamps)
    assert "Cooper Residence" in captions
    assert "twelve inches" in captions
    assert "minimum of eighteen inches" in captions
    assert (EVIDENCE / "foreman-voice-note.mp3").stat().st_size > 100_000


def test_pilot_photos_and_thumbnails_are_decodable_originals() -> None:
    originals = (
        "garage-east-wall-context.png",
        "receptacle-rough-in-detail.png",
        "box-elevation-measurement.png",
    )
    for filename in originals:
        path = EVIDENCE / filename
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width >= 1200 and image.height >= 900
            image.verify()
        thumb = EVIDENCE / f"{path.stem}-thumb.webp"
        with Image.open(thumb) as image:
            assert image.format == "WEBP"
            assert image.width <= 720 and image.height <= 720
            image.verify()


def test_current_plan_directly_supports_the_seeded_requirement() -> None:
    with fitz.open(SOURCE_PLAN) as document:
        assert document.page_count == 8
        electrical = document[-1].get_text("text").upper()

    normalized = " ".join(electrical.split())
    assert "COOPER RESIDENCE" in normalized
    assert "305 W. SHANGRI-LA" in normalized
    assert "GFCI PROTECTION OF OUTLETS" in normalized
    assert 'UP MIN. 18" OFF' in normalized
