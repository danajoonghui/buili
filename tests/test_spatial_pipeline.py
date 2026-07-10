from urllib.parse import urlsplit

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from services.api.buili.main import app as api_app
from services.api.buili.spatial.floorplan_extractor import AxisSegment, snap_and_merge_segments
from services.api.buili.spatial.geometry import build_design_glb
from services.api.buili.spatial.semantic_auto import (
    TextItem,
    _dimension_text,
    _filter_floorplan_segments,
    _room_labels_from_text,
    evaluate_scene_alignment,
    semantic_scene_to_plan_graph_payload,
)
from services.api.buili.spatial.eval_metrics import evaluate_plan_elements
from services.api.buili.spatial.semantic_scene import (
    SemanticObject,
    SemanticScene,
    SemanticWall,
    SourceTransform,
    build_maricopa_source_aligned_scene,
    render_semantic_scene,
)
from ml.collect_plan2field_eval50 import parse_cubicasa_svg


def _upload(
    client: TestClient, project_id: str, filename: str, mime: str, body: bytes, kind: str
) -> dict:
    presign = client.post(
        "/v1/uploads/presign",
        json={
            "project_id": project_id,
            "filename": filename,
            "mime": mime,
            "size": len(body),
            "kind": kind,
        },
    )
    assert presign.status_code == 200
    upload = client.post(
        urlsplit(presign.json()["upload_url"]).path,
        files={"file": (filename, body, mime)},
    )
    assert upload.status_code == 200
    complete = client.post(
        urlsplit(presign.json()["complete_url"]).path,
        json={"document_type": "plan", "revision": "E1.1"},
    )
    assert complete.status_code == 200
    return complete.json()


def _create_spatial_project(client: TestClient) -> dict:
    project = client.post(
        "/v1/projects",
        json={
            "name": "Spatial QA Cooper E1.1",
            "address": "QA",
            "project_type": "tenant_improvement",
        },
    ).json()
    body = (
        b"E1.1 Electrical Plans\n"
        b"Conference Room 203 north wall requires two duplex outlets.\n"
        b"AFCI protection, GFCI/WP exterior receptacles, smoke detector placement, "
        b"and ceiling fan switching require field verification.\n"
    )
    _upload(client, project["project_id"], "e11-plan.txt", "text/plain", body, "document")
    _upload(
        client,
        project["project_id"],
        "field.jpg",
        "image/jpeg",
        b"\xff\xd8field\xff\xd9",
        "media",
    )
    job = client.post(
        f"/v1/projects/{project['project_id']}/analyze", json={"spatial": True}
    ).json()
    refreshed = client.get(f"/v1/jobs/{job['job_id']}").json()
    assert refreshed["state"] == "review_ready"
    return project


def test_plan2field_spatial_pipeline_roundtrip() -> None:
    client = TestClient(api_app)
    project = _create_spatial_project(client)
    project_id = project["project_id"]

    graph_response = client.get(f"/v1/projects/{project_id}/spatial/plan-graph")
    assert graph_response.status_code == 200
    graph = graph_response.json()
    graph_json = graph["graph_json"]
    assert graph_json["rooms"]
    assert graph_json["walls"]
    assert graph_json["fixtures"]
    assert graph_json["sources"]
    assert graph_json["extraction"]["source_required_for_strong_evidence"] is True

    design_response = client.post(
        f"/v1/projects/{project_id}/spatial/design-3d",
        json={"plan_graph_id": graph["id"], "force": True},
    )
    assert design_response.status_code == 200
    design = design_response.json()
    assert design["type"] == "design_glb"
    download = client.get(f"/v1/spatial-assets/{design['id']}/download")
    assert download.status_code == 200
    assert download.content[:4] == b"glTF"

    media = client.get(f"/v1/projects/{project_id}/media").json()
    frame_response = client.post(
        f"/v1/projects/{project_id}/spatial/field-frame",
        json={
            "media_id": media[0]["media_id"],
            "timestamp": 12.5,
            "rgb_uri": "frame://qa/12.5.jpg",
            "depth_uri": "depth://qa/12.5.bin",
            "intrinsics_json": {"fx": 500, "fy": 500, "cx": 320, "cy": 240},
            "pose_json": {"translation": [0.1, 0.0, 0.2], "rotation": [0, 0, 0, 1]},
            "blur_score": 0.12,
            "room_hint": "Conference Room 203",
        },
    )
    assert frame_response.status_code == 200

    alignment_response = client.post(
        f"/v1/projects/{project_id}/spatial/align",
        json={
            "plan_graph_id": graph["id"],
            "anchor_pairs": [
                {"plan": [0, 0], "field": [0.02, 0.03], "label": "door corner"},
                {"plan": [4, 0], "field": [4.01, 0.02], "label": "north wall endpoint"},
                {"plan": [4, 3], "field": [4.03, 3.01], "label": "south wall endpoint"},
            ],
        },
    )
    assert alignment_response.status_code == 200
    alignment = alignment_response.json()
    assert alignment["confidence"] >= 0.9

    issues = client.get(f"/v1/projects/{project_id}/issues").json()
    assert issues
    assert issues[0]["spatial_context"]["spatial_evidence_id"]
    compare_response = client.post(
        f"/v1/projects/{project_id}/spatial/compare",
        json={
            "plan_graph_id": graph["id"],
            "alignment_id": alignment["id"],
            "issue_ids": [issues[0]["issue_id"]],
            "update_issue_status": False,
        },
    )
    assert compare_response.status_code == 200
    compared = compare_response.json()
    assert compared["evidence"]
    features = compared["evidence"][0]["geometry_features_json"]
    assert features["comparison_mode"] == "feature_comparison_not_mesh_diff"
    assert "room_alignment_confidence" in features

    issue_spatial = client.get(f"/v1/issues/{issues[0]['issue_id']}/spatial")
    assert issue_spatial.status_code == 200
    assert issue_spatial.json()

    statuses = client.get(f"/v1/projects/{project_id}/technology-status").json()
    plan2field = next(item for item in statuses if item["key"] == "plan2field_3d")
    assert plan2field["status"] == "ready"
    assert plan2field["evidence_count"] >= 1


def test_design_glb_unions_wall_segments_before_extrusion() -> None:
    graph = {
        "rooms": [
            {
                "id": "room_101",
                "name": "Open Office",
                "polygon": [[0, 0], [6, 0], [6, 4], [0, 4]],
            }
        ],
        "walls": [
            {"id": "w1", "from": [0, 0], "to": [6, 0], "height_m": 2.7},
            {"id": "w2", "from": [6, 0], "to": [6, 4], "height_m": 2.7},
            {"id": "w3", "from": [6, 4], "to": [0, 4], "height_m": 2.7},
            {"id": "w4", "from": [0, 4], "to": [0, 0], "height_m": 2.7},
        ],
        "openings": [],
        "fixtures": [],
    }

    uri, metadata = build_design_glb(graph, "test_union_geometry", "spa_union")

    assert uri.endswith("spa_union_design.glb")
    assert metadata["assembly"] == "deterministic_plangraph_union_geometry"
    assert metadata["wall_union_enabled"] is True
    assert metadata["wall_source_segments"] == 4
    assert metadata["wall_polygon_parts"] == 1
    assert metadata["floor_polygon_parts"] == 1
    assert metadata["grid"]["spacing_m"] == 1.0
    assert metadata["triangle_count"] > 0


def test_floorplan_segments_snap_and_merge_without_closing_large_openings() -> None:
    segments = [
        AxisSegment("h", fixed_px=100, start_px=10, end_px=100, thickness_px=6),
        AxisSegment("h", fixed_px=106, start_px=108, end_px=220, thickness_px=7),
        AxisSegment("h", fixed_px=101, start_px=285, end_px=360, thickness_px=6),
        AxisSegment("v", fixed_px=400, start_px=20, end_px=130, thickness_px=8),
        AxisSegment("v", fixed_px=407, start_px=138, end_px=260, thickness_px=7),
    ]

    merged = snap_and_merge_segments(
        segments,
        snap_tolerance_px=8,
        merge_gap_px=14,
        min_length_px=40,
    )

    horizontal = [segment for segment in merged if segment.orientation == "h"]
    vertical = [segment for segment in merged if segment.orientation == "v"]
    assert len(horizontal) == 2
    assert horizontal[0].start_px == 10
    assert horizontal[0].end_px == 220
    assert horizontal[1].start_px == 285
    assert horizontal[1].end_px == 360
    assert len(vertical) == 1
    assert vertical[0].start_px == 20
    assert vertical[0].end_px == 260


def test_plan2field_eval_metrics_match_objects_openings_and_walls() -> None:
    ground_truth = {
        "objects": [{"id": "gt_obj", "kind": "sink", "bbox": [10, 10, 50, 40]}],
        "openings": [{"id": "gt_door", "kind": "door", "bbox": [100, 20, 160, 42]}],
        "walls": [{"id": "gt_wall", "segment": [0, 0, 100, 0]}],
    }
    prediction = {
        "objects": [{"id": "pred_obj", "kind": "sink", "bbox": [12, 11, 51, 39]}],
        "openings": [{"id": "pred_door", "kind": "door", "bbox": [101, 20, 158, 44]}],
        "walls": [{"id": "pred_wall", "segment": [0, 2, 100, 2]}],
    }

    metrics = evaluate_plan_elements(prediction, ground_truth)

    assert metrics["objects"]["true_positive"] == 1
    assert metrics["openings"]["true_positive"] == 1
    assert metrics["walls"]["true_positive"] == 1


def test_cubicasa_svg_parser_extracts_wall_opening_and_object(tmp_path) -> None:
    svg = tmp_path / "model.svg"
    svg.write_text(
        """<?xml version="1.0"?>
<svg width="200" height="120" viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg">
  <g id="Wall" class="Wall External">
    <polygon points="10,10 190,10 190,20 10,20"/>
    <g id="Door" class="Door Swing Beside">
      <polygon points="60,10 90,10 90,20 60,20"/>
    </g>
    <g id="Window" class="Window Regular">
      <polygon points="120,10 160,10 160,20 120,20"/>
    </g>
  </g>
  <g class="FixedFurniture Sink" transform="matrix(1,0,0,1,40,50)">
    <g class="BoundaryPolygon"><polygon points="0,0 30,0 30,20 0,20"/></g>
  </g>
</svg>
""",
        encoding="utf-8",
    )

    parsed = parse_cubicasa_svg(svg)

    assert parsed["counts"]["walls"] == 1
    assert parsed["counts"]["openings"] == 2
    assert parsed["counts"]["objects"] == 1
    assert parsed["objects"][0]["kind"] == "sink"


def test_source_aligned_semantic_scene_renders_visible_3d_preview(tmp_path) -> None:
    scene = build_maricopa_source_aligned_scene()
    counts = scene.to_json()["counts"]

    assert counts["walls"] >= 30
    assert counts["openings"] >= 12
    assert counts["objects"] >= 16
    assert counts["labels"] >= 12
    assert counts["dimensions"] >= 5
    assert all(opening.mark for opening in scene.openings if opening.kind == "window")
    assert scene.transform.point((0, 0))[1] > scene.transform.point((0, 900))[1]

    preview = tmp_path / "maricopa_source_aligned_preview.png"
    summary = render_semantic_scene(scene, preview)
    graph_payload = semantic_scene_to_plan_graph_payload(
        scene,
        project_id="qa",
        sheet_id="E1.1",
        scale={"px_per_meter": 62.5, "source": "test"},
        source_doc_id="doc_qa",
        source_filename="maricopa.pdf",
    )

    assert preview.exists()
    assert preview.stat().st_size > 20_000
    assert summary["counts"] == counts
    assert summary["render_contract"] == "deterministic_source_px_to_scene_m_low_poly_presentation"
    assert summary["surface_count"] > 600
    assert summary["wall_render_segments"] > counts["walls"]
    assert summary["wall_opening_gaps"] >= 10
    assert {"sink", "toilet", "water_heater"}.issubset(set(summary["procedural_asset_modules"]))
    assert graph_payload["extraction"]["method"] == "automatic_pdf_semantic_scene_v1"
    assert graph_payload["fixtures"][0]["center_m"]


def test_auto_semantic_helpers_extract_rooms_and_filter_page_decoration() -> None:
    labels = _room_labels_from_text(
        [
            TextItem("LIVING ROOM", (100, 100, 180, 118), 0.99, "test"),
            TextItem("16'-6\" x 21'-0\"", (106, 120, 175, 134), 0.92, "test"),
            TextItem("Maricopa County footer", (0, 900, 200, 920), 0.99, "test"),
        ]
    )

    assert labels[0].name == "LIVING ROOM"
    assert labels[0].number == "16'-6\" x 21'-0\""
    assert _dimension_text("3040SH") == ""
    assert _dimension_text('30x45"') == ""
    assert _dimension_text('246" x 236') == "24'-6\" x 23'-6\""

    segments = [
        AxisSegment("h", fixed_px=940, start_px=0, end_px=1980, thickness_px=2),
        AxisSegment("h", fixed_px=300, start_px=500, end_px=900, thickness_px=8),
        AxisSegment("h", fixed_px=360, start_px=520, end_px=920, thickness_px=8),
        AxisSegment("h", fixed_px=420, start_px=520, end_px=920, thickness_px=8),
        AxisSegment("v", fixed_px=560, start_px=300, end_px=700, thickness_px=8),
        AxisSegment("v", fixed_px=620, start_px=300, end_px=700, thickness_px=8),
        AxisSegment("v", fixed_px=680, start_px=300, end_px=700, thickness_px=8),
        AxisSegment("v", fixed_px=500, start_px=300, end_px=700, thickness_px=8),
    ]
    filtered = _filter_floorplan_segments(segments, image_width=1980, image_height=1200)

    assert len(filtered) == 7
    assert all(segment.fixed_px != 940 for segment in filtered)


def test_scene_alignment_qa_measures_source_pixel_residuals(tmp_path) -> None:
    crop = tmp_path / "crop.png"
    image = Image.new("RGB", (120, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.line([(10, 20), (110, 20)], fill="black", width=4)
    draw.rectangle([50, 44, 70, 62], outline="black", width=3)
    image.save(crop)

    scene = SemanticScene(
        source_pdf="qa.pdf",
        source_page_png="page.png",
        source_crop_png=str(crop),
        transform=SourceTransform(width_px=120, height_px=80),
        walls=[
            SemanticWall(
                id="wall_1",
                start_px=(10, 20),
                end_px=(110, 20),
                wall_type="interior",
            )
        ],
        openings=[],
        objects=[
            SemanticObject(
                id="sink_1",
                kind="sink",
                center_px=(60, 53),
                width_px=24,
                depth_px=22,
                label="sink",
            )
        ],
        labels=[],
        tags=[],
        dimensions=[],
        source_scope="qa",
    )

    qa = evaluate_scene_alignment(scene, crop, tmp_path / "overlay.png")

    assert qa["coordinate_transform_error_px"] == 0.0
    assert qa["wall_distance_to_source_dark_px"]["p95_px"] <= 1.0
    assert qa["weak_object_count"] == 0
    assert (tmp_path / "overlay.png").exists()
