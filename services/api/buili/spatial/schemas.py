from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SpatialSourceRef(BaseModel):
    citation_chunk_id: str = ""
    doc_id: str = ""
    sheet_id: str = ""
    bbox: list[float] = Field(default_factory=list)
    source_type: str = "citation_chunk"
    source_strength: Literal["strong", "display_only"] = "strong"


class PlanGraphScale(BaseModel):
    px_per_meter: float = 126.4
    source: str = "default_estimate"
    confidence: float = 0.35


class PlanGraphRoom(BaseModel):
    id: str
    name: str
    polygon: list[list[float]]


class PlanGraphWall(BaseModel):
    id: str
    room_id: str
    from_: list[float] = Field(alias="from")
    to: list[float]
    height_m: float = 2.7

    model_config = {"populate_by_name": True}


class PlanGraphOpening(BaseModel):
    type: str
    wall_id: str
    x_m: float
    width_m: float = 0.9
    source_entity_id: str = ""


class PlanGraphFixture(BaseModel):
    type: str
    room_id: str
    wall_id: str = ""
    required_count: int = 1
    observed_count: int = 0
    bbox: list[float] = Field(default_factory=list)
    source_entity_id: str = ""


class PlanGraphPayload(BaseModel):
    project_id: str
    sheet_id: str
    scale: PlanGraphScale
    rooms: list[PlanGraphRoom] = Field(default_factory=list)
    walls: list[PlanGraphWall] = Field(default_factory=list)
    openings: list[PlanGraphOpening] = Field(default_factory=list)
    fixtures: list[PlanGraphFixture] = Field(default_factory=list)
    sources: list[SpatialSourceRef] = Field(default_factory=list)
    extraction: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class PlanGraphCreateRequest(BaseModel):
    source_doc_id: str | None = None
    sheet_id: str | None = None
    calibration_px: float | None = None
    calibration_m: float | None = None


class PlanGraphOut(BaseModel):
    id: str
    project_id: str
    sheet_id: str
    graph_json: dict[str, Any]
    scale_json: dict[str, Any]
    source_doc_id: str
    version: int

    model_config = {"from_attributes": True}


class Design3DRequest(BaseModel):
    plan_graph_id: str | None = None
    force: bool = False


class SpatialAssetOut(BaseModel):
    id: str
    project_id: str
    type: str
    uri: str
    metadata_json: dict[str, Any]

    model_config = {"from_attributes": True}


class FieldPoseFrameCreate(BaseModel):
    media_id: str
    timestamp: float = 0.0
    rgb_uri: str = ""
    depth_uri: str = ""
    intrinsics_json: dict[str, Any] = Field(default_factory=dict)
    pose_json: dict[str, Any] = Field(default_factory=dict)
    blur_score: float = 0.0
    room_hint: str = ""


class FieldPoseFrameOut(BaseModel):
    id: str
    media_id: str
    timestamp: float
    rgb_uri: str
    depth_uri: str
    intrinsics_json: dict[str, Any]
    pose_json: dict[str, Any]
    blur_score: float
    room_hint: str

    model_config = {"from_attributes": True}


class AnchorPair(BaseModel):
    plan: list[float] = Field(min_length=2, max_length=2)
    field: list[float] = Field(min_length=2, max_length=2)
    label: str = ""


class SpatialAlignmentCreate(BaseModel):
    plan_graph_id: str | None = None
    field_asset_id: str | None = None
    anchor_pairs: list[AnchorPair] = Field(default_factory=list)
    allow_low_confidence: bool = True


class SpatialAlignmentOut(BaseModel):
    id: str
    project_id: str
    plan_graph_id: str
    field_asset_id: str
    transform_json: dict[str, Any]
    anchor_pairs_json: list[dict[str, Any]]
    confidence: float

    model_config = {"from_attributes": True}


class SpatialCompareRequest(BaseModel):
    plan_graph_id: str | None = None
    alignment_id: str | None = None
    issue_ids: list[str] = Field(default_factory=list)
    update_issue_status: bool = True


class SpatialEvidenceOut(BaseModel):
    id: str
    issue_id: str
    room_graph_id: str
    design_asset_id: str
    field_asset_id: str
    geometry_features_json: dict[str, Any]
    snapshot_uri: str
    spatial_note: str

    model_config = {"from_attributes": True}


class SpatialCompareOut(BaseModel):
    plan_graph_id: str
    alignment_id: str
    evidence: list[SpatialEvidenceOut]
