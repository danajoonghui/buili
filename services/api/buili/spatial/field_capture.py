from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import FieldPoseFrame, SiteMedia, SpatialAsset, new_id


def _spatial_dir(project_id: str) -> Path:
    path = get_settings().storage_root / "spatial" / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def ingest_field_pose_frame(
    session: Session,
    project_id: str,
    *,
    media_id: str,
    timestamp: float = 0.0,
    rgb_uri: str = "",
    depth_uri: str = "",
    intrinsics_json: dict[str, Any] | None = None,
    pose_json: dict[str, Any] | None = None,
    blur_score: float = 0.0,
    room_hint: str = "",
) -> FieldPoseFrame:
    media = session.get(SiteMedia, media_id)
    if not media or media.project_id != project_id:
        raise ValueError("media_id is not part of this project")
    frame = FieldPoseFrame(
        media_id=media_id,
        timestamp=timestamp,
        rgb_uri=rgb_uri or media.r2_key,
        depth_uri=depth_uri,
        intrinsics_json=intrinsics_json or {},
        pose_json=pose_json or {},
        blur_score=blur_score,
        room_hint=room_hint,
    )
    session.add(frame)
    session.flush()
    return frame


def create_field_asset_from_frames(
    session: Session,
    project_id: str,
    *,
    field_asset_id: str | None = None,
) -> SpatialAsset | None:
    media_ids = select(SiteMedia.media_id).where(SiteMedia.project_id == project_id)
    frames = list(
        session.scalars(select(FieldPoseFrame).where(FieldPoseFrame.media_id.in_(media_ids))).all()
    )
    media = list(session.scalars(select(SiteMedia).where(SiteMedia.project_id == project_id)).all())
    if not frames and not media:
        return None

    asset_id = field_asset_id or new_id("spa")
    out_dir = _spatial_dir(project_id)
    filename = f"{asset_id}_field_evidence.json"
    path = out_dir / filename
    frame_payload = [
        {
            "field_pose_frame_id": frame.id,
            "media_id": frame.media_id,
            "timestamp": frame.timestamp,
            "rgb_uri": frame.rgb_uri,
            "depth_uri": frame.depth_uri,
            "has_depth": bool(frame.depth_uri),
            "has_pose": bool(frame.pose_json),
            "room_hint": frame.room_hint,
            "blur_score": frame.blur_score,
        }
        for frame in frames
    ]
    if not frame_payload:
        frame_payload = [
            {
                "media_id": item.media_id,
                "timestamp": 0.0,
                "rgb_uri": item.r2_key,
                "depth_uri": "",
                "has_depth": False,
                "has_pose": False,
                "room_hint": str((item.metadata_json or {}).get("room_hint") or ""),
                "blur_score": 0.0,
            }
            for item in media
        ]
    has_depth = any(item["has_depth"] for item in frame_payload)
    has_pose = any(item["has_pose"] for item in frame_payload)
    payload = {
        "project_id": project_id,
        "asset_type": "field_3d_evidence",
        "frames": frame_payload,
        "coverage": {
            "frame_count": len(frame_payload),
            "has_depth": has_depth,
            "has_pose": has_pose,
            "mode": "rgb_depth_pose" if has_depth and has_pose else "rgb_fallback",
        },
        "safety_note": (
            "Depth/pose evidence is available for guided alignment."
            if has_depth and has_pose
            else (
                "No depth/pose found; route spatial claims through needs_more_evidence "
                "when alignment is weak."
            )
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    asset = SpatialAsset(
        id=asset_id,
        project_id=project_id,
        type="field_evidence_json",
        uri=f"spatial/{project_id}/{filename}",
        metadata_json=payload["coverage"] | {"frame_count": len(frame_payload)},
    )
    session.add(asset)
    session.flush()
    return asset
