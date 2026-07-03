from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

RGBA = tuple[int, int, int, int]
Point3D = tuple[float, float, float]
Projector = Callable[[Point3D], tuple[float, float]]
PostDraw = Callable[[ImageDraw.ImageDraw, Projector], None]


@dataclass
class Surface:
    points: list[Point3D]
    fill: RGBA
    outline: RGBA | None = None
    width: int = 1
    layer: int = 0


class IsoRenderer:
    def __init__(
        self,
        *,
        width: int = 1800,
        height: int = 1120,
        yaw_deg: float = -38,
        elevation_deg: float = 52,
        supersample: float = 2.0,
    ) -> None:
        self.width = width
        self.height = height
        self.supersample = supersample
        yaw = math.radians(yaw_deg)
        elevation = math.radians(elevation_deg)
        camera = np.array(
            [
                math.cos(elevation) * math.cos(yaw),
                math.sin(elevation),
                math.cos(elevation) * math.sin(yaw),
            ],
            dtype=np.float64,
        )
        self.camera = camera / np.linalg.norm(camera)
        forward = -self.camera
        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, up_world)
        self.right = right / np.linalg.norm(right)
        up = np.cross(self.right, forward)
        self.up = up / np.linalg.norm(up)
        self.surfaces: list[Surface] = []
        self.post_draw: list[PostDraw] = []
        self._center = np.zeros(3, dtype=np.float64)
        self._project_min = np.zeros(2, dtype=np.float64)
        self._scale = 1.0
        self._center_tuple: Point3D = (0.0, 0.0, 0.0)
        self._right_tuple: Point3D = (0.0, 0.0, 0.0)
        self._up_tuple: Point3D = (0.0, 0.0, 0.0)
        self._camera_tuple: Point3D = (0.0, 0.0, 0.0)

    def add_surface(
        self,
        points: list[Point3D],
        fill: RGBA,
        *,
        outline: RGBA | None = None,
        width: int = 1,
        layer: int = 0,
    ) -> None:
        self.surfaces.append(Surface(points, fill, outline, width, layer))

    def add_box(
        self,
        x0: float,
        z0: float,
        x1: float,
        z1: float,
        y0: float,
        y1: float,
        *,
        side: RGBA,
        top: RGBA,
        outline: RGBA | None = None,
        layer: int = 0,
    ) -> None:
        p000 = (x0, y0, z0)
        p100 = (x1, y0, z0)
        p110 = (x1, y0, z1)
        p010 = (x0, y0, z1)
        p001 = (x0, y1, z0)
        p101 = (x1, y1, z0)
        p111 = (x1, y1, z1)
        p011 = (x0, y1, z1)
        self.add_surface([p001, p101, p111, p011], top, outline=outline, layer=layer + 2)
        self.add_surface([p000, p001, p101, p100], side, outline=outline, layer=layer)
        self.add_surface([p100, p101, p111, p110], side, outline=outline, layer=layer)
        self.add_surface([p110, p111, p011, p010], side, outline=outline, layer=layer)
        self.add_surface([p010, p011, p001, p000], side, outline=outline, layer=layer)

    def add_wall(
        self,
        x0: float,
        z0: float,
        x1: float,
        z1: float,
        *,
        thickness: float = 0.14,
        height: float = 2.7,
        side: RGBA = (166, 176, 170, 86),
        cap: RGBA = (33, 37, 34, 255),
        layer: int = 10,
    ) -> None:
        dx = x1 - x0
        dz = z1 - z0
        length = math.hypot(dx, dz)
        if length <= 1e-6:
            return
        px = -dz / length * thickness / 2
        pz = dx / length * thickness / 2
        corners = [
            (x0 + px, z0 + pz),
            (x1 + px, z1 + pz),
            (x1 - px, z1 - pz),
            (x0 - px, z0 - pz),
        ]
        xs = [point[0] for point in corners]
        zs = [point[1] for point in corners]
        # Axis-aligned walls render best as boxes; non-axis walls use a prism-like cap and sides.
        if abs(dx) < 1e-6 or abs(dz) < 1e-6:
            self.add_box(
                min(xs),
                min(zs),
                max(xs),
                max(zs),
                0.0,
                height,
                side=side,
                top=cap,
                outline=(42, 46, 43, 80),
                layer=layer,
            )
            return
        top = [(x, height, z) for x, z in corners]
        bottom = [(x, 0.0, z) for x, z in corners]
        self.add_surface(top, cap, outline=(42, 46, 43, 90), layer=layer + 2)
        for index, start in enumerate(corners):
            end = corners[(index + 1) % len(corners)]
            self.add_surface(
                [
                    (start[0], 0.0, start[1]),
                    (end[0], 0.0, end[1]),
                    (end[0], height, end[1]),
                    (start[0], height, start[1]),
                ],
                side,
                outline=(42, 46, 43, 55),
                layer=layer,
            )
        self.add_surface(bottom, (0, 0, 0, 0), layer=layer - 1)

    def add_floor(self, x0: float, z0: float, x1: float, z1: float) -> None:
        self.add_box(
            x0,
            z0,
            x1,
            z1,
            -0.06,
            0.0,
            side=(202, 206, 197, 255),
            top=(231, 233, 228, 255),
            outline=(160, 164, 156, 90),
            layer=-10,
        )

    def _prepare_projection(self) -> None:
        points = [point for surface in self.surfaces for point in surface.points]
        if not points:
            points = [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0)]
        array = np.array(points, dtype=np.float64)
        self._center = (array.min(axis=0) + array.max(axis=0)) / 2.0
        self._center_tuple = (
            float(self._center[0]),
            float(self._center[1]),
            float(self._center[2]),
        )
        self._right_tuple = (
            float(self.right[0]),
            float(self.right[1]),
            float(self.right[2]),
        )
        self._up_tuple = (
            float(self.up[0]),
            float(self.up[1]),
            float(self.up[2]),
        )
        self._camera_tuple = (
            float(self.camera[0]),
            float(self.camera[1]),
            float(self.camera[2]),
        )
        rel = array - self._center
        projected = np.column_stack((rel @ self.right, rel @ self.up))
        minimum = projected.min(axis=0)
        maximum = projected.max(axis=0)
        size = np.maximum(maximum - minimum, 1e-6)
        pad = 120.0 * self.supersample
        self._scale = min(
            (self.width * self.supersample - pad * 2) / size[0],
            (self.height * self.supersample - pad * 2) / size[1],
        )
        self._project_min = minimum

    def project(self, point: Point3D) -> tuple[float, float]:
        cx, cy, cz = self._center_tuple
        rx, ry, rz = self._right_tuple
        ux, uy, uz = self._up_tuple
        px, py, pz = point
        dx = px - cx
        dy = py - cy
        dz = pz - cz
        projected_x = dx * rx + dy * ry + dz * rz
        projected_y = dx * ux + dy * uy + dz * uz
        pad = 120.0 * self.supersample
        screen_x = (projected_x - float(self._project_min[0])) * self._scale + pad
        screen_y = (projected_y - float(self._project_min[1])) * self._scale + pad
        return float(screen_x), float(self.height * self.supersample - screen_y)

    def _depth(self, surface: Surface) -> float:
        cx, cy, cz = self._center_tuple
        cam_x, cam_y, cam_z = self._camera_tuple
        total = 0.0
        for px, py, pz in surface.points:
            total += (px - cx) * cam_x + (py - cy) * cam_y + (pz - cz) * cam_z
        return total / max(len(surface.points), 1)

    def render(self, path: Path) -> None:
        self._prepare_projection()
        canvas_size = (
            int(self.width * self.supersample),
            int(self.height * self.supersample),
        )
        image = Image.new("RGB", canvas_size, (255, 255, 255))
        shadow = Image.new("RGBA", canvas_size, (255, 255, 255, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        floor_points = [
            self.project((-0.45, -0.08, -0.45)),
            self.project((18.45, -0.08, -0.45)),
            self.project((18.45, -0.08, 10.95)),
            self.project((-0.45, -0.08, 10.95)),
        ]
        shadow_draw.polygon(floor_points, fill=(0, 0, 0, 28))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=int(18 * self.supersample)))
        image.paste(shadow, mask=shadow.getchannel("A"))

        draw = ImageDraw.Draw(image, "RGBA")
        ordered = sorted(self.surfaces, key=lambda surface: (surface.layer, self._depth(surface)))
        for surface in ordered:
            if surface.fill[3] == 0:
                continue
            points = [self.project(point) for point in surface.points]
            draw.polygon(points, fill=surface.fill)
            if surface.outline:
                draw.line(
                    points + [points[0]],
                    fill=surface.outline,
                    width=max(1, int(surface.width * self.supersample)),
                )

        for callback in self.post_draw:
            callback(draw, self.project)

        cropped = self._crop(image)
        final = cropped.resize(
            (int(cropped.width / self.supersample), int(cropped.height / self.supersample)),
            Image.Resampling.LANCZOS,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        final.save(path, compress_level=3)

    @staticmethod
    def _crop(image: Image.Image) -> Image.Image:
        background = Image.new("RGB", image.size, "white")
        diff = ImageChops.difference(image, background).convert("L")
        mask = diff.point(lambda value: 255 if value > 8 else 0)
        bbox = mask.getbbox()
        if not bbox:
            return image
        left, top, right, bottom = bbox
        pad = 80
        return image.crop(
            (
                max(0, left - pad),
                max(0, top - pad),
                min(image.width, right + pad),
                min(image.height, bottom + pad),
            )
        )


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _add_floor_grid(renderer: IsoRenderer, *, x0: float, z0: float, x1: float, z1: float) -> None:
    def draw(draw: ImageDraw.ImageDraw, project: Callable[[Point3D], tuple[float, float]]) -> None:
        step = 1.0
        x = math.ceil(x0 / step) * step
        while x <= x1:
            draw.line(
                [project((x, 0.012, z0)), project((x, 0.012, z1))],
                fill=(116, 120, 114, 72),
                width=1,
            )
            x += step
        z = math.ceil(z0 / step) * step
        while z <= z1:
            draw.line(
                [project((x0, 0.012, z)), project((x1, 0.012, z))],
                fill=(116, 120, 114, 72),
                width=1,
            )
            z += step

    renderer.post_draw.append(draw)


def _add_label(renderer: IsoRenderer, label: str, room_no: str, x: float, z: float) -> None:
    def draw(draw: ImageDraw.ImageDraw, project: Callable[[Point3D], tuple[float, float]]) -> None:
        font_label = _font(30)
        font_no = _font(22)
        sx, sy = project((x, 0.035, z))
        width = draw.textlength(label, font=font_label)
        draw.text((sx - width / 2, sy - 22), label, fill=(32, 34, 32, 225), font=font_label)
        if room_no:
            no_width = draw.textlength(room_no, font=font_no)
            pad_x = 7
            box = [sx - no_width / 2 - pad_x, sy + 12, sx + no_width / 2 + pad_x, sy + 40]
            draw.rectangle(box, outline=(72, 75, 70, 160), width=1)
            draw.text((sx - no_width / 2, sy + 14), room_no, fill=(32, 34, 32, 220), font=font_no)

    renderer.post_draw.append(draw)


def _add_tag(
    renderer: IsoRenderer,
    label: str,
    x: float,
    z: float,
    color: tuple[int, int, int],
    *,
    side: str = "right",
) -> None:
    def draw(draw: ImageDraw.ImageDraw, project: Callable[[Point3D], tuple[float, float]]) -> None:
        font = _font(18, bold=True)
        sx, sy = project((x, 1.25, z))
        anchor = project((x + 0.35, 0.45, z + 0.18))
        rgba = (*color, 230)
        draw.line([sx, sy, anchor[0], anchor[1]], fill=rgba, width=4)
        draw.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], fill=(*color, 255))
        text_width = draw.textlength(label, font=font)
        height = 29
        padding = 10
        if side == "left":
            box = [sx - text_width - padding * 2 - 10, sy - height / 2, sx - 10, sy + height / 2]
        else:
            box = [sx + 10, sy - height / 2, sx + text_width + padding * 2 + 10, sy + height / 2]
        draw.rounded_rectangle(box, radius=6, fill=(*color, 235))
        draw.text((box[0] + padding, box[1] + 4), label, fill=(255, 255, 255, 255), font=font)

    renderer.post_draw.append(draw)


def _add_dimension(
    renderer: IsoRenderer,
    start: Point3D,
    end: Point3D,
    label: str,
) -> None:
    def draw(draw: ImageDraw.ImageDraw, project: Callable[[Point3D], tuple[float, float]]) -> None:
        font = _font(20)
        a = project(start)
        b = project(end)
        draw.line([a, b], fill=(55, 58, 55, 125), width=1)
        for point in (a, b):
            draw.ellipse(
                [point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4],
                outline=(55, 58, 55, 130),
                width=1,
            )
        mx = (a[0] + b[0]) / 2
        my = (a[1] + b[1]) / 2
        width = draw.textlength(label, font=font)
        draw.text((mx - width / 2, my - 30), label, fill=(55, 58, 55, 180), font=font)

    renderer.post_draw.append(draw)


def _add_grid_bubble(renderer: IsoRenderer, label: str, point: Point3D) -> None:
    def draw(draw: ImageDraw.ImageDraw, project: Callable[[Point3D], tuple[float, float]]) -> None:
        font = _font(21)
        x, y = project(point)
        draw.ellipse([x - 22, y - 22, x + 22, y + 22], outline=(55, 58, 55, 130), width=1)
        width = draw.textlength(label, font=font)
        draw.text((x - width / 2, y - 13), label, fill=(55, 58, 55, 185), font=font)

    renderer.post_draw.append(draw)


def _add_light(renderer: IsoRenderer, x: float, z: float, width: float = 1.35) -> None:
    renderer.add_box(
        x - width / 2,
        z - 0.09,
        x + width / 2,
        z + 0.09,
        0.05,
        0.11,
        side=(230, 232, 226, 255),
        top=(255, 255, 245, 255),
        outline=(210, 214, 205, 180),
        layer=4,
    )


def _add_floor_vent(renderer: IsoRenderer, x: float, z: float) -> None:
    def draw(draw: ImageDraw.ImageDraw, project: Callable[[Point3D], tuple[float, float]]) -> None:
        center = np.array(project((x, 0.05, z)))
        u = np.array(project((x + 0.34, 0.05, z))) - center
        v = np.array(project((x, 0.05, z + 0.34))) - center
        for scale in (1.0, 0.72, 0.45):
            pts = [
                tuple(center - u * scale - v * scale),
                tuple(center + u * scale - v * scale),
                tuple(center + u * scale + v * scale),
                tuple(center - u * scale + v * scale),
            ]
            draw.line(pts + [pts[0]], fill=(72, 76, 72, 165), width=2)

    renderer.post_draw.append(draw)


def _add_cabinet_run(renderer: IsoRenderer, x0: float, z0: float, x1: float, z1: float) -> None:
    renderer.add_box(
        x0,
        z0,
        x1,
        z1,
        0.0,
        0.75,
        side=(136, 105, 72, 255),
        top=(154, 124, 86, 255),
        outline=(82, 70, 58, 150),
        layer=5,
    )


def _add_simple_toilet(renderer: IsoRenderer, x: float, z: float) -> None:
    renderer.add_box(
        x - 0.18,
        z - 0.28,
        x + 0.18,
        z + 0.24,
        0.0,
        0.28,
        side=(218, 221, 216, 255),
        top=(245, 246, 242, 255),
        outline=(122, 126, 120, 120),
        layer=5,
    )
    renderer.add_box(
        x - 0.22,
        z + 0.22,
        x + 0.22,
        z + 0.48,
        0.0,
        0.42,
        side=(218, 221, 216, 255),
        top=(245, 246, 242, 255),
        outline=(122, 126, 120, 120),
        layer=5,
    )


def render_reference_grade_demo(path: Path) -> dict[str, int | str]:
    renderer = IsoRenderer(width=1850, height=1120, yaw_deg=-36, elevation_deg=50)
    renderer.add_floor(0, 0, 18, 10.5)
    _add_floor_grid(renderer, x0=0, z0=0, x1=18, z1=10.5)

    glass_side = (155, 166, 162, 82)
    interior_side = (184, 188, 182, 118)
    cap = (31, 35, 32, 255)
    light_cap = (82, 88, 82, 235)

    # Exterior and primary partitions.
    for start, end in [
        ((0, 0), (18, 0)),
        ((18, 0), (18, 10.5)),
        ((18, 10.5), (0, 10.5)),
        ((0, 10.5), (0, 0)),
    ]:
        renderer.add_wall(*start, *end, thickness=0.22, height=2.75, side=glass_side, cap=cap)
    for start, end in [
        ((4.6, 0), (4.6, 10.5)),
        ((8.4, 5.6), (8.4, 10.5)),
        ((12.2, 5.6), (12.2, 10.5)),
        ((14.2, 0), (14.2, 10.5)),
        ((4.6, 5.6), (14.2, 5.6)),
        ((8.4, 8.0), (14.2, 8.0)),
        ((10.3, 8.0), (10.3, 10.5)),
        ((12.2, 8.0), (12.2, 10.5)),
    ]:
        renderer.add_wall(
            *start, *end, thickness=0.14, height=2.55, side=interior_side, cap=light_cap
        )

    # Curtain wall mullions and door frames.
    for x in np.linspace(0.6, 17.4, 13):
        renderer.add_box(
            x - 0.035,
            -0.02,
            x + 0.035,
            0.25,
            0,
            2.45,
            side=(48, 54, 52, 220),
            top=(39, 43, 40, 255),
            layer=18,
        )
    for x, z in [
        (3.8, 0.15),
        (6.2, 5.65),
        (8.9, 5.65),
        (12.7, 5.65),
        (14.25, 4.2),
    ]:
        renderer.add_wall(
            x,
            z,
            x + 0.9,
            z,
            thickness=0.05,
            height=2.2,
            side=(110, 84, 58, 220),
            cap=(83, 64, 45, 255),
            layer=20,
        )

    # Lightweight fixtures.
    for x, z in [
        (2.2, 2.3),
        (6.5, 2.3),
        (10.8, 2.0),
        (15.8, 2.4),
        (2.1, 7.5),
        (6.6, 8.6),
        (15.9, 7.8),
    ]:
        _add_light(renderer, x, z)
    for x, z in [(2.6, 3.7), (6.8, 4.4), (10.2, 3.4), (15.4, 3.5), (2.4, 8.2), (15.8, 8.2)]:
        _add_floor_vent(renderer, x, z)
    _add_cabinet_run(renderer, 8.65, 9.85, 12.0, 10.25)
    _add_cabinet_run(renderer, 8.65, 8.25, 9.15, 9.7)
    for x, z in [(10.9, 9.1), (12.8, 9.1), (10.9, 7.1), (12.8, 7.1)]:
        _add_simple_toilet(renderer, x, z)

    # Labels, pins, and dimensions.
    _add_label(renderer, "LOBBY", "100", 2.4, 2.0)
    _add_label(renderer, "OPEN OFFICE", "101", 9.4, 3.2)
    _add_label(renderer, "BREAK ROOM", "103", 10.1, 9.15)
    _add_tag(renderer, "ISSUE-01", 1.4, 4.2, (206, 54, 42))
    _add_tag(renderer, "RFI-01", 1.0, 8.8, (54, 114, 191))
    _add_tag(renderer, "CLASH-01", 7.0, 0.35, (226, 177, 33))
    _add_tag(renderer, "RFI-02", 17.4, 7.8, (54, 114, 191), side="left")
    _add_tag(renderer, "ISSUE-02", 17.3, 6.3, (206, 54, 42), side="left")
    _add_dimension(renderer, (-0.65, 0.02, 0), (-0.65, 0.02, 3.3), "23'-0\"")
    _add_dimension(renderer, (-0.9, 0.02, 3.3), (-0.9, 0.02, 6.4), "46'-0\"")
    _add_dimension(renderer, (0, 0.02, -0.75), (5.8, 0.02, -0.75), "24'-0\"")
    _add_dimension(renderer, (5.8, 0.02, -0.75), (11.2, 0.02, -0.75), "12'-0\"")
    _add_dimension(renderer, (11.2, 0.02, -0.75), (18, 0.02, -0.75), "18'-0\"")
    for label, point in [
        ("A", (-1.0, 0.02, 8.7)),
        ("B", (-1.0, 0.02, 5.5)),
        ("C", (-1.0, 0.02, 1.2)),
        ("2", (5.8, 0.02, -1.1)),
        ("4", (11.2, 0.02, -1.1)),
    ]:
        _add_grid_bubble(renderer, label, point)

    renderer.render(path)
    return {
        "path": str(path),
        "surface_count": len(renderer.surfaces),
        "width": path.stat().st_size,
    }


if __name__ == "__main__":
    output = Path("docs/plan2field3d_house_conversion/reference_grade_lightweight_preview.png")
    print(render_reference_grade_demo(output))
