"""Multi-view software rendering for the optional CNN part-family classifier.

Rendering is deliberately dependency-light: orthographic projection + a painter's
algorithm with Pillow's C polygon rasterizer. No VTK, OpenGL, xvfb or display is
required, and the output is deterministic for a given mesh/view configuration.

Each part is rendered from six orthographic views (±X, ±Y, ±Z) into grayscale
images. The CNN consumes them as six input channels.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from breplens.schema import MeshData

VIEW_NAMES: tuple[str, ...] = ("right", "left", "front", "back", "top", "bottom")
VIEW_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "right": (1.0, 0.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
}
LIGHT = np.asarray([0.4, -0.7, 0.8], dtype=np.float64)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _basis(direction: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = _normalize(np.asarray(direction, dtype=np.float64))
    up = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(d, up))) > 0.9:
        up = np.asarray([0.0, 1.0, 0.0])
    right = _normalize(np.cross(up, d))
    up2 = _normalize(np.cross(d, right))
    return d, right, up2


def render_views(
    mesh: MeshData,
    *,
    size: int = 96,
    view_names: Sequence[str] = VIEW_NAMES,
    supersample: int = 2,
) -> np.ndarray:
    """Render grayscale views with shape ``(len(view_names), size, size)``."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.size == 0 or triangles.size == 0:
        return np.zeros((len(view_names), size, size), dtype=np.uint8)
    # Center and scale the part once so views share a common world size.
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = (lo + hi) * 0.5
    scale = float(np.max(hi - lo))
    if scale <= 1e-9:
        scale = 1.0
    verts = (vertices - center) / scale * 2.0  # roughly [-1, 1]

    # Flat triangle normals in the normalized world.
    a = verts[triangles[:, 0]]
    b = verts[triangles[:, 1]]
    c = verts[triangles[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-10
    a, b, c = a[valid], b[valid], c[valid]
    normals = normals[valid] / lengths[valid, None]
    light = _normalize(LIGHT)
    shade = 0.22 + 0.78 * np.abs(normals @ light)
    shade = np.clip(shade, 0.0, 1.0)

    big = size * supersample
    images: list[np.ndarray] = []
    for name in view_names:
        d, right, up2 = _basis(VIEW_DIRECTIONS[name])
        x = verts @ right
        y = verts @ up2
        depth = verts @ d
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        span = max(xmax - xmin, ymax - ymin, 1e-9)
        pad = 0.06 * span
        xmin, xmax = xmin - pad, xmax + pad
        ymin, ymax = ymin - pad, ymax + pad
        sx = (big - 1) / (xmax - xmin)
        sy = (big - 1) / (ymax - ymin)
        s = min(sx, sy)
        # Center the fitted projection in the canvas.
        cx = (big - 1) / 2 - 0.5 * (xmin + xmax) * s
        cy = (big - 1) / 2 + 0.5 * (ymin + ymax) * s

        px = x * s + cx
        py = -y * s + cy  # image y grows downward
        tri_depth = (depth[triangles[:, 0]] + depth[triangles[:, 1]] + depth[triangles[:, 2]]) / 3.0
        order = np.argsort(tri_depth)  # far first
        img = Image.new("L", (big, big), color=0)
        draw = ImageDraw.Draw(img)
        for idx in order:
            tri = triangles[idx]
            if not valid[idx]:
                continue
            fill = round(255.0 * shade[idx])
            pts = [
                (float(px[tri[0]]), float(py[tri[0]])),
                (float(px[tri[1]]), float(py[tri[1]])),
                (float(px[tri[2]]), float(py[tri[2]])),
            ]
            draw.polygon(pts, fill=fill)
        if supersample != 1:
            img = img.resize((size, size), Image.Resampling.LANCZOS)
        images.append(np.asarray(img, dtype=np.uint8))
    return np.stack(images, axis=0)


def render_scene_png(mesh: MeshData, path: str, *, size: int = 512) -> None:
    """Save a filmstrip of the three views (documentation helper)."""
    views = render_views(mesh, size=size, supersample=2)
    strip = Image.new("L", (size * len(views), size))
    for i, view in enumerate(views):
        strip.paste(Image.fromarray(view), (i * size, 0))
    strip.save(path)


__all__ = ["VIEW_DIRECTIONS", "VIEW_NAMES", "render_scene_png", "render_views"]
