"""Small geometry helpers owned by this project (not copied from upstream)."""

from __future__ import annotations


def iou(a, b) -> float:
    """Intersection over union of two normalized ``[x, y, w, h]`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def area(box) -> float:
    return float(box[2] * box[3])


def touches_edge(box, margin: float) -> bool:
    x, y, w, h = box
    return x <= margin or y <= margin or x + w >= 1 - margin or y + h >= 1 - margin
