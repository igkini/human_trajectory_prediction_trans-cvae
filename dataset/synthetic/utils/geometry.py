import math
import random
import numpy as np
import matplotlib.patches as patches
import matplotlib.transforms as mtrans

# ---------------------------
# SHAPES WITH PADDING (DRAW?)
# ---------------------------

def create_padded(
    ax,
    position,
    width,
    height,
    color,
    edgecolor,
    *,
    pad=0.2,
    rounding=0.15,
    zorder=1,
    rotation_deg=0,
    jitter_x=(0, 0),
    jitter_y=(0, 0),
    draw=True,
    shape='rectangle'
):
    """
    Draw OR just compute a padded shape (rectangle or circle) with optional jitter and rotation.

    For circles:
    - 'width' is used as the diameter
    - 'height' is ignored
    - 'rotation_deg' is ignored

    Returns dict with:
      - original_collision_data
      - collision_data (padded)
      - center
    """
    # jitter
    x0 = position[0] + random.uniform(*jitter_x)
    y0 = position[1] + random.uniform(*jitter_y)

    if shape == 'circle':
        return _create_padded_circle(
            ax, x0, y0, width, color, edgecolor,
            pad, zorder, draw
        )
    else:
        return _create_padded_rectangle(
            ax, x0, y0, width, height, color, edgecolor,
            pad, rounding, zorder, rotation_deg, draw
        )

def _create_padded_circle(ax, x0, y0, diameter, color, edgecolor, pad, zorder, draw):
    """
    Circle representation is centered at (cx, cy) with 'radius'.
    We interpret x0,y0 as TOP-LEFT of bounding box for consistency with old code.
    """
    radius = diameter / 2.0
    cx, cy = x0 + radius, y0 + radius

    original_collision_data = {
        'type': 'circle',
        'center': (cx, cy),
        'radius': radius,
        'original_center': (cx, cy),
        'cumulative_translation': (0, 0),
    }

    collision_data = {
        'type': 'circle',
        'center': (cx, cy),
        'radius': radius + pad,
        'original_center': (cx, cy),
        'cumulative_translation': (0, 0),
    }

    if ax is None or not draw:
        return {
            "patch": None,
            "pad": None,
            "outline": None,
            "center": (cx, cy),
            "original_collision_data": original_collision_data,
            "collision_data": collision_data
        }

    patch = patches.Circle((cx, cy), radius,
                           linewidth=1,
                           edgecolor=edgecolor,
                           facecolor=color,
                           alpha=0.8,
                           zorder=zorder)
    ax.add_patch(patch)

    outline = patches.Circle((cx, cy), radius + pad,
                             linewidth=1,
                             edgecolor='black',
                             facecolor='none',
                             zorder=zorder - 0.1)
    ax.add_patch(outline)

    pad_circle = patches.Circle((cx, cy), radius + pad,
                                linewidth=0,
                                facecolor="none",
                                zorder=zorder - 0.1)

    return {
        "patch": patch,
        "pad": pad_circle,
        "outline": outline,
        "center": (cx, cy),
        "original_collision_data": original_collision_data,
        "collision_data": collision_data
    }

def _create_padded_rectangle(ax, x0, y0, width, height, color, edgecolor,
                             pad, rounding, zorder, rotation_deg, draw):
    # center of ORIGINAL rect
    cx, cy = x0 + width/2, y0 + height/2

    # original corners
    orig_corners = [
        (x0, y0),
        (x0 + width, y0),
        (x0 + width, y0 + height),
        (x0, y0 + height)
    ]

    if rotation_deg != 0:
        ang = math.radians(rotation_deg)
        cos_a, sin_a = math.cos(ang), math.sin(ang)

        def rot(pt):
            x, y = pt
            xr = (x - cx) * cos_a - (y - cy) * sin_a + cx
            yr = (x - cx) * sin_a + (y - cy) * cos_a + cy
            return (xr, yr)

        orig_corners = [rot(pt) for pt in orig_corners]

    original_collision_data = {
        'type': 'rectangle',
        'corners': orig_corners,
        'original_center': (cx, cy),
        'width': width,
        'height': height,
        'cumulative_translation': (0, 0),
        'cumulative_rotation': rotation_deg
    }

    # padded rectangle
    pad_w, pad_h = width + 2*pad, height + 2*pad
    pad_x, pad_y = cx - pad_w/2, cy - pad_h/2
    pad_corners = [
        (pad_x, pad_y),
        (pad_x + pad_w, pad_y),
        (pad_x + pad_w, pad_y + pad_h),
        (pad_x, pad_y + pad_h)
    ]

    if rotation_deg != 0:
        ang = math.radians(rotation_deg)
        cos_a, sin_a = math.cos(ang), math.sin(ang)

        def rot(pt):
            x, y = pt
            xr = (x - cx) * cos_a - (y - cy) * sin_a + cx
            yr = (x - cx) * sin_a + (y - cy) * cos_a + cy
            return (xr, yr)

        pad_corners = [rot(pt) for pt in pad_corners]

    collision_data = {
        'type': 'rectangle',
        'corners': pad_corners,
        'original_center': (cx, cy),
        'width': pad_w,
        'height': pad_h,
        'cumulative_translation': (0, 0),
        'cumulative_rotation': rotation_deg
    }

    if ax is None or not draw:
        return {
            "patch": None,
            "pad": None,
            "outline": None,
            "center": (cx, cy),
            "original_collision_data": original_collision_data,
            "collision_data": collision_data
        }

    patch = patches.Rectangle((x0, y0), width, height,
                              linewidth=1,
                              edgecolor=edgecolor,
                              facecolor=color,
                              alpha=0.8,
                              zorder=zorder)

    transform = mtrans.Affine2D().rotate_deg_around(cx, cy, rotation_deg) + ax.transData
    patch.set_transform(transform)
    ax.add_patch(patch)

    outline = patches.FancyBboxPatch((pad_x, pad_y), pad_w, pad_h,
                                    boxstyle=f"round,pad=0.,rounding_size={rounding}",
                                    linewidth=1,
                                    edgecolor='black',
                                    facecolor='none',
                                    transform=transform,
                                    zorder=zorder - 0.1)
    ax.add_patch(outline)

    pad_rect = patches.Rectangle((pad_x, pad_y), pad_w, pad_h,
                                 linewidth=0,
                                 facecolor="none",
                                 transform=transform,
                                 zorder=zorder - 0.1)

    return {
        "patch": patch,
        "pad": pad_rect,
        "outline": outline,
        "center": (cx, cy),
        "original_collision_data": original_collision_data,
        "collision_data": collision_data
    }

# ---------------------------
# COLLISION / GEOMETRY UTILS
# ---------------------------

def point_in_transformed_rectangle_fast(pt, rect_data):
    """Fast point-in-rotated-rectangle test using half-space checks."""
    x, y = pt
    corners = rect_data['corners']

    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if x < min_x or x > max_x or y < min_y or y > max_y:
        return False

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(pt, corners[0], corners[1])
    d2 = sign(pt, corners[1], corners[2])
    d3 = sign(pt, corners[2], corners[3])
    d4 = sign(pt, corners[3], corners[0])

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)
    return not (has_neg and has_pos)

def point_in_patch(pt, patch_data):
    """Collision test for rectangle/circle pad dictionaries."""
    t = patch_data.get('type', 'rectangle')
    if t == 'rectangle':
        return point_in_transformed_rectangle_fast(pt, patch_data)
    if t == 'circle':
        cx, cy = patch_data['center']
        r = patch_data.get('radius', None)
        if r is None:
            # backwards-compatible fallback
            r = patch_data.get('width', 0.0) / 2.0
        return math.hypot(pt[0]-cx, pt[1]-cy) <= r
    return False

def line_of_sight(p1, p2, pads, step):
    dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    n = max(1, int(dist/step))
    for i in range(1, n+1):
        t = i / n
        x = p1[0] + t*(p2[0]-p1[0])
        y = p1[1] + t*(p2[1]-p1[1])
        if any(point_in_patch((x, y), p) for p in pads):
            return False
    return True

def rotate_point(x, y, theta_deg):
    rad = math.radians(theta_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
