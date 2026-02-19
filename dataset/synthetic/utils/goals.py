import math
import random
import numpy as np

from .grid import NoFreeCellsError
from .geometry import rotate_point

def spawn_goals(
    n, grid, pads,
    x_bounds, y_bounds,
    *, rng=None,
    max_attempts=10000,
    loosen_factor=1.5,
    min_distance=None
):
    if n <= 0:
        return []
    if rng is None:
        rng = np.random.default_rng()

    if min_distance is None:
        min_distance = grid.resolution * 5

    i_min = max(0, int((x_bounds[0] - grid.x_min) / grid.resolution))
    i_max = min(grid.cols, int(math.ceil((x_bounds[1] - grid.x_min) / grid.resolution)))
    j_min = max(0, int((y_bounds[0] - grid.y_min) / grid.resolution))
    j_max = min(grid.rows, int(math.ceil((y_bounds[1] - grid.y_min) / grid.resolution)))

    free = 0
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            if grid.passable((i, j), pads):
                free += 1
    if free == 0:
        raise NoFreeCellsError(f"No free cells in region x={x_bounds}, y={y_bounds}")

    goals = []
    attempts = 0
    attempts_since_loosen = 0
    total_cap = max_attempts * 10
    cur_xb, cur_yb = x_bounds, y_bounds
    cur_min_dist = min_distance

    while len(goals) < n:
        if attempts_since_loosen >= max_attempts:
            if cur_min_dist > grid.resolution * 1.5:
                cur_min_dist *= 0.75
                print(f"Reducing minimum distance to {cur_min_dist:.3f}")
                attempts_since_loosen = 0
            else:
                cx = (cur_xb[0] + cur_xb[1]) / 2
                cy = (cur_yb[0] + cur_yb[1]) / 2
                half_w = (cur_xb[1] - cur_xb[0]) * loosen_factor / 2
                half_h = (cur_yb[1] - cur_yb[0]) * loosen_factor / 2
                cur_xb = (cx - half_w, cx + half_w)
                cur_yb = (cy - half_h, cy + half_h)
                print(f"Loosening bounds to x={cur_xb}, y={cur_yb}")
                attempts_since_loosen = 0

        gx = rng.uniform(*cur_xb)
        gy = rng.uniform(*cur_yb)
        idx = grid.to_idx((gx, gy))

        if grid.in_bounds(idx) and grid.passable(idx, pads):
            too_close = any((gx - ex)**2 + (gy - ey)**2 < cur_min_dist**2 for ex, ey in goals)
            if not too_close:
                goals.append((gx, gy))
                attempts_since_loosen = 0
            else:
                attempts_since_loosen += 1
        else:
            attempts_since_loosen += 1

        attempts += 1
        if attempts > total_cap:
            raise RuntimeError(
                f"Unable to place {n} goals after {attempts} attempts "
                f"(initial x={x_bounds}, y={y_bounds})."
            )
    return goals

# ------------------------------
# Dict-based collision for pads
# ------------------------------

def check_collision_with_pads(px, py, pads):
    """
    Checks if point is inside any obstacle.
    Compatible with:
      - rectangles: uses original_center + width/height + cumulative_rotation
      - circles: uses center + radius (preferred) OR width/2 fallback
    """
    for pad in pads:
        shape_type = pad.get('type', 'rectangle')

        if shape_type == 'rectangle':
            cx, cy = pad['original_center']
            rot = pad.get('cumulative_rotation', 0.0)
            w = pad['width']
            h = pad['height']

            tx, ty = px - cx, py - cy
            rad = math.radians(-rot)
            cos_a, sin_a = math.cos(rad), math.sin(rad)

            rx = tx * cos_a - ty * sin_a
            ry = tx * sin_a + ty * cos_a

            if (-w/2 <= rx <= w/2) and (-h/2 <= ry <= h/2):
                return True

        elif shape_type == 'circle':
            # preferred
            cx, cy = pad.get('center', pad.get('original_center'))
            r = pad.get('radius', None)
            if r is None:
                r = pad.get('width', 0.0) / 2.0
            dist_sq = (px - cx)**2 + (py - cy)**2
            if dist_sq <= r**2:
                return True

    return False

def spawn_goals_near_obstacles(num_goals, grid, pads, bounds, offset_dist):
    goals = []
    attempts = 0
    max_attempts = num_goals * 50
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    if not pads:
        return spawn_goals(num_goals, grid, [], bounds[0], bounds[1])

    while len(goals) < num_goals and attempts < max_attempts:
        attempts += 1

        pad = random.choice(pads)
        shape_type = pad.get('type', 'rectangle')

        gx, gy = 0.0, 0.0

        if shape_type == 'rectangle':
            cx, cy = pad['original_center']
            rot = pad.get('cumulative_rotation', 0.0)
            w_pad = pad['width']
            h_pad = pad['height']

            side = random.randint(0, 3)
            if side == 0:      # top
                local_x = random.uniform(-w_pad/2, w_pad/2)
                local_y = h_pad/2 + offset_dist
            elif side == 1:    # right
                local_x = w_pad/2 + offset_dist
                local_y = random.uniform(-h_pad/2, h_pad/2)
            elif side == 2:    # bottom
                local_x = random.uniform(-w_pad/2, w_pad/2)
                local_y = -(h_pad/2 + offset_dist)
            else:              # left
                local_x = -(w_pad/2 + offset_dist)
                local_y = random.uniform(-h_pad/2, h_pad/2)

            rx, ry = rotate_point(local_x, local_y, rot)
            gx = cx + rx
            gy = cy + ry

        elif shape_type == 'circle':
            cx, cy = pad.get('center', pad.get('original_center'))
            r = pad.get('radius', None)
            if r is None:
                r = pad.get('width', 0.0) / 2.0

            angle = random.uniform(0, 2 * math.pi)
            dist = r + offset_dist
            gx = cx + dist * math.cos(angle)
            gy = cy + dist * math.sin(angle)

        # bounds check
        if not (x_min <= gx <= x_max and y_min <= gy <= y_max):
            continue

        # not inside obstacles
        if check_collision_with_pads(gx, gy, pads):
            continue

        # spacing between goals
        if any(np.hypot(gx - ox, gy - oy) < 1.0 for (ox, oy) in goals):
            continue

        goals.append((gx, gy))

    if len(goals) < num_goals:
        print(f"    Warning: Could only place {len(goals)}/{num_goals} goals near obstacles. Filling randomly.")
        remaining = num_goals - len(goals)
        random_goals = spawn_goals(remaining, grid, pads, bounds[0], bounds[1])
        goals.extend(random_goals)

    return goals


def filter_far_points(points, avoid_points, min_dist=1.0):
    """Keep only points that are at least min_dist away from all avoid_points."""
    if not avoid_points:
        return list(points)
    min_dist_sq = min_dist * min_dist
    out = []
    for px, py in points:
        ok = True
        for ax, ay in avoid_points:
            if (px - ax) ** 2 + (py - ay) ** 2 < min_dist_sq:
                ok = False
                break
        if ok:
            out.append((px, py))
    return out
