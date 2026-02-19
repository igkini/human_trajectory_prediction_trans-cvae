import math
import heapq
import numpy as np

from .geometry import line_of_sight

def theta_star(start, goal, grid, pads, los_step=0.05):
    s_idx, g_idx = grid.to_idx(start), grid.to_idx(goal)
    if not grid.passable(s_idx, pads) or not grid.passable(g_idx, pads):
        return []

    dirs = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    g = {s_idx: 0.0}
    parent = {s_idx: s_idx}
    los_edge = {}
    h0 = math.hypot(*(np.subtract(grid.to_coord(s_idx), grid.to_coord(g_idx))))
    open_set = [(h0, s_idx)]

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == g_idx:
            idx_path = []
            i = cur
            while i != parent[i]:
                idx_path.append(i)
                i = parent[i]
            idx_path.append(s_idx)
            idx_path.reverse()

            dense = [grid.to_coord(idx_path[0])]
            for u, v in zip(idx_path, idx_path[1:]):
                a = grid.to_coord(u)
                b = grid.to_coord(v)
                if los_edge.get(v, False):
                    dist = math.hypot(b[0]-a[0], b[1]-a[1])
                    n = max(1, int(dist/los_step))
                    for t in [i/n for i in range(1, n+1)]:
                        dense.append((a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1])))
                else:
                    dense.append(b)
            return dense

        for dx, dy in dirs:
            nb = (cur[0] + dx, cur[1] + dy)
            if not grid.in_bounds(nb) or not grid.passable(nb, pads):
                continue

            parent_coord = grid.to_coord(parent[cur])
            nb_coord = grid.to_coord(nb)
            cur_coord = grid.to_coord(cur)

            if line_of_sight(parent_coord, nb_coord, pads, grid.resolution/2):
                new_g = g[parent[cur]] + math.hypot(*(np.subtract(parent_coord, nb_coord)))
                used_los = True
                candidate_parent = parent[cur]
            else:
                new_g = g[cur] + math.hypot(*(np.subtract(cur_coord, nb_coord)))
                used_los = False
                candidate_parent = cur

            if nb not in g or new_g < g[nb]:
                g[nb] = new_g
                parent[nb] = candidate_parent
                los_edge[nb] = used_los
                f = new_g + math.hypot(*(np.subtract(nb_coord, grid.to_coord(g_idx))))
                heapq.heappush(open_set, (f, nb))

    return []

def smooth_path_with_beziers(path, radius=0.2, resolution=20):
    import numpy as np

    if len(path) < 3:
        return path

    smoothed_path = []
    last_point = np.array(path[0])
    smoothed_path.append(tuple(last_point))

    for i in range(1, len(path) - 1):
        p_prev = np.array(path[i-1])
        p_turn = np.array(path[i])
        p_next = np.array(path[i+1])

        v_prev = p_prev - p_turn
        v_next = p_next - p_turn

        dist_prev = np.linalg.norm(v_prev)
        dist_next = np.linalg.norm(v_next)

        if dist_prev < 1e-9 or dist_next < 1e-9:
            smoothed_path.append(tuple(p_turn))
            continue

        effective_radius = min(radius, dist_prev / 2, dist_next / 2)

        v_prev_norm = v_prev / dist_prev
        v_next_norm = v_next / dist_next

        arc_start = p_turn + v_prev_norm * effective_radius
        arc_end = p_turn + v_next_norm * effective_radius

        smoothed_path.append(tuple(arc_start))

        t_values = np.linspace(0, 1, resolution)
        for t in t_values:
            point = (1-t)**2 * arc_start + 2*t*(1-t) * p_turn + t**2 * arc_end
            smoothed_path.append(tuple(point))

        last_point = arc_end

    smoothed_path.append(path[-1])
    return smoothed_path
