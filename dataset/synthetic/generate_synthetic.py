import csv
import random
import os

from .utils.config import (
    ORIGIN_MODE,
    NUM_RANDOM_MAPS, VISIBLE_STATIONS_PER_MAP, HIDDEN_STATIONS_PER_MAP, CSV_PATH, GRID_OUTPUT_DIR,
    MASK_PROBABILITY,
    OBSTACLE_DENSITY,
    REGULAR_OBSTACLE_SIZE_RANGE,
    NUM_SMALL_OBSTACLES, SMALL_OBSTACLE_SIZE_RANGE,
    NUM_NARROW_OBSTACLES, NARROW_OBSTACLE_WIDTH_RANGE, NARROW_OBSTACLE_HEIGHT_RANGE,
    DRAW, FRAME_INTERVAL, MIN_POINTS,
    GRID_SIZE, GRID_RESOLUTION,
    ORIG_GRID_WIDTH_RANGE, ORIG_GRID_HEIGHT_RANGE,
    X_MIN_RANGE, Y_MIN_RANGE,
    FIXED_MAP_ORIGIN_X, FIXED_MAP_ORIGIN_Y,
    STATION_OFFSET_DISTANCE
)

from .utils.goals import spawn_goals_near_obstacles, spawn_goals, filter_far_points
from .utils.plotting import setup_plot, finalize_plot
from .utils.environment import build_random_environment
from .utils.grid import Grid, NoFreeCellsError
from .utils.trajectories import generate_paths_and_log
from .utils.occupancy_grid import generate_occupancy_grid


def calculate_num_regular_obstacles(workspace_area, density=OBSTACLE_DENSITY):
    import numpy as np
    return int(np.sqrt(workspace_area) * density)

def calculate_workspace_bounds(map_origin_x, map_origin_y, workspace_width_px, workspace_height_px,
                               grid_size=GRID_SIZE, grid_resolution=GRID_RESOLUTION):
    pad_x_pixels = grid_size - workspace_width_px
    pad_y_pixels = grid_size - workspace_height_px
    pad_left = pad_x_pixels // 2
    pad_bottom = pad_y_pixels // 2

    x_min = map_origin_x + pad_left * grid_resolution
    y_min = map_origin_y + pad_bottom * grid_resolution

    workspace_width_m = workspace_width_px * grid_resolution
    workspace_height_m = workspace_height_px * grid_resolution

    x_max = x_min + workspace_width_m
    y_max = y_min + workspace_height_m

    return {
        'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
        'width': workspace_width_m, 'height': workspace_height_m,
        'area': workspace_width_m * workspace_height_m,
        'orig_width': workspace_width_px, 'orig_height': workspace_height_px,
        'pad_x_pixels': pad_x_pixels, 'pad_y_pixels': pad_y_pixels
    }

def main():
    traj_id_counter = 0
    grid_id_counter = 0

    visible_n = VISIBLE_STATIONS_PER_MAP
    hidden_n = HIDDEN_STATIONS_PER_MAP

    print("=== FULLY RANDOM MODE ===")
    print(f"Generating {NUM_RANDOM_MAPS} random maps")
    print(f"Visible stations/map: {visible_n} (written to CSV)")
    print(f"Hidden stations/map:  {hidden_n} (NOT written to CSV; used for start/goal)")

    csv_header = [
        "task_id", "grid_id", "traj_id", "frame_id",
        "agent_id", "agent_type", "x", "y", "z", "pos_mask",
        "stations_pos/mask", "robot_pos/mask",
    ]
    for i in range(visible_n):
        csv_header.append(f"station_{i}_x")
        csv_header.append(f"station_{i}_y")

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

        for map_num in range(NUM_RANDOM_MAPS):
            workspace_width_px = random.randint(*ORIG_GRID_WIDTH_RANGE)
            workspace_height_px = random.randint(*ORIG_GRID_HEIGHT_RANGE)

            if ORIGIN_MODE == "fixed":
                map_origin_x = FIXED_MAP_ORIGIN_X
                map_origin_y = FIXED_MAP_ORIGIN_Y
            elif ORIGIN_MODE == "random":
                map_origin_x = random.uniform(*X_MIN_RANGE)
                map_origin_y = random.uniform(*Y_MIN_RANGE)
            else:
                raise ValueError("Invalid ORIGIN_MODE")

            workspace = calculate_workspace_bounds(
                map_origin_x, map_origin_y,
                workspace_width_px, workspace_height_px
            )
            x_min, x_max = workspace["x_min"], workspace["x_max"]
            y_min, y_max = workspace["y_min"], workspace["y_max"]
            workspace_area = workspace["area"]

            num_regular_obstacles = calculate_num_regular_obstacles(workspace_area)
            total_obstacles = num_regular_obstacles + NUM_SMALL_OBSTACLES + NUM_NARROW_OBSTACLES

            print(f"\n[Map {map_num + 1}] Area: {workspace_area:.2f}m², Obs: {total_obstacles}")

            ax = setup_plot(f"Map {map_num}", x_min, x_max, y_min, y_max) if DRAW else None

            for retry in range(10):
                try:
                    obstacle_config = {
                        "regular": {"count": num_regular_obstacles, "size_range": REGULAR_OBSTACLE_SIZE_RANGE},
                        "small": {"count": NUM_SMALL_OBSTACLES, "size_range": SMALL_OBSTACLE_SIZE_RANGE, "pad": 0.1},
                        "narrow": {
                            "count": NUM_NARROW_OBSTACLES,
                            "width_range": NARROW_OBSTACLE_WIDTH_RANGE,
                            "height_range": NARROW_OBSTACLE_HEIGHT_RANGE,
                            "pad": 0.15
                        }
                    }

                    pads, (agv_c, mob_c), objects_no_pad = build_random_environment(
                        ax, x_min, x_max, y_min, y_max, total_obstacles,
                        obstacle_config=obstacle_config, draw=DRAW
                    )

                    grid = Grid((x_min, x_max), (y_min, y_max), resolution=0.2)

                    visible_stations = spawn_goals_near_obstacles(
                        visible_n, grid, pads,
                        ((x_min, x_max), (y_min, y_max)),
                        offset_dist=STATION_OFFSET_DISTANCE
                    )

                    hidden_stations = []
                    if hidden_n > 0:
                        hidden_stations = spawn_goals(
                            hidden_n, grid, pads,
                            (x_min, x_max), (y_min, y_max),
                            min_distance=grid.resolution * 5
                        )
                        hidden_stations = filter_far_points(hidden_stations, visible_stations, min_dist=1.0)

                        safety_iters = 0
                        while len(hidden_stations) < hidden_n and safety_iters < 20:
                            safety_iters += 1
                            needed = hidden_n - len(hidden_stations)
                            extra = spawn_goals(
                                needed, grid, pads,
                                (x_min, x_max), (y_min, y_max),
                                min_distance=grid.resolution * 5
                            )
                            extra = filter_far_points(extra, visible_stations + hidden_stations, min_dist=1.0)
                            hidden_stations.extend(extra)

                        hidden_stations = hidden_stations[:hidden_n]

                    all_stations_for_paths = list(visible_stations) + list(hidden_stations)

                    station_coords_flat = []
                    for (sx, sy) in visible_stations:
                        station_coords_flat.extend([sx, sy])

                    while len(station_coords_flat) < visible_n * 2:
                        station_coords_flat.extend([0.0, 0.0])

                    task_id = f"random_{map_num}"
                    grid_id = f"random_{grid_id_counter}"

                    generate_occupancy_grid(
                        objects_no_pad, grid_id,
                        grid_size=GRID_SIZE, resolution=GRID_RESOLUTION,
                        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                        map_origin_x=map_origin_x, map_origin_y=map_origin_y,
                        output_dir=GRID_OUTPUT_DIR
                    )
                    grid_id_counter += 1

                    if DRAW and ax:
                        if visible_stations:
                            ax.plot(*zip(*visible_stations), "r*", ms=8, label="visible stations")
                        if hidden_stations:
                            ax.plot(*zip(*hidden_stations), "kx", ms=6, label="hidden stations")

                    traj_id_counter = generate_paths_and_log(
                        ax, all_stations_for_paths, grid, pads, writer,
                        task_id, grid_id, mob_c, agv_c,
                        station_coords_flat,
                        traj_id_counter, FRAME_INTERVAL, MIN_POINTS, DRAW,
                        mask_prob=MASK_PROBABILITY, n_trajectories=10
                    )

                    break

                except NoFreeCellsError:
                    print(f"  Retry {retry + 1}...")
                    if DRAW and ax:
                        ax.cla()

            if DRAW and ax:
                finalize_plot(ax)

    print(f"\n✓ Generated {NUM_RANDOM_MAPS} maps, {traj_id_counter} trajectories")

if __name__ == '__main__':
    main()