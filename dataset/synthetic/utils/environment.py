import random
import numpy as np

from .geometry import create_padded
from .config import MOBILE_WIDTH, MOBILE_HEIGHT, AGV_WIDTH, AGV_HEIGHT

def build_random_environment(ax, x_min, x_max, y_min, y_max, num_obstacles=5,
                            obstacle_config=None, draw=True):
    """
    Build random environment with obstacles at random positions.

    Returns:
      pads: list of padded collision_data dicts
      centers: (agv_center, mobile_center)
      objects_no_pad: list of original (no pad) collision dicts
    """
    pads = []
    objects_no_pad = []

    # Smaller workspace margins to allow objects near edges
    margin = 0.0
    work_x_min, work_x_max = x_min + margin, x_max - margin
    work_y_min, work_y_max = y_min + margin, y_max - margin

    color_configs = [
        {"color": "lightgrey", "edge": "black"},
        {"color": "wheat", "edge": "brown"},
        {"color": "lightblue", "edge": "blue"},
        {"color": "lightgreen", "edge": "green"},
        {"color": "lightcoral", "edge": "red"},
        {"color": "lightyellow", "edge": "orange"},
        {"color": "lavender", "edge": "purple"},
        {"color": "peachpuff", "edge": "darkorange"},
    ]

    placed_centers = []
    min_spacing = 3.0

    def cp(ax_, *args, **kwargs):
        return create_padded(ax_, *args, draw=draw, **kwargs)

    def place_obstacle(width_range, height_range, pad_value=None, attempts=50):
        for attempt in range(attempts):
            center_x = random.uniform(work_x_min, work_x_max)
            center_y = random.uniform(work_y_min, work_y_max)

            too_close = False
            for px, py in placed_centers:
                dist = np.sqrt((center_x - px)**2 + (center_y - py)**2)
                if dist < min_spacing:
                    too_close = True
                    break

            if not too_close or attempt == attempts - 1:
                width = random.uniform(width_range[0], width_range[1])
                height = random.uniform(height_range[0], height_range[1])
                rotation = random.uniform(-45, 45)
                colors = random.choice(color_configs)

                if pad_value is not None:
                    obstacle = create_padded(
                        ax, (center_x, center_y), width, height,
                        color=colors["color"], edgecolor=colors["edge"],
                        jitter_x=(0, 0), jitter_y=(0, 0),
                        rotation_deg=rotation, pad=pad_value, draw=draw
                    )
                else:
                    obstacle = cp(
                        ax, (center_x, center_y), width, height,
                        color=colors["color"], edgecolor=colors["edge"],
                        jitter_x=(0, 0), jitter_y=(0, 0),
                        rotation_deg=rotation
                    )

                pads.append(obstacle["collision_data"])
                objects_no_pad.append(obstacle["original_collision_data"])
                placed_centers.append((center_x, center_y))
                return True
        return False

    if obstacle_config is not None:
        for group_name, group_params in obstacle_config.items():
            count = group_params['count']
            pad_value = group_params.get('pad', None)

            if 'width_range' in group_params and 'height_range' in group_params:
                width_range = group_params['width_range']
                height_range = group_params['height_range']
            elif 'size_range' in group_params:
                width_range = group_params['size_range']
                height_range = group_params['size_range']
            else:
                raise ValueError(
                    f"Obstacle group '{group_name}' must specify either "
                    "'size_range' or both 'width_range' and 'height_range'"
                )

            for i in range(count):
                placed = place_obstacle(width_range, height_range, pad_value=pad_value)
                if not placed:
                    print(f"  Warning: Could not place {group_name} obstacle {i+1}/{count}")
    else:
        default_size_range = (0.7, 2.0)
        for i in range(num_obstacles):
            placed = place_obstacle(default_size_range, default_size_range)
            if not placed:
                print(f"  Warning: Could not place obstacle {i+1}/{num_obstacles}")

    # Random AGV
    for attempt in range(50):
        agv_x = random.uniform(work_x_min, work_x_max)
        agv_y = random.uniform(work_y_min, work_y_max)
        too_close = any(
            np.sqrt((agv_x - px)**2 + (agv_y - py)**2) < min_spacing
            for px, py in placed_centers
        )
        if not too_close or attempt == 49:
            break

    agv = cp(
        ax, (agv_x, agv_y), AGV_WIDTH, AGV_HEIGHT,
        color="lightcoral", edgecolor="red",
        jitter_x=(0, 0), jitter_y=(0, 0),
        rotation_deg=random.uniform(-30, 30)
    )
    pads.append(agv["collision_data"])
    objects_no_pad.append(agv["original_collision_data"])
    placed_centers.append((agv_x, agv_y))

    # Random Mobile
    for attempt in range(50):
        mobile_x = random.uniform(work_x_min, work_x_max)
        mobile_y = random.uniform(work_y_min, work_y_max)
        too_close = any(
            np.sqrt((mobile_x - px)**2 + (mobile_y - py)**2) < min_spacing
            for px, py in placed_centers
        )
        if not too_close or attempt == 49:
            break

    mobile = cp(
        ax, (mobile_x, mobile_y), MOBILE_WIDTH, MOBILE_HEIGHT,
        color="plum", edgecolor="purple",
        jitter_x=(0, 0), jitter_y=(0, 0),
        rotation_deg=random.uniform(-30, 30)
    )
    pads.append(mobile["collision_data"])
    objects_no_pad.append(mobile["original_collision_data"])

    centers = (agv["center"], mobile["center"])
    return pads, centers, objects_no_pad
