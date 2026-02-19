import os
import random
import numpy as np
from shapely.geometry import Polygon, Point
from scipy import ndimage

def ensure_directory_exists(directory):
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def write_pgm(occupancy_grid, filename, output_dir='256'):
    ensure_directory_exists(output_dir)
    filepath = os.path.join(output_dir, filename) if output_dir else filename

    height, width = occupancy_grid.shape
    with open(filepath, 'wb') as f:
        header = f"P5\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))
        f.write(occupancy_grid.tobytes())

    print(f"Saved occupancy grid to {filepath} ({width}x{height} PGM)")

def save_map_yaml(grid_id, resolution, origin_x, origin_y,
                  output_dir='256', grid_dir='256'):
    ensure_directory_exists(output_dir)

    if grid_dir and output_dir:
        grid_path = os.path.relpath(
            os.path.join(grid_dir, f"grid_{grid_id}.pgm"),
            output_dir
        )
    elif grid_dir:
        grid_path = os.path.join(grid_dir, f"grid_{grid_id}.pgm")
    else:
        grid_path = f"grid_{grid_id}.pgm"

    origin_z = 0.0
    yaml_content = f"""image: {grid_path}
mode: trinary
resolution: {resolution:.3f}
origin: [{origin_x:.3f}, {origin_y:.3f}, {origin_z:.3f}]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""

    filename = f"map_{grid_id}.yaml"
    filepath = os.path.join(output_dir, filename) if output_dir else filename

    with open(filepath, 'w') as f:
        f.write(yaml_content)

    print(f"Saved map YAML to {filepath}")
    print(f"  Origin: [{origin_x:.3f}, {origin_y:.3f}, {origin_z:.3f}]")
    print(f"  Image path: {grid_path}")

def generate_occupancy_grid(objects_no_pad, grid_id, grid_size=256, resolution=0.05,
                            wall_thickness=2, x_min=-5.3, x_max=5.4, y_min=-6.3, y_max=6.3,
                            map_origin_x=-6.4, map_origin_y=-6.4, output_dir='256'):
    """
    Generate occupancy grid as a PGM file with proper workspace mapping.
    objects_no_pad: list of ORIGINAL (no pad) object collision dicts (rectangles/circles)
    """
    workspace_width = x_max - x_min
    workspace_height = y_max - y_min

    workspace_width_pixels = int(workspace_width / resolution)
    workspace_height_pixels = int(workspace_height / resolution)

    print(f"Workspace dimensions: {workspace_width:.2f}m x {workspace_height:.2f}m")
    print(f"Workspace in pixels: {workspace_width_pixels} x {workspace_height_pixels}")

    padding_x_total = grid_size - workspace_width_pixels
    padding_y_total = grid_size - workspace_height_pixels

    padding_x_left = padding_x_total // 2
    padding_x_right = padding_x_total - padding_x_left
    padding_y_bottom = padding_y_total // 2
    padding_y_top = padding_y_total - padding_y_bottom

    print(f"Padding: x_left={padding_x_left}, x_right={padding_x_right}, "
          f"y_bottom={padding_y_bottom}, y_top={padding_y_top}")

    full_grid = np.full((grid_size, grid_size), 192, dtype=np.uint8)
    workspace_grid = np.full((workspace_height_pixels, workspace_width_pixels), 255, dtype=np.uint8)

    def env_to_workspace_grid(x, y):
        i = int((x - x_min) / resolution)
        j = int((y - y_min) / resolution)
        i = max(0, min(workspace_width_pixels - 1, i))
        j = max(0, min(workspace_height_pixels - 1, j))
        return i, j

    # walls (black borders)
    workspace_grid[0:wall_thickness, :] = 0
    workspace_grid[-wall_thickness:, :] = 0
    workspace_grid[:, 0:wall_thickness] = 0
    workspace_grid[:, -wall_thickness:] = 0

    obstacle_outline_value = 0
    outline_thickness = 2  # pixels

    for obj in objects_no_pad:
        shape_type = obj.get('type', 'rectangle')

        # This will hold True for occupied interior pixels of this obstacle
        obstacle_mask = np.zeros((workspace_height_pixels, workspace_width_pixels), dtype=bool)

        if shape_type == 'rectangle':
            poly = Polygon(obj['corners'])

            env_coords = obj['corners']
            grid_coords = [env_to_workspace_grid(x, y) for x, y in env_coords]

            i_min = max(0, min(i for i, j in grid_coords))
            i_max = min(workspace_width_pixels - 1, max(i for i, j in grid_coords))
            j_min = max(0, min(j for i, j in grid_coords))
            j_max = min(workspace_height_pixels - 1, max(j for i, j in grid_coords))

            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    x = x_min + (i + 0.5) * resolution
                    y = y_min + (j + 0.5) * resolution

                    if poly.contains(Point(x, y)):
                        flipped_j = workspace_height_pixels - 1 - j
                        obstacle_mask[flipped_j, i] = True

        elif shape_type == 'circle':
            # Preferred format (from your create_padded_circle): center + radius
            cx, cy = obj.get('center', obj.get('original_center', (None, None)))
            r = obj.get('radius', None)

            # Backwards fallback if you ever stored circles as width/2
            if r is None:
                r = obj.get('width', 0.0) / 2.0

            if cx is None or cy is None or r <= 0:
                continue

            # Bounding box in env coords
            x0, x1 = cx - r, cx + r
            y0, y1 = cy - r, cy + r

            # Convert bbox to workspace pixel index range (clamped)
            i_min, j_min = env_to_workspace_grid(x0, y0)
            i_max, j_max = env_to_workspace_grid(x1, y1)

            # Ensure proper ordering after clamping
            if i_min > i_max:
                i_min, i_max = i_max, i_min
            if j_min > j_max:
                j_min, j_max = j_max, j_min

            r_sq = r * r

            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    x = x_min + (i + 0.5) * resolution
                    y = y_min + (j + 0.5) * resolution
                    if (x - cx) ** 2 + (y - cy) ** 2 <= r_sq:
                        flipped_j = workspace_height_pixels - 1 - j
                        obstacle_mask[flipped_j, i] = True

        else:
            # Unknown shape type
            continue

        # Fill interior with random gray/white-ish values (your existing behavior)
        current_interior_value = random.choice([192, 255, 128, 216])
        workspace_grid[obstacle_mask] = current_interior_value

        # Outline via erosion difference
        structure_size = 2 * outline_thickness + 1
        erode_structure = np.ones((structure_size, structure_size))
        eroded = ndimage.binary_erosion(obstacle_mask, structure=erode_structure)

        outline_mask = obstacle_mask & ~eroded
        workspace_grid[outline_mask] = obstacle_outline_value

    # Place workspace into full padded grid
    start_row = padding_y_bottom
    end_row = start_row + workspace_height_pixels
    start_col = padding_x_left
    end_col = start_col + workspace_width_pixels

    full_grid[start_row:end_row, start_col:end_col] = workspace_grid

    write_pgm(full_grid, f"grid_{grid_id}.pgm", output_dir=output_dir)
    save_map_yaml(grid_id, resolution, map_origin_x, map_origin_y, output_dir=output_dir, grid_dir=output_dir)

    return grid_id
