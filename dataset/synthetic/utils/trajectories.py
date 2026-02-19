import random
import numpy as np

from .planning import theta_star, smooth_path_with_beziers

def interpolate_position_at_time(path, cumulative_time, target_time):
    if target_time <= 0:
        return path[0]
    if target_time >= cumulative_time[-1]:
        return path[-1]

    for i in range(1, len(cumulative_time)):
        if cumulative_time[i] >= target_time:
            t0, t1 = cumulative_time[i-1], cumulative_time[i]
            alpha = (target_time - t0) / (t1 - t0)
            x = path[i-1][0] + alpha * (path[i][0] - path[i-1][0])
            y = path[i-1][1] + alpha * (path[i][1] - path[i-1][1])
            return (x, y)
    return path[-1]

def generate_paths_and_log(ax, stations, grid, pads, writer,
                           task_id, grid_id, mobile_center, agv_center,
                           station_coords_flat,
                           traj_id_start=0, frame_interval=0.25,
                           min_points=25, draw=True,
                           mask_prob=0.3, n_trajectories=20):

    traj_id = traj_id_start
    MIN_SPEED, MAX_SPEED = 0.8, 1.5
    SPEED_VARIABILITY = 0.1
    MIN_STANDING_FRAMES, MAX_STANDING_FRAMES = 5, 7
    STANDING_POSITION_NOISE = 0.02

    if not stations:
        print("  ERROR: No stations available!")
        return traj_id

    successful_trajectories = 0
    max_attempts = n_trajectories * 10
    attempts = 0

    while successful_trajectories < n_trajectories and attempts < max_attempts:
        attempts += 1

        start_station = random.choice(stations)
        goal_station = random.choice(stations)

        sx, sy = start_station
        gx, gy = goal_station

        raw = theta_star((sx, sy), (gx, gy), grid, pads)
        if not raw:
            continue

        path = smooth_path_with_beziers(raw)
        if len(path) < 2:
            continue

        frame_id = 0
        trajectory_frames = []
        base_speed = np.random.uniform(MIN_SPEED, MAX_SPEED)

        distances = []
        cumulative_dist = [0.0]
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(dist)
            cumulative_dist.append(cumulative_dist[-1] + dist)

        cumulative_time = [0.0]
        for dist in distances:
            segment_speed = base_speed * np.random.uniform(1 - SPEED_VARIABILITY, 1 + SPEED_VARIABILITY)
            cumulative_time.append(cumulative_time[-1] + dist / segment_speed)

        total_time = cumulative_time[-1]

        current_time = 0.0
        while current_time <= total_time:
            x, y = interpolate_position_at_time(path, cumulative_time, current_time)
            trajectory_frames.append((x, y))
            current_time += frame_interval

        if trajectory_frames and trajectory_frames[-1] != path[-1]:
            trajectory_frames.append(path[-1])

        if len(trajectory_frames) < min_points:
            frames_needed = min_points - len(trajectory_frames)
            start_standing = min(
                np.random.randint(MIN_STANDING_FRAMES, MAX_STANDING_FRAMES + 1),
                int(frames_needed * 0.6)
            )
            end_standing = frames_needed - start_standing

            initial_frames = []
            start_x, start_y = trajectory_frames[0] if trajectory_frames else (sx, sy)
            for _ in range(start_standing):
                nx = np.random.normal(0, STANDING_POSITION_NOISE)
                ny = np.random.normal(0, STANDING_POSITION_NOISE)
                initial_frames.append((start_x + nx, start_y + ny))

            final_frames = []
            end_x, end_y = trajectory_frames[-1] if trajectory_frames else (gx, gy)
            for _ in range(end_standing):
                nx = np.random.normal(0, STANDING_POSITION_NOISE)
                ny = np.random.normal(0, STANDING_POSITION_NOISE)
                final_frames.append((end_x + nx, end_y + ny))

            trajectory_frames = initial_frames + trajectory_frames + final_frames

        for x, y in trajectory_frames:
            pos_mask = 1 if np.random.random() > mask_prob else 0

            # Always 1 (as requested)
            stations_pos_mask = 1
            robot_pos_mask = 1

            row = [
                task_id, grid_id, traj_id, frame_id,
                0, 1, x, y, 0, pos_mask,
                stations_pos_mask, robot_pos_mask
            ]

            row.extend(station_coords_flat)
            writer.writerow(row)
            frame_id += 1

        if draw and ax is not None:
            px, py = zip(*path)
            ax.plot(px, py, lw=1.2, alpha=0.7)

        successful_trajectories += 1
        traj_id += 1

    if successful_trajectories < n_trajectories:
        print(f"    ⚠️  Warning: Could only generate {successful_trajectories}/{n_trajectories} trajectories after {attempts} attempts")

    return traj_id
