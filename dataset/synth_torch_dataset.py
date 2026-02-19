import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import yaml
import glob
import re
from collections import OrderedDict

class TrajectoryPredictionDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int, pred_len: int, stride: int):
        self.data_path = data_path
        self.seq_len: int = seq_len
        self.pred_len: int = pred_len
        self.max_human_agents: int = 1
        self.max_robot_agents: int = 2
        self.max_stations: int = 0
        self.stride = stride
        self.max_cache_size = 80

        self.map_cache: OrderedDict = OrderedDict()
        self.map_metadata_cache: dict = {}

        self.input_windows: list[dict] = []
        self.prediction_windows: list[dict] = []
        self.map_ids: list[str] = []
        self.dataset_indices: list[int] = []

        self._read_data()

    def _find_paths_and_maps(self):
        paths_files = glob.glob(os.path.join(self.data_path, "paths_*.csv"))

        if not paths_files:
            print(f"No paths_*.csv files found in {self.data_path}")
            return []

        dataset_pairs = []
        for paths_file in paths_files:
            match = re.search(r'paths_(\d+)\.csv', os.path.basename(paths_file))
            if match:
                idx = int(match.group(1))
                maps_folder = os.path.join(self.data_path, f"maps_{idx}")

                if os.path.exists(maps_folder):
                    dataset_pairs.append((idx, paths_file, maps_folder))
                else:
                    print(f"Warning: maps_{idx} folder not found for {paths_file}")

        dataset_pairs.sort(key=lambda x: x[0])
        return dataset_pairs

    def _load_map_metadata(self, map_id: str, dataset_idx: int):
        cache_key = (dataset_idx, map_id)
        if cache_key in self.map_metadata_cache:
            return self.map_metadata_cache[cache_key]

        maps_folder = os.path.join(self.data_path, f"maps_{dataset_idx}")
        yaml_path = os.path.join(maps_folder, f"map_{map_id}.yaml")

        default_meta = {'origin': [-6.4, -6.4], 'resolution': 0.05}

        if not os.path.exists(yaml_path):
            print(f"yaml with name {yaml_path} does not exist")
            self.map_metadata_cache[cache_key] = default_meta
            return default_meta

        try:
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            metadata = {
                'origin': yaml_data.get('origin'),
                'resolution': yaml_data.get('resolution', 0.05)
            }
            self.map_metadata_cache[cache_key] = metadata
            return metadata
        except Exception as e:
            print(f"Error loading yaml: {e}")
            self.map_metadata_cache[cache_key] = default_meta
            return default_meta

    def _load_map_image(self, map_id: str, dataset_idx: int):
        cache_key = (dataset_idx, map_id)

        if cache_key in self.map_cache:
            self.map_cache.move_to_end(cache_key)
            return self.map_cache[cache_key]

        maps_folder = os.path.join(self.data_path, f"maps_{dataset_idx}")
        image_path = os.path.join(maps_folder, f"grid_{map_id}.pgm")

        if not os.path.exists(image_path):
            empty_image = torch.zeros(1, 256, 256)
            self._add_to_cache(cache_key, empty_image)
            return empty_image

        try:
            image = Image.open(image_path)
            if image.mode != 'L':
                image = image.convert('L')
            image_tensor = transforms.ToTensor()(image)
            self._add_to_cache(cache_key, image_tensor)
            return image_tensor
        except Exception as e:
            print(f"Error loading map image {image_path}: {e}")
            empty_image = torch.zeros(1, 256, 256)
            self._add_to_cache(cache_key, empty_image)
            return empty_image

    def _add_to_cache(self, cache_key, image_tensor):
        self.map_cache[cache_key] = image_tensor
        if len(self.map_cache) > self.max_cache_size:
            self.map_cache.popitem(last=False)

    def _process_csv_file(self, csv_file: str, dataset_idx: int):
        if not os.path.exists(csv_file):
            print(f"CSV file {csv_file} not found")
            return

        df = pd.read_csv(csv_file)
        if 'grid_id' not in df.columns:
            print(f"No grid_id in {csv_file}, skipping")
            return

        all_cols = df.columns.tolist()

        station_x_cols = sorted(
            [c for c in all_cols if re.match(r'station_\d+_x', c)],
            key=lambda x: int(re.search(r'(\d+)', x).group(1))
        )
        station_y_cols = sorted(
            [c for c in all_cols if re.match(r'station_\d+_y', c)],
            key=lambda x: int(re.search(r'(\d+)', x).group(1))
        )

        base_cols = ['x', 'y', 'z', 'agent_type', 'frame_id', 'agent_id']
        mask_cols = ['pos_mask', 'robot_pos/mask']

        station_cols_ordered = []
        for sx, sy in zip(station_x_cols, station_y_cols):
            station_cols_ordered.extend([sx, sy])

        feature_cols = base_cols + station_cols_ordered + mask_cols
        col_map = {name: i for i, name in enumerate(feature_cols)}

        traj_ids = df['traj_id'].unique()

        for traj_id in tqdm(traj_ids, desc=f'Processing paths_{dataset_idx}', unit='trajectory'):
            group = df[df['traj_id'] == traj_id]
            map_id = str(group['grid_id'].iloc[0])

            sensor_data = group[feature_cols].values

            frame_col_idx = col_map['frame_id']
            total_frames = int(sensor_data[:, frame_col_idx].max()) + 1

            frame_dict = {}
            for row in sensor_data:
                fid = int(row[frame_col_idx])
                if fid not in frame_dict:
                    frame_dict[fid] = []
                frame_dict[fid].append(row)

            window_len = self.seq_len + self.pred_len

            idx_agent_type = col_map['agent_type']
            idx_agent_id   = col_map['agent_id']
            idx_pos_mask   = col_map['pos_mask']
            idx_robot_mask = col_map['robot_pos/mask']

            for start_frame in range(0, total_frames - window_len + 1, self.stride):
                human_window     = np.full((self.max_human_agents, self.seq_len, 3), np.nan, dtype=np.float32)
                prediction_window = np.full((self.max_human_agents, self.pred_len, 3), np.nan, dtype=np.float32)
                robot_window     = np.full((self.max_robot_agents, self.seq_len, 3), np.nan, dtype=np.float32)

                human_pos_mask      = np.zeros((self.max_human_agents, self.seq_len), dtype=np.float32)
                prediction_pos_mask = np.zeros((self.max_human_agents, self.pred_len), dtype=np.float32)
                robot_pos_mask      = np.zeros((self.max_robot_agents, self.seq_len), dtype=np.float32)

                human_velocity = np.zeros((self.max_human_agents, self.seq_len), dtype=np.float32)

                station_window = np.zeros((self.max_stations, self.seq_len, 2), dtype=np.float32)

                human_ids = set()
                robot_ids = set()

                for fid in range(start_frame, start_frame + window_len):
                    if fid in frame_dict:
                        for row in frame_dict[fid]:
                            if row[idx_agent_type] == 1:
                                human_ids.add(int(row[idx_agent_id]))
                            else:
                                robot_ids.add(int(row[idx_agent_id]))

                human_id_to_slot = {aid: i for i, aid in enumerate(sorted(human_ids)[:self.max_human_agents])}
                robot_id_to_slot = {aid: i for i, aid in enumerate(sorted(robot_ids)[:self.max_robot_agents])}

                for j in range(self.pred_len):
                    fid = start_frame + self.seq_len + j
                    if fid in frame_dict:
                        for row in frame_dict[fid]:
                            a_id = int(row[idx_agent_id])
                            if row[idx_agent_type] == 1 and a_id in human_id_to_slot:
                                slot = human_id_to_slot[a_id]
                                if row[idx_pos_mask] == 1:
                                    prediction_window[slot, j, :] = row[0:3]
                                    prediction_pos_mask[slot, j] = 1.0
                                # masked frames left as zero-initialized default

                if np.isnan(prediction_window).all():
                    continue

                for i in range(self.seq_len):
                    fid = start_frame + i
                    if fid in frame_dict:
                        for row in frame_dict[fid]:
                            agent_id = int(row[idx_agent_id])
                            pos_mask = row[idx_pos_mask]

                            if row[idx_agent_type] == 1 and agent_id in human_id_to_slot:
                                slot = human_id_to_slot[agent_id]
                                if pos_mask == 1:
                                    human_window[slot, i, :] = row[0:3]
                                    human_pos_mask[slot, i] = 1.0

                            elif row[idx_agent_type] == 0 and agent_id in robot_id_to_slot:
                                slot = robot_id_to_slot[agent_id]
                                robot_window[slot, i, :] = row[0:3]
                                robot_pos_mask[slot, i] = row[idx_robot_mask]

                        if len(frame_dict[fid]) > 0:
                            first_row = frame_dict[fid][0]
                            for station_idx in range(self.max_stations):
                                sx_key = f'station_{station_idx}_x'
                                sy_key = f'station_{station_idx}_y'
                                if sx_key in col_map and sy_key in col_map:
                                    station_window[station_idx, i, 0] = first_row[col_map[sx_key]]
                                    station_window[station_idx, i, 1] = first_row[col_map[sy_key]]

                if np.isnan(human_window).all():
                    continue

                for agent_idx in range(self.max_human_agents):
                    for t in range(1, self.seq_len):
                        if (human_pos_mask[agent_idx, t] == 1 and
                            human_pos_mask[agent_idx, t-1] == 1 and
                            not np.isnan(human_window[agent_idx, t, 0]) and
                            not np.isnan(human_window[agent_idx, t-1, 0])):
                            human_velocity[agent_idx, t] = np.linalg.norm(
                                human_window[agent_idx, t, 0:2] - human_window[agent_idx, t-1, 0:2]
                            )
                    human_velocity[agent_idx, 0] = human_velocity[agent_idx, 1]

                human_window      = np.nan_to_num(human_window, 0.0).astype(np.float32)
                robot_window      = np.nan_to_num(robot_window, 0.0).astype(np.float32)
                prediction_window = np.nan_to_num(prediction_window, 0.0).astype(np.float32)

                all_agents_pos = np.concatenate([
                    human_window[:, :, 0:2],
                    robot_window[:, :, 0:2],
                    station_window
                ], axis=0)

                self.input_windows.append({
                    'human_pos':       human_window[:, :, 0:2],
                    'human_vel':       human_velocity,
                    'human_pos/mask':  np.expand_dims(human_pos_mask, axis=-1),
                    'robot_pos':       robot_window[:, :, 0:2],
                    'robot_pos/mask':  np.expand_dims(robot_pos_mask, axis=-1),
                    'stations_pos':    station_window,
                    'all_agents_pos':  all_agents_pos,
                })

                self.prediction_windows.append({
                    'prediction_pos':      prediction_window[:, :, 0:2],
                    'prediction_pos_mask': prediction_pos_mask,
                    'prediction_pos/mask': np.expand_dims(prediction_pos_mask, axis=-1),
                })

                self.map_ids.append(map_id)
                self.dataset_indices.append(dataset_idx)

    def _detect_max_stations(self, dataset_pairs: list) -> int:
        max_stations = 0
        for _, paths_file, _ in dataset_pairs:
            try:
                header = pd.read_csv(paths_file, nrows=0).columns.tolist()
                count = sum(1 for c in header if re.match(r'station_\d+_x', c))
                max_stations = max(max_stations, count)
            except Exception as e:
                print(f"Warning: could not read header of {paths_file}: {e}")
        return max_stations

    def _read_data(self):
        dataset_pairs = self._find_paths_and_maps()

        if not dataset_pairs:
            print("No valid dataset pairs found")
            return

        print(f"Found {len(dataset_pairs)} dataset pairs")

        self.max_stations = self._detect_max_stations(dataset_pairs)
        print(f"Detected max stations across all files: {self.max_stations}")

        for dataset_idx, paths_file, maps_folder in dataset_pairs:
            print(f"\nProcessing dataset {dataset_idx}: {os.path.basename(paths_file)} with {os.path.basename(maps_folder)}")
            self._process_csv_file(paths_file, dataset_idx)

        print(f"\nTotal windows extracted: {len(self.input_windows)}")

    def __len__(self):
        return len(self.input_windows)

    def __getitem__(self, idx: int):
        input_dict  = self.input_windows[idx]
        pred_dict   = self.prediction_windows[idx]
        map_id      = self.map_ids[idx]
        dataset_idx = self.dataset_indices[idx]

        map_image    = self._load_map_image(map_id, dataset_idx)
        map_metadata = self._load_map_metadata(map_id, dataset_idx)

        x = {key: torch.tensor(value, dtype=torch.float32) for key, value in input_dict.items()}
        x['map_image']      = map_image
        x['map_origin']     = torch.tensor(map_metadata['origin'], dtype=torch.float32)
        x['map_resolution'] = torch.tensor(map_metadata['resolution'], dtype=torch.float32)

        y = {key: torch.tensor(value, dtype=torch.float32) for key, value in pred_dict.items()}

        return x, y