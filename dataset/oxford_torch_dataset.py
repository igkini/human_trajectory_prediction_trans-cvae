import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
import os
from tqdm import tqdm
import re
import numpy as np
import json
from PIL import Image
import torchvision.transforms as transforms
import yaml


FEATURE_COLS=['x', 'y', 'z', 'agent_type', 'frame_id','agent_id', 'bottle_x', 'bottle_y', 'plant_x', 'plant_y', 'table_x', 'table_y', 'food_x', 'food_y', 'entrance_x', 'entrance_y', 'exit_x', 'exit_y']
DESIRED_KPTS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,91,112]

class OxfordrajectoryPredictionDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int, pred_len: int, stride: int):
        self.all_files: list[str]=glob.glob(os.path.join(data_path, "*.csv"))
        self.json_files: list[str]=glob.glob(os.path.join(data_path, "*.json"))
        self.seq_len:int=seq_len
        self.pred_len: int=pred_len
        self.data_path = data_path
        self.max_human_agents: int=1
        self.max_robot_agents: int=2
        self.max_stations: int=6  # bottle, plant, table, food, entrance, exit
        self.stride: int=stride
        self.desired_kpts: list[int]=DESIRED_KPTS
        self.input_windows: list[dict]=[]
        self.prediction_windows: list[dict]=[]
        self.map_ids: list[str] = []

        self.map_cache: dict = {}
        self.map_metadata_cache: dict = {}

        self.kpt_thresh: float=0.4
        self.target_seq: list[np.ndarray]=[]
        self._read_data()

    def _load_map_metadata(self, map_id: str):
        if map_id in self.map_metadata_cache:
            return self.map_metadata_cache[map_id]

        yaml_path = os.path.join(self.data_path, f"maps/map_{map_id}.yaml")
        if not os.path.exists(yaml_path):
            print(f"No path {yaml_path}")
            metadata = {'origin': [0.0, 0.0, 0.0], 'resolution': 0.05}
            self.map_metadata_cache[map_id] = metadata
            return metadata

        try:
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            metadata = {
                'origin': yaml_data.get('origin', [0.0, 0.0, 0.0]),
                'resolution': yaml_data.get('resolution', 0.05)
            }
            self.map_metadata_cache[map_id] = metadata
            return metadata
        except Exception as e:
            print(f"Error loading yaml: {e}")
            metadata = {'origin': [0.0, 0.0, 0.0], 'resolution': 0.05}
            self.map_metadata_cache[map_id] = metadata
            return metadata

    def _load_map_image(self, map_id: str):
        if map_id in self.map_cache:
            return self.map_cache[map_id]

        image_path = os.path.join(self.data_path, f"maps/map_{map_id}.pgm")

        if not os.path.exists(image_path):
            empty_image = torch.zeros(1, 256, 256)
            self.map_cache[map_id] = empty_image
            return empty_image

        try:
            image = Image.open(image_path)
            if image.mode != 'L':
                image = image.convert('L')
            image_tensor = transforms.ToTensor()(image)
            self.map_cache[map_id] = image_tensor
            return image_tensor
        except Exception as e:
            print(f"Error loading map image {image_path}: {e}")
            empty_image = torch.zeros(1, 256, 256)
            self.map_cache[map_id] = empty_image
            return empty_image

    def _read_data(self):
        for file in tqdm(self.all_files, desc='Loading files', unit='file'):
            m=re.search(r"map(\d)_run(\d)_(\d)",file)
            if not m:
                print("No csv with the desired name pattern found")
                continue
            map_id, run_id, fps = m.groups()

            jsonfile = None
            for jf in self.json_files:
                m_json = re.search(r"map(\d+)_run(\d+)", jf)
                if m_json:
                    map_id_json, run_id_json = m_json.groups()
                    if map_id_json == map_id and run_id_json == run_id:
                        jsonfile = jf
                        break

            json_data = None
            if jsonfile is not None:
                try:
                    with open(jsonfile, "r") as f:
                        json_data = json.load(f)
                    print(f"Loaded JSON data for map{map_id}_run{run_id}")
                except Exception as e:
                    print(f"Error loading JSON {jsonfile}: {e}")

            sensor_df = pd.read_csv(file)[FEATURE_COLS]
            sensor_data = sensor_df.values
            total_frames = int(sensor_data[:, 4].max()) + 1

            frame_dict = {}
            for row in sensor_data:
                fid = int(row[4])
                if fid not in frame_dict:
                    frame_dict[fid] = []
                frame_dict[fid].append(row)

            window_len = self.seq_len + self.pred_len
            for start_frame in range(0, total_frames - window_len + 1, self.stride):
                human_window = np.full((self.max_human_agents, self.seq_len, 4 + 4*len(self.desired_kpts)), np.nan, dtype=np.float32)
                human_kpts_window = np.full((self.max_human_agents, self.seq_len, len(self.desired_kpts), 4), np.nan, dtype=np.float32)
                prediction_window = np.full((self.max_human_agents, self.pred_len, 3), np.nan, dtype=np.float32)
                # TODO: add robot_window to input_windows once robot features are integrated
                # robot_window = np.full((self.max_robot_agents, self.seq_len, 4), np.nan, dtype=np.float32)
                stations_window = np.full((self.max_stations, self.seq_len, 2), np.nan, dtype=np.float32)

                human_ids = set()
                robot_ids = set()
                for fid in range(start_frame, start_frame + window_len):
                    if fid in frame_dict:
                        for row in frame_dict[fid]:
                            if row[3] == 1:
                                human_ids.add(int(row[5]))
                            else:
                                robot_ids.add(int(row[5]))

                human_id_to_slot = {aid: i for i, aid in enumerate(sorted(human_ids)[:self.max_human_agents])}
                robot_id_to_slot = {aid: i for i, aid in enumerate(sorted(robot_ids)[:self.max_robot_agents])}

                for j in range(self.pred_len):
                    fid = start_frame + self.seq_len + j
                    if fid in frame_dict:
                        for row in frame_dict[fid]:
                            if row[3] == 1 and int(row[5]) in human_id_to_slot:
                                slot = human_id_to_slot[int(row[5])]
                                prediction_window[slot, j, :] = row[0:3]

                if np.isnan(prediction_window).any():
                    continue

                for i in range(self.seq_len):
                    fid = start_frame + i

                    if fid in frame_dict:
                        for row in frame_dict[fid]:
                            agent_id = int(row[5])
                            if row[3] == 1 and agent_id in human_id_to_slot:
                                slot = human_id_to_slot[agent_id]
                                human_window[slot, i, 0:3] = row[0:3]
                            # TODO: populate robot_window once integrated
                            # elif row[3] == 0 and agent_id in robot_id_to_slot:
                            #     slot = robot_id_to_slot[agent_id]
                            #     robot_window[slot, i, 0:3] = row[0:3]

                        for row in frame_dict[fid]:
                            stations_window[0, i, :] = row[6:8]   # bottle
                            stations_window[1, i, :] = row[8:10]  # plant
                            stations_window[2, i, :] = row[10:12] # table
                            stations_window[3, i, :] = row[12:14] # food
                            stations_window[4, i, :] = row[14:16] # entrance
                            stations_window[5, i, :] = row[16:18] # exit

                    human_missing = np.isnan(human_window[:self.max_human_agents, i, 0:3]).all(axis=1)
                    # robot_missing = np.isnan(robot_window[:self.max_robot_agents, i, 0:3]).all(axis=1)

                    human_window[:self.max_human_agents, i, 3] = (~human_missing).astype(np.float32)
                    # robot_window[:self.max_robot_agents, i, 3] = (~robot_missing).astype(np.float32)

                    if json_data is not None:
                        if i < len(json_data["instance_info"]) and start_frame + i < len(json_data["instance_info"]):
                            instances = json_data["instance_info"][start_frame + i]['instances']
                            if len(instances) > 0 and len(human_id_to_slot) > 0:
                                human_instance = instances[0]
                                kpts = np.array(human_instance["keypoints"], dtype=np.float32)[self.desired_kpts]
                                scores = np.array(human_instance["keypoint_scores"], dtype=np.float32)[self.desired_kpts]

                                kpts_mask = scores < self.kpt_thresh
                                kpts_with_mask = np.zeros((25, 4), dtype=np.float32)
                                kpts_with_mask[:, :3] = kpts
                                kpts_with_mask[:, 3] = (~kpts_mask).astype(np.float32)
                                kpts_with_mask[kpts_mask, :3] = np.nan

                                first_slot = 0
                                human_kpts_window[first_slot, i, :, 0:3] = kpts
                                human_kpts_window[first_slot, i, :, 3] = ~kpts_mask
                                human_window[first_slot, i, 4:] = kpts_with_mask.reshape(-1)

                if np.isnan(human_window[:, :, 0:3]).any():
                    continue

                human_window_mask = np.where(np.isnan(human_window), 0.0, 1.0).astype(np.float32)
                human_window_mask[:, :, 7::4] = 0.0
                human_window_mask[:, :, 3] = 0.0

                # robot_window_mask = np.where(np.isnan(robot_window), 0.0, 1.0).astype(np.float32)
                # robot_window_mask[:, :, 3] = 0.0

                prediction_window_mask = np.where(np.isnan(prediction_window), 0.0, 1.0).astype(np.float32)

                stations_window_mask = (~np.isnan(stations_window[:, :, 0:1])).astype(np.float32)

                human_window = np.nan_to_num(human_window, 0.0).astype(np.float32)
                human_kpts_window = np.nan_to_num(human_kpts_window, 0.0).astype(np.float32)
                # robot_window = np.nan_to_num(robot_window, 0.0).astype(np.float32)
                prediction_window = np.nan_to_num(prediction_window, 0.0).astype(np.float32)
                stations_window = np.nan_to_num(stations_window, 0.0).astype(np.float32)

                self.input_windows.append({
                    'human_pos':          human_window[:, :, 0:2],
                    'human_pos_mask':     human_window_mask[:, :, 0:4],
                    'stations_pos':       stations_window,
                    'stations_pos_mask':  stations_window_mask,
                    # TODO: uncomment once robot_window is integrated
                    # 'robot_pos':        robot_window[:, :, 0:2],
                    # 'robot_pos_mask':   robot_window_mask[:, :, 0:4],
                })

                self.prediction_windows.append({
                    'prediction_pos':      prediction_window[:, :, 0:2],
                    'prediction_pos_mask': prediction_window_mask[:, :, 0:1]
                })

                self.map_ids.append(map_id)

    def __len__(self):
        return len(self.input_windows)

    def __getitem__(self, idx: int):
        input_dict = self.input_windows[idx]
        pred_dict  = self.prediction_windows[idx]
        map_id     = self.map_ids[idx]

        map_image    = self._load_map_image(map_id)
        map_metadata = self._load_map_metadata(map_id)

        x = {key: torch.tensor(value, dtype=torch.float32) for key, value in input_dict.items()}
        x['map_image']      = map_image
        x['map_origin']     = torch.tensor(map_metadata['origin'], dtype=torch.float32)
        x['map_resolution'] = torch.tensor(map_metadata['resolution'], dtype=torch.float32)

        y = {key: torch.tensor(value, dtype=torch.float32) for key, value in pred_dict.items()}

        return x, y