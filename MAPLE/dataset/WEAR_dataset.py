import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class WEARDataset(Dataset):
    def __init__(self, npy_folder, json_file, split='train'):
        self.npy_folder = npy_folder
        self.json_data = self.load_json(json_file)
        self.data = self.load_data(split)

    def load_json(self, json_file):
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        return json_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        visual_feature = sample['action_feature']
        if isinstance(visual_feature, np.ndarray):
            visual_feature = torch.tensor(visual_feature, dtype=torch.float32)
        label = torch.as_tensor(sample['label'])

        # Padding visual_feature to length 366
        padding_length = 366
        current_length = visual_feature.shape[0]

        if current_length < padding_length:
            # If current length is less than padding length, pad the sequence
            padding = F.pad(visual_feature.clone().detach(), (0, 0, 0, padding_length - current_length), value=0)
        else:
            # If current length is greater or equal, truncate the sequence
            padding = visual_feature[:padding_length, :]

        return padding, label, current_length

    def load_data(self, split):
        data = []
        for sbj, sbj_data in self.json_data['database'].items():
            npy_file = os.path.join(self.npy_folder, f"{sbj}.npy")
            full_video_feature = np.load(npy_file)
            if sbj_data['subset'].lower() == split.lower():
                for annotation in sbj_data['annotations']:
                    label = annotation['label_id']

                    # Adjust start and end seconds based on your sampling strategy, Adjust for the 0.5-second sampling and feature extraction
                    start_sec, end_sec = annotation['segment']
                    start_v = int(start_sec/0.5)
                    end_v = int(end_sec/0.5)

                    action_feature = full_video_feature[start_v:end_v, :]
                    data.append({'action_feature': action_feature,  'label': label})
            del full_video_feature
        return data

