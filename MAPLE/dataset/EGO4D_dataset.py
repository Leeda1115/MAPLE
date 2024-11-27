import os
import json
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class EGO4D(Dataset):
    def __init__(self, hdf5_folder, json_file, split='train'):
        self.hdf5_folder = hdf5_folder
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
        feature = sample['action_feature']
        label = sample['label']

        # Padding visual_feature to length 49
        padding_length = 15
        current_length = feature.shape[0]

        if current_length < padding_length:
            # If current length is less than padding length, pad the sequence
            padding = F.pad(torch.tensor(feature), (0, 0, padding_length - current_length, 0), value=0)
        else:
            padding = torch.tensor(feature)

        return padding, label, current_length

    def load_data(self, split):
        data = []
        len = 0
        for annotation in self.json_data['cmg_annotation']:
            video_uid = annotation['video_uid']
            start_frame = annotation['action_clip_start_frame']
            end_frame = annotation['action_clip_end_frame']
            start_token = start_frame // 16
            end_token = end_frame // 16
            action_feature = self.load_feature_from_hdf5(self.hdf5_folder, video_uid, start_token, end_token)
            try:
                label = {'verb': annotation['verb_label'], 'noun': annotation['noun_label']}
            except KeyError:
                label = {'verb': annotation['verb'], 'noun': annotation['noun']}
            len+=action_feature.shape[0]
            data.append({'action_feature': action_feature, 'label': label, "len":len})
        return data

    def load_feature_from_hdf5(self, hdf5_file, video_uid, start_v, end_v):
        with h5py.File(hdf5_file, 'r') as f:
            feature_data = f[video_uid][start_v:end_v, :]
        return feature_data

# feature_path = r'/home/share1/lida/cmg/data/EGO4D/features.h5'
# labels_pickle = r'/home/share1/lida/cmg/data/EGO4D/annotations/cmg_train.json'
# dataset = EGO4D(feature_path, labels_pickle,)
# lenth = dataset.__len__()  # 36821
# print(lenth)
# train_dataloader = DataLoader(
#     EGO4D(feature_path, labels_pickle,),
#     batch_size=80,
#     shuffle=True,
#     num_workers=8,
#     pin_memory=True
# )