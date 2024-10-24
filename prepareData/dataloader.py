import os
import numpy as np
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from util.util import lowpass_filter


class GestureDataset(Dataset):

    def __init__(self, root_dir, gesture_types, classifierStep='tap'):
        self.root_dir = root_dir
        self.gesture_types = gesture_types
        self.data = []
        self.labels = []

        self.acc_mean = 0
        self.acc_std = 1
        self.audio_mean = 0
        self.audio_std = 1

        self.classifierStep = classifierStep

        self._load_data()
        self._calculate_normalization()

    def _load_data(self):
        if self.classifierStep == 'typeClassifier':
            label_map = {'slide': {'3_0': 0, '3_1': 0, '3_2': 0, '3_3': 0}, 'tap': {'0': 1, '1': 1, '2': 1, '3': 1}}
        else:
            label_map = {'slide': {'3_0': 0, '3_1': 1, '3_2': 2, '3_3': 3}, 'tap': {'0': 4, '1': 5, '2': 6, '3': 7}}

        for gesture in self.gesture_types:
            for class_name, class_id in label_map[gesture].items():
                acc_dir = os.path.join(self.root_dir, gesture, class_name, 'acc')
                audio_dir = os.path.join(self.root_dir, gesture, class_name, 'audio')
                acc_files = sorted([f for f in os.listdir(acc_dir) if f.endswith('.csv')])
                audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

                for acc_file, audio_file in zip(acc_files, audio_files):
                    acc_path = os.path.join(acc_dir, acc_file)
                    audio_path = os.path.join(audio_dir, audio_file)

                    acc_data = pd.read_csv(acc_path)[['x', 'y', 'z']].values
                    try:
                        audio_data, sample_rate = torchaudio.load(audio_path)
                        audio_data = lowpass_filter(audio_data, sample_rate, 2000).squeeze().numpy()
                    except Exception as e:
                        print(f"Failed to load {audio_path}: {e}")
                        continue

                    # Pad acc_data to match the length of filtered audio_data
                    acc_data_padded = np.zeros((12800, 3))
                    acc_data_padded[:acc_data.shape[0], :acc_data.shape[1]] = acc_data

                    # Pad or truncate audio_data to (12800,)
                    audio_data_padded = np.zeros(12800)
                    audio_data_padded[:len(audio_data)] = audio_data

                    self.data.append((acc_data_padded, audio_data_padded))
                    self.labels.append(class_id)

    def _calculate_normalization(self):
        acc_data_all = np.concatenate([d[0] for d in self.data], axis=0)
        audio_data_all = np.concatenate([d[1] for d in self.data])

        self.acc_mean = np.mean(acc_data_all, axis=0)
        self.acc_std = np.std(acc_data_all, axis=0)

        self.audio_mean = np.mean(audio_data_all)
        self.audio_std = np.std(audio_data_all)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        acc_data, audio_data = self.data[idx]
        label = self.labels[idx]

        # Normalize the data
        acc_data = (acc_data - self.acc_mean) / self.acc_std
        audio_data = (audio_data - self.audio_mean) / self.audio_std

        acc_data = torch.tensor(acc_data, dtype=torch.float32)
        audio_data = torch.tensor(audio_data, dtype=torch.float32)

        return acc_data, audio_data, label

def collate_fn(batch):
    acc_data, audio_data, labels = zip(*batch)
    acc_data = torch.stack(acc_data)
    audio_data = torch.stack(audio_data)
    labels = torch.tensor(labels)

    return acc_data, audio_data, labels
