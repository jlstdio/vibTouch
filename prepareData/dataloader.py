import os
import numpy as np
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset


class GestureDataset(Dataset):
    def __init__(self, root_dir, gesture_types):
        self.root_dir = root_dir
        self.gesture_types = gesture_types
        self.data = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        label_map = {'slide': {'3_0': 0, '3_1': 1, '3_2': 2, '3_3': 3},
                     'tap': {'0': 4, '1': 5, '2': 6, '3': 7}}
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
                        audio_data, _ = torchaudio.load(audio_path)
                        audio_data = audio_data.squeeze().numpy()  # Flatten audio data
                    except Exception as e:
                        print(f"Failed to load {audio_path}: {e}")
                        continue

                    # Pad acc_data to (80, 3)
                    acc_data_padded = np.zeros((80, 3))
                    acc_data_padded[:acc_data.shape[0], :acc_data.shape[1]] = acc_data

                    # Pad audio_data to (12800,)
                    audio_data_padded = np.zeros(12800)
                    audio_data_padded[:audio_data.shape[0]] = audio_data

                    self.data.append((acc_data_padded, audio_data_padded))
                    self.labels.append(class_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        acc_data, audio_data = self.data[idx]
        label = self.labels[idx]

        acc_data = torch.tensor(acc_data, dtype=torch.float32)
        audio_data = torch.tensor(audio_data, dtype=torch.float32)

        return acc_data, audio_data, label


def collate_fn(batch):
    acc_data, audio_data, labels = zip(*batch)
    acc_data = torch.stack(acc_data)
    audio_data = torch.stack(audio_data)
    labels = torch.tensor(labels)

    return acc_data, audio_data, labels