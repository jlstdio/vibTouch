import os
import numpy as np
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from util.util import bandpass_filter  # 밴드패스 필터 함수 사용
import scipy.signal
import torch.nn.functional as F


class GestureDataset(Dataset):
    def __init__(self, root_dir, gesture_types):
        """
        Args:
            root_dir (str): 데이터가 저장된 루트 디렉토리 경로
            gesture_types (list): ['tap', 'slide']와 같은 제스처 유형 리스트
        """
        self.root_dir = root_dir
        self.gesture_types = gesture_types
        self.data = []
        self.main_labels = []
        self.sub_labels = []
        self.audio_lengths = []

        self.acc_mean = 0
        self.acc_std = 1
        self.audio_mean = 0
        self.audio_std = 1

        self._load_data()
        self._calculate_normalization()

    def _load_data(self):
        # Main Label과 Sub Label을 동시에 처리
        label_map = {
            'slide': {'3_0': 0, '3_1': 1, '3_2': 2, '3_3': 3},
            'tap': {'0': 4, '1': 5, '2': 6, '3': 7}
        }

        for gesture in self.gesture_types:
            for class_name, class_id in label_map[gesture].items():
                acc_dir = os.path.join(self.root_dir, gesture, class_name, 'acc')
                audio_dir = os.path.join(self.root_dir, gesture, class_name, 'audio')

                # 가속도계와 오디오 파일 목록 가져오기
                acc_files = sorted([f for f in os.listdir(acc_dir) if f.endswith('.csv')])
                audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

                for acc_file, audio_file in zip(acc_files, audio_files):
                    acc_path = os.path.join(acc_dir, acc_file)
                    audio_path = os.path.join(audio_dir, audio_file)

                    # 가속도계 데이터 로드
                    acc_data = pd.read_csv(acc_path)[['x', 'y', 'z']].values

                    # 오디오 데이터 로드 및 밴드패스 필터 적용
                    try:
                        audio_data, sample_rate = torchaudio.load(audio_path)  # [channels, samples]
                        if audio_data.shape[0] > 1:
                            audio_data = audio_data.mean(dim=0).numpy()  # 모노로 변환 (평균)
                        else:
                            audio_data = audio_data.squeeze().numpy()  # [samples]
                        audio_data = bandpass_filter(audio_data, sample_rate, low_freq=10, high_freq=1000, order=5)
                        audio_length = len(audio_data)
                    except Exception as e:
                        print(f"Failed to load or filter {audio_path}: {e}")
                        continue

                    # 가속도계 데이터 패딩 또는 트렁케이션을 오디오 길이에 맞춤
                    acc_length = len(acc_data)
                    if acc_length >= audio_length:
                        acc_data_padded = acc_data[:audio_length, :]
                    else:
                        padding = np.zeros((audio_length - acc_length, 3))
                        acc_data_padded = np.vstack((acc_data, padding))  # [audio_length, 3]

                    # 오디오 데이터는 패딩/트렁케이션하지 않음
                    self.data.append((acc_data_padded, audio_data))
                    self.sub_labels.append(class_id)

                    # Main Label: 'tap' -> 1, 'slide' -> 0
                    main_label = 1 if gesture == 'tap' else 0
                    self.main_labels.append(main_label)
                    self.audio_lengths.append(audio_length)

    def _calculate_normalization(self):
        """
        전체 데이터에 대한 평균과 표준편차 계산
        """
        # 모든 가속도계 데이터와 오디오 데이터를 하나로 합침
        acc_data_all = np.concatenate([d[0] for d in self.data], axis=0)  # [total_time, 3]
        audio_data_all = np.concatenate([d[1] for d in self.data])  # [total_time]

        self.acc_mean = np.mean(acc_data_all, axis=0)  # [3]
        self.acc_std = np.std(acc_data_all, axis=0)  # [3]

        self.audio_mean = np.mean(audio_data_all)  # scalar
        self.audio_std = np.std(audio_data_all)  # scalar

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        acc_data, audio_data = self.data[idx]
        main_label = self.main_labels[idx]
        sub_label = self.sub_labels[idx]
        audio_length = self.audio_lengths[idx]

        # 데이터 정규화
        acc_data = (acc_data - self.acc_mean) / self.acc_std
        audio_data = (audio_data - self.audio_mean) / self.audio_std

        # 텐서 변환 및 차원 조정
        acc_data = torch.tensor(acc_data, dtype=torch.float32).transpose(0, 1)  # [3, audio_length]
        audio_data = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)  # [1, audio_length]

        main_label = torch.tensor(main_label, dtype=torch.float32)  # [1]
        sub_label = torch.tensor(sub_label, dtype=torch.long)  # [1]

        return acc_data, audio_data, main_label, sub_label


def collate_fn(batch):
    acc_data, audio_data, main_labels, sub_labels = zip(*batch)

    # 배치 내 최대 길이 찾기
    max_length = max([acc.shape[1] for acc in acc_data])  # time dimension

    # 가속도계 데이터와 오디오 데이터 패딩
    padded_acc = []
    padded_audio = []
    for acc, audio in zip(acc_data, audio_data):
        current_length = acc.shape[1]
        if current_length < max_length:
            pad_size = max_length - current_length
            # [C, L] 형태에서 L을 패딩
            acc_padded = F.pad(acc, (0, pad_size))  # [3, max_length]
            audio_padded = F.pad(audio, (0, pad_size))  # [1, max_length]
        else:
            acc_padded = acc[:, :max_length]
            audio_padded = audio[:, :max_length]
        padded_acc.append(acc_padded)
        padded_audio.append(audio_padded)

    # 배치 텐서로 변환
    acc_data_padded = torch.stack(padded_acc)  # [batch_size, 3, max_length]
    audio_data_padded = torch.stack(padded_audio)  # [batch_size, 1, max_length]

    main_labels = torch.stack(main_labels)  # [batch_size]
    sub_labels = torch.stack(sub_labels)  # [batch_size]

    return acc_data_padded, audio_data_padded, main_labels, sub_labels

