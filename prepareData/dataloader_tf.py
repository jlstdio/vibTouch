import random
import librosa
import numpy as np
from scipy.signal import butter, sosfilt
import pandas as pd
import os
import tensorflow as tf


def bandpass_filter(audio, sample_rate, low_freq=10, high_freq=1000, order=5):
    sos = butter(order, [low_freq, high_freq], btype='band', fs=sample_rate, output='sos')
    filtered = sosfilt(sos, audio)
    return filtered


import random
import librosa
import numpy as np
from scipy.signal import butter, sosfilt
import pandas as pd
import os
import tensorflow as tf


def bandpass_filter(audio, sample_rate, low_freq=10, high_freq=1000, order=5):
    sos = butter(order, [low_freq, high_freq], btype='band', fs=sample_rate, output='sos')
    filtered = sosfilt(sos, audio)
    return filtered


class GestureDataset(tf.keras.utils.Sequence):
    def __init__(self, root_dir, gesture_types, batch_size=32, shuffle=True, indices=None):
        self.root_dir = root_dir
        self.gesture_types = gesture_types
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = indices  # list of indices to include
        self.data = []
        self.main_labels = []
        self.sub_labels = []
        self.audio_lengths = []

        self.acc_mean = None
        self.acc_std = None
        self.audio_mean = None
        self.audio_std = None

        self.label_map = {
            'slide': {'3_0': 0, '3_1': 1, '3_2': 2, '3_3': 3},
            'tap': {'0': 4, '1': 5, '2': 6, '3': 7}
        }

        # 클래스별 데이터 개수를 저장할 딕셔너리 초기화
        self.class_counts = {gesture: {class_name: 0 for class_name in classes}
                             for gesture, classes in self.label_map.items()}

        self._load_data()
        self._calculate_normalization()
        self.on_epoch_end()

    def _load_data(self):
        for gesture in self.gesture_types:
            for class_name, class_id in self.label_map[gesture].items():
                acc_dir = os.path.join(self.root_dir, gesture, class_name, 'acc')
                audio_dir = os.path.join(self.root_dir, gesture, class_name, 'audio')

                if not os.path.exists(acc_dir) or not os.path.exists(audio_dir):
                    print(f"Directory not found: {acc_dir} or {audio_dir}")
                    continue

                acc_files = sorted([f for f in os.listdir(acc_dir) if f.endswith('.csv')])
                audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

                for acc_file, audio_file in zip(acc_files, audio_files):
                    acc_path = os.path.join(acc_dir, acc_file)
                    audio_path = os.path.join(audio_dir, audio_file)

                    # 가속도계 데이터 로드
                    acc_df = pd.read_csv(acc_path)[['x', 'y', 'z']]
                    acc_data = acc_df.values  # [time, 3]

                    # 오디오 데이터 로드 및 밴드패스 필터 적용
                    try:
                        audio_data, sample_rate = librosa.load(audio_path, sr=None)
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

                    self.data.append((acc_data_padded, audio_data))
                    self.sub_labels.append(class_id)

                    # 메인 레이블: 'tap' -> 1, 'slide' -> 0
                    main_label = 1 if gesture == 'tap' else 0
                    self.main_labels.append(main_label)
                    self.audio_lengths.append(audio_length)

                    # 클래스별 데이터 개수 증가
                    self.class_counts[gesture][class_name] += 1

        if self.indices is not None:
            self.data = [self.data[i] for i in self.indices]
            self.main_labels = [self.main_labels[i] for i in self.indices]
            self.sub_labels = [self.sub_labels[i] for i in self.indices]
            self.audio_lengths = [self.audio_lengths[i] for i in self.indices]

    def _calculate_normalization(self):
        # 전체 데이터에 대한 평균과 표준편차 계산
        acc_data_all = np.concatenate([d[0] for d in self.data], axis=0)  # [total_time, 3]
        audio_data_all = np.concatenate([d[1] for d in self.data])  # [total_time]

        self.acc_mean = acc_data_all.mean(axis=0)  # [3]
        self.acc_std = acc_data_all.std(axis=0)  # [3]
        self.audio_mean = audio_data_all.mean()  # scalar
        self.audio_std = audio_data_all.std()  # scalar

        print(f'acc mean : {self.acc_mean}')
        print(f'acc std : {self.acc_std}')
        print(f'audio mean : {self.audio_mean}')
        print(f'audio std : {self.audio_std}')

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_main_labels = self.main_labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_sub_labels = self.sub_labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_audio_lengths = self.audio_lengths[idx * self.batch_size: (idx + 1) * self.batch_size]

        acc_batch = [d[0] for d in batch_data]
        audio_batch = [d[1] for d in batch_data]

        # 배치 내 최대 길이 찾기
        max_length = max([len(a) for a in audio_batch])

        # 가속도계 데이터 패딩 또는 트렁케이션
        acc_padded = np.array([
            np.pad(a, ((0, max_length - a.shape[0]), (0, 0)), 'constant') if a.shape[0] < max_length else a[:max_length,
                                                                                                          :]
            for a in acc_batch
        ])  # [batch, max_length, 3]

        # 오디오 데이터 패딩 또는 트렁케이션
        audio_padded = np.array([
            np.pad(a, (0, max_length - len(a)), 'constant') if len(a) < max_length else a[:max_length]
            for a in audio_batch
        ])  # [batch, max_length]

        # 정규화
        acc_padded = (acc_padded - self.acc_mean) / self.acc_std  # [batch, max_length, 3]
        audio_padded = (audio_padded - self.audio_mean) / self.audio_std  # [batch, max_length]

        # 오디오 데이터 차원 추가
        audio_padded = np.expand_dims(audio_padded, axis=2)  # [batch, max_length, 1]

        # 레이블 변환 및 형식 수정
        main_labels = np.array(batch_main_labels, dtype=np.float32)  # [batch]
        sub_labels = np.array(batch_sub_labels, dtype=np.int32)  # [batch]

        return {'accel_input': acc_padded, 'audio_input': audio_padded}, {'main_output': main_labels,
                                                                          'sub_output': sub_labels}

    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.data, self.main_labels, self.sub_labels, self.audio_lengths))
            random.shuffle(combined)
            self.data, self.main_labels, self.sub_labels, self.audio_lengths = zip(*combined)

    def display_class_counts(self):
        print("클래스별 데이터 개수:")
        for gesture, classes in self.class_counts.items():
            print(f"Gesture: {gesture}")
            for class_name, count in classes.items():
                print(f"  Class '{class_name}': {count}개")
