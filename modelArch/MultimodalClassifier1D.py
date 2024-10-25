import torch
import torch.nn as nn


class MultimodalClassifier1D(nn.Module):
    def __init__(self, num_subclasses=4):
        super(MultimodalClassifier1D, self).__init__()
        # 가속도계용 1D CNN
        self.accel_conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),  # 입력 채널: 3
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 출력: [batch, 128, 1]
        )

        # 오디오용 1D CNN
        self.audio_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  # 입력 채널: 1
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 출력: [batch, 128, 1]
        )

        # 결합된 피처를 위한 분류기
        self.classifier_main = nn.Sequential(
            nn.Linear(128 * 2, 128),  # 가속도계 + 오디오 피처 결합
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Tap vs Slide
        )

        self.classifier_sub = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_subclasses * 2)  # 8개 클래스 (Tap1-4, Slide1-4)
        )

    def forward(self, accel, audio):
        """
        Args:
            accel: [batch_size, 3, L]
            audio: [batch_size, 1, L]
        Returns:
            main_output: [batch_size] (Tap vs Slide)
            sub_output: [batch_size, 8] (Tap1-4, Slide1-4)
        """
        accel_feat = self.accel_conv(accel).squeeze(-1)  # [batch, 128]
        audio_feat = self.audio_conv(audio).squeeze(-1)  # [batch, 128]

        combined = torch.cat((accel_feat, audio_feat), dim=1)  # [batch, 256]

        main_output = self.classifier_main(combined).squeeze(1)  # [batch]
        sub_output = self.classifier_sub(combined)  # [batch, 8]

        return main_output, sub_output
