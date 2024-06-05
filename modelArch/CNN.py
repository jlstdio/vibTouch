import torch
import torch.nn as nn
import torch.nn.functional as F


class GestureCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(GestureCNN, self).__init__()

        # Convolutional layers for accelerometer data
        self.acc_conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.acc_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.acc_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Convolutional layers for audio data
        self.audio_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.audio_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.audio_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 40 + 32 * 6400, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, acc_data, audio_data):
        # Accelerometer data through CNN
        acc_data = acc_data.permute(0, 2, 1)  # (batch, channels, seq_len)
        acc_out = self.acc_pool(F.relu(self.acc_conv1(acc_data)))
        acc_out = self.acc_pool(F.relu(self.acc_conv2(acc_out)))
        acc_out = acc_out.view(acc_out.size(0), -1)  # Flatten

        # Audio data through CNN
        audio_data = audio_data.unsqueeze(1)  # (batch, 1, seq_len)
        audio_out = self.audio_pool(F.relu(self.audio_conv1(audio_data)))
        audio_out = self.audio_pool(F.relu(self.audio_conv2(audio_out)))
        audio_out = audio_out.view(audio_out.size(0), -1)  # Flatten

        # Combine and pass through fully connected layers
        combined_out = torch.cat((acc_out, audio_out), dim=1)
        combined_out = self.dropout(F.relu(self.fc1(combined_out)))
        output = self.fc2(combined_out)

        return output