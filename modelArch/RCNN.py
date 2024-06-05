import torch
import torch.nn as nn
import torch.nn.functional as F

class RCNN(nn.Module):
    def __init__(self):
        super(RCNN, self).__init__()

        # Model parameters
        hidden_dim = 512
        num_layers = 2
        num_classes = 8

        # Convolutional layers for accelerometer data
        self.acc_conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.acc_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.acc_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.acc_upsample = nn.Upsample(size=12800, mode='linear', align_corners=True)

        # Convolutional layers for audio data
        self.audio_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.audio_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.audio_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, acc_data, audio_data):
        # Accelerometer data through CNN
        acc_data = acc_data.permute(0, 2, 1)  # (batch, channels, seq_len)
        acc_out = self.acc_pool(F.relu(self.acc_conv1(acc_data)))
        acc_out = self.acc_pool(F.relu(self.acc_conv2(acc_out)))
        acc_out = self.acc_upsample(acc_out)  # Upsample to match audio_out

        # Audio data through CNN
        audio_data = audio_data.unsqueeze(1)  # (batch, 1, seq_len)
        audio_out = self.audio_pool(F.relu(self.audio_conv1(audio_data)))
        audio_out = self.audio_pool(F.relu(self.audio_conv2(audio_out)))

        # Ensure both outputs have the same sequence length
        acc_out = acc_out.permute(0, 2, 1)  # (batch, seq_len, channels)
        audio_out = audio_out.permute(0, 2, 1)  # (batch, seq_len, channels)
        combined_out = torch.cat((acc_out, audio_out), dim=2)  # (batch, seq_len, 64)

        # LSTM layer
        lstm_out, _ = self.lstm(combined_out)

        # Fully connected layer
        output = self.fc(lstm_out[:, -1, :])

        return output