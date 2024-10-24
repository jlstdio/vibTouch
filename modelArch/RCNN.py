import torch
import torch.nn as nn
import torch.nn.functional as F


class RCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(RCNN, self).__init__()

        # Accelerometer CNN
        self.acc_cnn = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Audio CNN
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # RNN Layers
        self.rnn = nn.LSTM(input_size=64 * 10 + 64 * 1600, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, acc, audio):
        acc = self.acc_cnn(acc)
        acc = acc.view(acc.size(0), -1)

        audio = audio.unsqueeze(1)  # Add channel dimension
        audio = self.audio_cnn(audio)
        audio = audio.view(audio.size(0), -1)

        combined = torch.cat((acc, audio), dim=1).unsqueeze(1)  # Add sequence dimension

        rnn_out, _ = self.rnn(combined)
        rnn_out = rnn_out[:, -1, :]  # Use the last output

        output = self.fc(rnn_out)

        return output