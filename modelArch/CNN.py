import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # CNN for accelerometer data (3 channels)
        self.acc_cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, stride=1, padding=2),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        # CNN for audio data (1 channel)
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        # Additional CNN layers to combine features from both modalities
        self.combine_cnn = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(51200, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # Assuming 8 classes as per label_map
        )

    def forward(self, acc_data, audio_data):
        # Process accelerometer data
        acc_features = self.acc_cnn(acc_data.permute(0, 2, 1))  # Shape: [batch_size, channels, seq_length]

        # Process audio data
        audio_features = self.audio_cnn(audio_data.unsqueeze(1))  # Shape: [batch_size, 1, seq_length]

        # Concatenate features along the channel dimension
        combined_features = torch.cat((acc_features, audio_features), dim=1)

        # Further combine features with additional CNN layers
        combined_features = self.combine_cnn(combined_features)

        # Flatten and pass through fully connected layers
        combined_features = combined_features.view(combined_features.size(0), -1)
        output = self.fc_layers(combined_features)

        return output