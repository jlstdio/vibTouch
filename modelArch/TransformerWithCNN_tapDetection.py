import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerWithCNN_tapDetection(nn.Module):
    def __init__(self):
        def __init__(self):
            super(TransformerWithCNN_tapDetection, self).__init__()
            # CNN for accelerometer data (3 channels)
            self.acc_cnn = nn.Sequential(
                nn.Conv1d(3, 32, kernel_size=5, stride=1, padding=2),
                # nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4),
                nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
                # nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4),
                # nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
                # nn.ReLU(),
                # nn.MaxPool1d(kernel_size=4, stride=4)
            )

            # CNN for audio data (1 channel)
            self.audio_cnn = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
                # nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4),
                nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
                # nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4),
                # nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
                # nn.ReLU(),
                # nn.MaxPool1d(kernel_size=4, stride=4)
            )

            # Transformer for integrating features from both CNNs
            self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
            self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)

            # Final classifier
            self.fc = nn.Linear(128, 8)  # Assuming 8 classes as per label_map

        def forward(self, acc_data, audio_data):
            # Process accelerometer data
            acc_features = self.acc_cnn(acc_data.permute(0, 2, 1))  # Shape: [batch_size, channels, seq_length]
            # print(acc_features.shape)

            # Process audio data
            audio_features = self.audio_cnn(audio_data.unsqueeze(1))  # Shape: [batch_size, 1, seq_length]
            # print(audio_features.shape)

            # Concatenate features along the feature dimension and prepare for Transformer
            combined_features = torch.cat((acc_features, audio_features), dim=1).permute(2, 0, 1)
            # print(combined_features.shape)

            # Transformer processes features
            transformed_features = self.transformer_encoder(combined_features)

            # Average pooling across sequence length and classify
            pooled_features = transformed_features.mean(dim=0)
            output = self.fc(pooled_features)

            return output