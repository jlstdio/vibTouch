import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMWithCNNTransformer_slideDetection(nn.Module):
    def __init__(self):
        super(LSTMWithCNNTransformer_slideDetection, self).__init__()
        # CNN for accelerometer data (3 channels)
        self.acc_cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.25),
            # Removed MaxPool1d
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.25)
            # Removed MaxPool1d
        )

        # CNN for audio data (1 channel)
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.25),
            # Removed MaxPool1d
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.25)
            # Removed MaxPool1d
        )

        # LSTM for time series analysis
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)

        # Transformer for integrating features
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)

        # Final classifier
        self.fc = nn.Linear(128, 8)  # Assuming 8 classes as per label_map

    def forward(self, acc_data, audio_data):
        # Process accelerometer data
        acc_features = self.acc_cnn(acc_data.permute(0, 2, 1))  # Shape: [batch_size, channels, seq_length]

        # Process audio data
        audio_features = self.audio_cnn(audio_data.unsqueeze(1))  # Shape: [batch_size, 1, seq_length]

        # Concatenate features along the feature dimension
        combined_features = torch.cat((acc_features, audio_features), dim=1)
        combined_features = combined_features.permute(0, 2, 1)  # Shape: [batch_size, seq_length, feature_dim]

        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(combined_features)

        # Transformer processes the output of the LSTM
        transformed_features = self.transformer_encoder(lstm_out.permute(1, 0, 2))  # Shape: [seq_length, batch_size, feature_dim]

        # Average pooling across sequence length and classify
        pooled_features = transformed_features.mean(dim=0)
        output = self.fc(pooled_features)

        return output