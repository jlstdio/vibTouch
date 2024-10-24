import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch


class GestureTransformerZeropadding(nn.Module):
    def __init__(self):
        super(GestureTransformerZeropadding, self).__init__()

        # Model parameters
        acc_input_size = 3  # x, y, z
        audio_input_size = 12800  # length of audio data
        nhead = 4
        hidden_dim = 256
        num_layers = 1
        num_classes = 8
        dropout_rate = 0.1  # Adding dropout rate

        # Embedding layers
        self.acc_embedding = nn.Linear(acc_input_size, hidden_dim)
        self.audio_embedding = nn.Linear(audio_input_size, hidden_dim)

        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True,
            dropout=dropout_rate,  # Apply dropout to each encoder layer
            norm_first=True  # Applying Pre-Normalization
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Interaction layers for cross-attention
        self.acc_to_audio_att = nn.Linear(hidden_dim, hidden_dim)
        self.audio_to_acc_att = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, acc_data, audio_data):
        # Padding acc_data to match audio_data length
        batch_size, seq_len, _ = acc_data.size()
        padded_acc_data = torch.zeros(batch_size, 12800, 3).to(acc_data.device)
        padded_acc_data[:, :seq_len, :] = acc_data

        # Embedding
        acc_embedded = self.acc_embedding(padded_acc_data)
        audio_embedded = self.audio_embedding(audio_data.unsqueeze(1))

        # Interaction between accelerometer and audio data
        acc_to_audio = self.acc_to_audio_att(acc_embedded)
        audio_to_acc = self.audio_to_acc_att(audio_embedded)

        combined_embedded = acc_embedded + acc_to_audio + audio_embedded + audio_to_acc

        # Transformer encoder
        transformer_output = self.transformer_encoder(combined_embedded)

        # Fully connected layer
        output = self.fc(transformer_output.mean(dim=1))

        return output
