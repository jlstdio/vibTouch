import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch


class GestureTransformerZeropadding(nn.Module):

    def __init__(self):
        super(GestureTransformerZeropadding, self).__init__()

        # Model parameters
        acc_input_size = 3  # x, y, z
        audio_input_size = 12800  # length of audio data
        nhead = 4  # Reduce number of heads
        hidden_dim = 256
        num_layers = 2
        num_classes = 8

        self.acc_embedding = nn.Linear(acc_input_size, hidden_dim)
        self.audio_embedding = nn.Linear(audio_input_size, hidden_dim)

        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, acc_data, audio_data):
        # Padding acc_data to match audio_data length
        batch_size, seq_len, _ = acc_data.size()
        padded_acc_data = torch.zeros(batch_size, 12800, 3).to(acc_data.device)
        padded_acc_data[:, :seq_len, :] = acc_data

        acc_embedded = self.acc_embedding(padded_acc_data)
        audio_embedded = self.audio_embedding(audio_data.unsqueeze(1))  # Add channel dimension and embed

        combined_embedded = acc_embedded + audio_embedded

        transformer_output = self.transformer_encoder(combined_embedded)

        output = self.fc(transformer_output.mean(dim=1))

        return output