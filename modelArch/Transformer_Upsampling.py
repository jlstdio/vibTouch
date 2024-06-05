import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class GestureTransformerUpsampling(nn.Module):
    def __init__(self):
        super(GestureTransformerUpsampling, self).__init__()

        # Model parameters
        acc_input_size = 3  # x, y, z
        audio_input_size = 12800  # length of audio data
        nhead = 4  # Reduce number of heads
        hidden_dim = 256  # Reduce hidden dimension
        num_layers = 3  # Reduce number of layers
        num_classes = 8

        self.acc_embedding = nn.Linear(acc_input_size, hidden_dim)
        self.audio_embedding = nn.Linear(audio_input_size, hidden_dim)

        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, num_classes)
        self.acc_upsample = nn.Upsample(size=12800, mode='linear', align_corners=True)

    def forward(self, acc_data, audio_data):
        acc_embedded = self.acc_embedding(acc_data)
        acc_embedded = self.acc_upsample(acc_embedded.permute(0, 2, 1)).permute(0, 2, 1)  # Upsample to match audio_data length

        audio_embedded = self.audio_embedding(audio_data.unsqueeze(1))  # Add channel dimension and embed

        combined_embedded = acc_embedded + audio_embedded

        transformer_output = self.transformer_encoder(combined_embedded)

        output = self.fc(transformer_output.mean(dim=1))

        return output