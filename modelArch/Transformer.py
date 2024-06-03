import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class GestureTransformer(nn.Module):
    def __init__(self, acc_input_size, audio_input_size, nhead, hidden_dim, num_layers, num_classes):
        super(GestureTransformer, self).__init__()

        self.acc_embedding = nn.Linear(acc_input_size, hidden_dim)
        self.audio_embedding = nn.Linear(audio_input_size, hidden_dim)

        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, acc_data, audio_data):
        acc_embedded = self.acc_embedding(acc_data)
        audio_embedded = self.audio_embedding(audio_data)

        # Expand audio_data to match the sequence length of acc_data
        audio_embedded = audio_embedded.unsqueeze(1).repeat(1, acc_embedded.size(1), 1)

        combined_embedded = acc_embedded + audio_embedded

        transformer_output = self.transformer_encoder(combined_embedded)

        output = self.fc(transformer_output.mean(dim=1))

        return output