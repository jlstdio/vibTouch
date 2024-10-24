import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PhaseAlignmentLayer(nn.Module):
    def __init__(self, delay):
        super(PhaseAlignmentLayer, self).__init__()
        self.delay = delay

    def forward(self, x):
        # 시간 지연을 모의하기 위해 입력 텐서를 시프트
        # delay가 양수이면 데이터를 오른쪽으로, 음수이면 왼쪽으로 시프트
        return torch.roll(x, shifts=self.delay, dims=1)

class TransformerWithCNNCrossAttention(nn.Module):
    def __init__(self):
        super(TransformerWithCNNCrossAttention, self).__init__()

        # Model parameters
        acc_input_size = 3  # x, y, z dimensions
        audio_input_size = 12800  # length of audio data
        hidden_dim = 256
        combined_dim = 512  # acc_embedded와 audio_embedded를 결합한 후의 차원
        nhead = 4
        num_layers = 1
        num_classes = 8

        self.acc_embedding = nn.Linear(acc_input_size, hidden_dim)
        self.audio_embedding = nn.Linear(audio_input_size, hidden_dim)
        self.dim_reduction = nn.Linear(combined_dim, hidden_dim)  # 차원 축소를 위한 선형 계층

        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, acc_data, audio_data):
        acc_embedded = self.acc_embedding(acc_data)
        audio_embedded = self.audio_embedding(audio_data)
        audio_embedded = audio_embedded.unsqueeze(1)

        # 임베딩 데이터 결합
        combined_embedded = torch.cat((acc_embedded, audio_embedded.expand(-1, acc_embedded.size(1), -1)), dim=-1)
        # 차원 축소
        combined_embedded = self.dim_reduction(combined_embedded)

        transformer_output = self.transformer_encoder(combined_embedded)
        output = self.fc(transformer_output.mean(dim=1))

        return output