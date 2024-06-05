import torch.nn as nn


class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, num_classes):
        super(GestureLSTM, self).__init__()

        self.lstm_acc = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm_audio = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, acc_data, audio_data):
        acc_out, _ = self.lstm_acc(acc_data)
        audio_out, _ = self.lstm_audio(audio_data.unsqueeze(-1))

        combined_out = acc_out + audio_out

        output = self.fc(combined_out.mean(dim=1))

        return output