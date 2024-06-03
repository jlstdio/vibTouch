import torch
import torch.optim as optim
from modelArch.Transformer import GestureTransformer
import torch.nn as nn


class trainManager:
    def __init__(self):
        # Model parameters
        acc_input_size = 3  # x, y, z
        audio_input_size = 12800  # length of audio data
        nhead = 8
        hidden_dim = 512
        num_layers = 6
        num_classes = 8

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self. model = GestureTransformer(acc_input_size, audio_input_size, nhead, hidden_dim, num_layers, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self. optimizer = optim.Adam(self.model.parameters(), lr=0.005)

    def train_model(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for acc_data, audio_data, labels in train_loader:
                acc_data, audio_data, labels = acc_data.to(self.device), audio_data.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(acc_data, audio_data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for acc_data, audio_data, labels in val_loader:
                    acc_data, audio_data, labels = acc_data.to(self.device), audio_data.to(self.device), labels.to(self.device)

                    outputs = self.model(acc_data, audio_data)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
