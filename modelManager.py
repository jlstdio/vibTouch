import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from modelArch.RCNN import RCNN
from modelArch.Transformer_Upsampling import GestureTransformerUpsampling
from modelArch.Transformer_Zeropadding import GestureTransformerZeropadding
import torch.nn as nn


# state_dict에서 'module.' 제거하는 함수
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class modelManager:
    def __init__(self, modelPath=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.model = GestureTransformerUpsampling().to(self.device)
        self.model = GestureTransformerZeropadding().to(self.device)
        # self.model = RCNN().to(self.device)

        if not torch.cuda.is_available():
            self.model.to('cpu')

        if modelPath is not None:
            prefixRemoved = remove_module_prefix(torch.load(modelPath, map_location=torch.device('cpu')))
            self.model.load_state_dict(prefixRemoved)

        # Wrap the model with DataParallel
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def train_model(self, train_loader, val_loader, num_epochs):

        f = open("trainEvolution.txt", "w")
        f.write('starting seq \n')
        f.close()

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for acc_data, audio_data, labels in train_loader:
                acc_data, audio_data, labels = acc_data.to(self.device), audio_data.to(self.device), labels.to(
                    self.device)

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
                    acc_data, audio_data, labels = acc_data.to(self.device), audio_data.to(self.device), labels.to(
                        self.device)

                    outputs = self.model(acc_data, audio_data)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            f = open("trainEvolution.txt", "a")
            f.write(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
            f.close()

            torch.save(self.model.state_dict(), f'pths/gesture_transformer_epoch{epoch}.pth')

    def print_model(self):
        return self.model

    def test_model(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for acc_data, audio_data, labels in test_loader:
                acc_data, audio_data, labels = acc_data.to(self.device), audio_data.to(self.device), labels.to(
                    self.device)

                outputs = self.model(acc_data, audio_data)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        test_loss /= len(test_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Test Loss: {test_loss:.4f}, F1 Score: {f1:.4f}")

        return f1
