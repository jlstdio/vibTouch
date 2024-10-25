import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
# import wandb
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# state_dict에서 'module' 제거하는 함수
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class modelManager:
    def __init__(self, modelPath=None, model=None, device=None, enableParallel=True, lr=1e-4):
        self.model = model
        self.device = device

        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

        if not torch.cuda.is_available():
            self.model.to('cpu')

        if modelPath is not None:
            prefixRemoved = remove_module_prefix(torch.load(modelPath, map_location=torch.device('cpu')))
            self.model.load_state_dict(prefixRemoved)

        # Wrap the model with DataParallel
        if torch.cuda.device_count() > 1 and enableParallel:
           self.model = nn.DataParallel(self.model)

        # 손실 함수
        self.criterion_main = nn.BCEWithLogitsLoss()
        self.criterion_sub = nn.CrossEntropyLoss()

        # 옵티마이저
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_model(self, train_loader, val_loader, num_epochs):

        f = open("trainEvolution.txt", "w")
        f.write('starting seq \n')
        f.close()

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for acc, audio, main_label, sub_label in train_loader:
                acc = acc.to(self.device).float()
                audio = audio.to(self.device).float()
                main_label = main_label.to(self.device).float()
                sub_label = sub_label.to(self.device)

                self.optimizer.zero_grad()
                outputs_main, outputs_sub = self.model(acc, audio)

                loss_main = self.criterion_main(outputs_main, main_label)
                loss_sub = self.criterion_sub(outputs_sub, sub_label)
                loss = loss_main + loss_sub  # 가중치를 조절할 수 있음

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * acc.size(0)

            train_loss = running_loss / len(train_loader.dataset)

            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for acc, audio, main_label, sub_label in val_loader:
                    acc = acc.to(self.device).float()
                    audio = audio.to(self.device).float()
                    main_label = main_label.to(self.device).float()
                    sub_label = sub_label.to(self.device)

                    outputs_main, outputs_sub = self.model(acc, audio)

                    loss_main = self.criterion_main(outputs_main, main_label)
                    loss_sub = self.criterion_sub(outputs_sub, sub_label)
                    loss = loss_main + loss_sub  # 가중치를 조절할 수 있음
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            f = open("trainEvolution.txt", "a")
            f.write(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
            f.close()

            # wandb.log({"Training loss": train_loss})
            # wandb.log({"Validation loss": val_loss})

            torch.save(self.model.state_dict(), f'pths/gesture_transformer_epoch{epoch}.pth')

    def print_model(self):
        return self.model

    def test_model(self, test_loader, matrix=False):
        self.model.eval()
        test_loss = 0.0
        all_main_preds = []
        all_main_labels = []
        all_sub_preds = []
        all_sub_labels = []

        with torch.no_grad():
            for acc_data, audio_data, main_labels, sub_labels in test_loader:
                # 데이터 장치로 이동
                acc_data = acc_data.to(self.device).float()
                audio_data = audio_data.to(self.device).float()
                main_labels = main_labels.to(self.device).float()
                sub_labels = sub_labels.to(self.device).long()

                # 모델 예측
                outputs_main, outputs_sub = self.model(acc_data, audio_data)

                # 손실 계산
                loss_main = self.criterion_main(outputs_main.squeeze(), main_labels)
                loss_sub = self.criterion_sub(outputs_sub, sub_labels)
                loss = loss_main + loss_sub
                test_loss += loss.item()

                # Main 예측 (Tap vs Slide)
                preds_main = torch.sigmoid(outputs_main.squeeze()) >= 0.5
                preds_main = preds_main.int()
                all_main_preds.append(preds_main.cpu().numpy())
                all_main_labels.append(main_labels.cpu().numpy())

                # Sub 예측 (Tap1-4, Slide1-4)
                preds_sub = torch.argmax(outputs_sub, dim=1)
                all_sub_preds.append(preds_sub.cpu().numpy())
                all_sub_labels.append(sub_labels.cpu().numpy())

        # 평균 손실 계산
        test_loss /= len(test_loader)

        # NumPy 배열로 변환
        all_main_preds = np.concatenate(all_main_preds)
        all_main_labels = np.concatenate(all_main_labels)
        all_sub_preds = np.concatenate(all_sub_preds)
        all_sub_labels = np.concatenate(all_sub_labels)

        # F1 스코어 계산
        f1_main = f1_score(all_main_labels, all_main_preds, average='binary')
        f1_sub = f1_score(all_sub_labels, all_sub_preds, average='weighted')

        # Sub를 Slide와 Tap으로 분리하여 추가 F1 스코어 계산
        # Slide: sub_label < 4
        # Tap: sub_label >= 4
        sub_labels_slide = (all_sub_labels < 4).astype(int)
        sub_labels_tap = (all_sub_labels >= 4).astype(int)
        sub_preds_slide = (all_sub_preds < 4).astype(int)
        sub_preds_tap = (all_sub_preds >= 4).astype(int)

        f1_sub_slide = f1_score(sub_labels_slide, sub_preds_slide, average='binary')
        f1_sub_tap = f1_score(sub_labels_tap, sub_preds_tap, average='binary')

        # 결과 출력
        print(f"Test Loss: {test_loss:.4f}, Main F1 Score: {f1_main:.4f}, Sub F1 Score: {f1_sub:.4f}")
        print(f"Sub Slide F1 Score: {f1_sub_slide:.4f}, Sub Tap F1 Score: {f1_sub_tap:.4f}")

        if matrix:
            # Main Confusion Matrix
            cm_main = confusion_matrix(all_main_labels, all_main_preds)
            cm_main_normalized = cm_main.astype('float') / cm_main.sum(axis=1)[:, np.newaxis]
            fig_main, ax_main = plt.subplots()
            disp_main = ConfusionMatrixDisplay(confusion_matrix=cm_main_normalized, display_labels=['Slide', 'Tap'])
            disp_main.plot(cmap=plt.cm.Blues, ax=ax_main)
            plt.title('Main Confusion Matrix')
            plt.savefig('confusion_matrix_main.png')
            plt.close(fig_main)

            # Sub Confusion Matrix
            cm_sub = confusion_matrix(all_sub_labels, all_sub_preds)
            cm_sub_normalized = cm_sub.astype('float') / cm_sub.sum(axis=1)[:, np.newaxis]
            fig_sub, ax_sub = plt.subplots(figsize=(10, 10))  # 크기 조절 가능
            labels_sub = ['Slide0', 'Slide1', 'Slide2', 'Slide3', 'Tap1', 'Tap2', 'Tap3', 'Tap4']
            disp_sub = ConfusionMatrixDisplay(confusion_matrix=cm_sub_normalized, display_labels=labels_sub)
            disp_sub.plot(cmap=plt.cm.Blues, ax=ax_sub)
            plt.title('Sub Confusion Matrix')
            plt.savefig('confusion_matrix_sub.png')
            plt.close(fig_sub)

            # Sub Slide Confusion Matrix
            cm_sub_slide = confusion_matrix(sub_labels_slide, sub_preds_slide)
            cm_sub_slide_normalized = cm_sub_slide.astype('float') / cm_sub_slide.sum(axis=1)[:, np.newaxis]
            fig_sub_slide, ax_sub_slide = plt.subplots()
            disp_sub_slide = ConfusionMatrixDisplay(confusion_matrix=cm_sub_slide_normalized, display_labels=['Not Slide', 'Slide'])
            disp_sub_slide.plot(cmap=plt.cm.Blues, ax=ax_sub_slide)
            plt.title('Sub Slide Confusion Matrix')
            plt.savefig('confusion_matrix_sub_slide.png')
            plt.close(fig_sub_slide)

            # Sub Tap Confusion Matrix
            cm_sub_tap = confusion_matrix(sub_labels_tap, sub_preds_tap)
            cm_sub_tap_normalized = cm_sub_tap.astype('float') / cm_sub_tap.sum(axis=1)[:, np.newaxis]
            fig_sub_tap, ax_sub_tap = plt.subplots()
            disp_sub_tap = ConfusionMatrixDisplay(confusion_matrix=cm_sub_tap_normalized, display_labels=['Not Tap', 'Tap'])
            disp_sub_tap.plot(cmap=plt.cm.Blues, ax=ax_sub_tap)
            plt.title('Sub Tap Confusion Matrix')
            plt.savefig('confusion_matrix_sub_tap.png')
            plt.close(fig_sub_tap)

        return f1_main, f1_sub, f1_sub_slide, f1_sub_tap
