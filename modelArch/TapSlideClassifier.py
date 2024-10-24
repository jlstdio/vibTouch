import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


class TapSlideClassifier(nn.Module):
    def __init__(self, num_subclasses=4):
        super(TapSlideClassifier, self).__init__()
        # 공유된 CNN 백본 (예: ResNet18 사용)
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 백본의 마지막 FC 레이어 제거

        # 첫 번째 분류기: Tap vs Slide (이진 분류)
        self.classifier_main = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # 출력: 1 (Tap vs Slide)
        )

        # 두 번째 분류기: 서브클래스 분류 (Tap1-4 또는 Slide1-4)
        self.classifier_sub = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_subclasses * 2)  # Tap1-4와 Slide1-4를 구분하기 위해 8개 클래스
        )

    def forward(self, x):
        features = self.backbone(x)

        main_output = self.classifier_main(features).squeeze(1)  # [batch_size]
        sub_output = self.classifier_sub(features)  # [batch_size, 8]

        return main_output, sub_output


# 모델 초기화
model = TapSlideClassifier(num_subclasses=4)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 손실 함수
criterion_main = nn.BCEWithLogitsLoss()
criterion_sub = nn.CrossEntropyLoss()

# 옵티마이저
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# 학습 루프 예제
def train(model, dataloader, optimizer, criterion_main, criterion_sub, device):
    model.train()
    running_loss = 0.0
    for inputs, labels_main, labels_sub in dataloader:
        inputs = inputs.to(device)
        labels_main = labels_main.to(device).float()
        labels_sub = labels_sub.to(device)

        optimizer.zero_grad()

        outputs_main, outputs_sub = model(inputs)

        loss_main = criterion_main(outputs_main, labels_main)
        loss_sub = criterion_sub(outputs_sub, labels_sub)
        loss = loss_main + loss_sub  # 가중치를 조절할 수 있음

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


# 평가 루프 예제
def evaluate(model, dataloader, criterion_main, criterion_sub, device):
    model.eval()
    running_loss = 0.0
    correct_main = 0
    correct_sub = 0
    total = 0
    with torch.no_grad():
        for inputs, labels_main, labels_sub in dataloader:
            inputs = inputs.to(device)
            labels_main = labels_main.to(device).float()
            labels_sub = labels_sub.to(device)

            outputs_main, outputs_sub = model(inputs)

            loss_main = criterion_main(outputs_main, labels_main)
            loss_sub = criterion_sub(outputs_sub, labels_sub)
            loss = loss_main + loss_sub

            running_loss += loss.item() * inputs.size(0)

            # 예측
            preds_main = torch.sigmoid(outputs_main) >= 0.5
            correct_main += (preds_main.int() == labels_main.int()).sum().item()

            preds_sub = torch.argmax(outputs_sub, dim=1)
            correct_sub += (preds_sub == labels_sub).sum().item()

            total += inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy_main = correct_main / total
    accuracy_sub = correct_sub / total
    return epoch_loss, accuracy_main, accuracy_sub

# 예제 학습 및 평가
# for epoch in range(num_epochs):
#     train_loss = train(model, train_loader, optimizer, criterion_main, criterion_sub, device)
#     val_loss, val_acc_main, val_acc_sub = evaluate(model, val_loader, criterion_main, criterion_sub, device)
#     print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Acc Main: {val_acc_main}, Val Acc Sub: {val_acc_sub}")
