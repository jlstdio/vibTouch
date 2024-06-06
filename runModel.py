import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

model = SimpleModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 평가 모드로 설정
