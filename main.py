import torch
from torch.utils.data import DataLoader
from prepareData.dataloader import GestureDataset, collate_fn
from modelManager import modelManager

# Load datasets
gesture_types = ['slide', 'tap']
# gesture_types = ['tap']
print(f'init gesture types => {gesture_types}')
dataset = GestureDataset(root_dir='data/train', gesture_types=gesture_types)

# Split datasets into train, validation, and test sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
batchSize = 32
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False, collate_fn=collate_fn)

print(f'data loaded')

'''
# train_loader에서 클래스 분포 계산
def get_class_distribution(loader):
    labels = []
    for _, _, label in loader:
        labels.extend(label.cpu().numpy())
    label_counts = Counter(labels)
    return label_counts

train_class_distribution = get_class_distribution(val_loader)

# 클래스 이름 설정 (필요에 따라 변경)
class_names = ['Slide 3_0', 'Slide 3_1', 'Slide 3_2', 'Slide 3_3', 'Tap 0', 'Tap 1', 'Tap 2', 'Tap 3']

# 데이터 시각화
plt.figure(figsize=(10, 5))
plt.bar(class_names, [train_class_distribution[i] for i in range(len(class_names))], color='blue')
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Train Loader')
plt.xticks(rotation=45)
plt.show()
'''


# Train the model
num_epochs = 90
trainer = modelManager()
trainer.train_model(train_loader, val_loader, num_epochs)
