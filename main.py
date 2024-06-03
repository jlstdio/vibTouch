import torch
from torch.utils.data import DataLoader
from prepareData.dataloader import GestureDataset, collate_fn
from trainManager import trainManager

# Load datasets
gesture_types = ['slide', 'tap']
dataset = GestureDataset(root_dir='data/sliced', gesture_types=gesture_types)

# Split datasets into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Train the model
num_epochs = 100
trainer = trainManager()
trainer.train_model(train_loader, val_loader, num_epochs)