from modelManager import modelManager
from torch.utils.data import DataLoader
from prepareData.dataloader import GestureDataset, collate_fn

# Load datasets
gesture_types = ['slide', 'tap']
print(f'init gesture types => {gesture_types}')
test_dataset = GestureDataset(root_dir='data/test', gesture_types=gesture_types)


# Create data loaders
batchSize = 32
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, collate_fn=collate_fn)

# test the model
trainer = modelManager(modelPath="pths/gesture_transformer_epoch82.pth")
f1 = trainer.test_model(test_loader)

