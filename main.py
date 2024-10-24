import torch
import wandb
from torch.utils.data import DataLoader

from modelArch.TransformerWithCNN_typeClassifier import TransformerWithCNN_typeClassifier
from modelArch.CNN import CNN
from modelArch.TransformerWithCNN_slideDetection import LSTMWithCNNTransformer_slideDetection
from modelArch.TransformerWithCNN_tapDetection import TransformerWithCNN_tapDetection
from modelArch.TransformerWithCNN_CrossAttention import TransformerWithCNNCrossAttention
from modelArch.Transformer_Upsampling import GestureTransformerUpsampling
from modelArch.Transformer_Zeropadding import GestureTransformerZeropadding
from prepareData.dataloader import GestureDataset, collate_fn
from modelManager import modelManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# models
'''
GestureTransformerUpsampling().to(device)
GestureTransformerZeropadding().to(device)
RCNN().to(device)
TransformerWithCNN_tapDetection().to(device)
TransformerWithCNNCrossAttention().to(device)
TransformerWithCNN().to(device)
CNN().to(device)
LSTMWithCNNTransformer_slideDetection().to(device)
TransformerWithCNN_tapDetection.to(device)
'''

config = {
    "learning_rate": 0.0001,
    "architecture": "Transformer With CNN advanced",
    "dataset": ['tap', 'slide'],  # ['tap'] || ['slide']
    "modelType": "typeClassifier",  # "tapClassifier" || "slideClassifier"
    "model": TransformerWithCNN_tapDetection().to(device),
    "epochs": 600,
    "batchSize": 32,
}

wandb.init(
    project="FL",
    config=config
)

# Load datasets
gestureType = config['dataset']
print(f'init gesture types => {gestureType}')
dataset = GestureDataset(root_dir='data/train', gesture_types=gestureType, classifierStep=config['modelType'])

# Split datasets into train, validation, and test sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
batchSize = config['batchSize']
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False, collate_fn=collate_fn)

print(f'data loaded')

# Train the model
num_epochs = config['epochs']
trainer = modelManager(None, config['model'], device, True, config['learning_rate'])
trainer.train_model(train_loader, val_loader, num_epochs)

wandb.finish()