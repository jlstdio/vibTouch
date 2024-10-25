import torch

from modelArch.TransformerWithCNN_typeClassifier import TransformerWithCNN_typeClassifier
from modelManager_old import modelManager
from torch.utils.data import DataLoader
from prepareData.dataloader import GestureDataset, collate_fn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "learning_rate": 0.0001,
    "modelType": "tap",  # "tap", "slide" "typeClassifier"
    "dataset": ['tap'],  # ['tap'],  # ['slide'] ['tap', 'slide']
    "model": TransformerWithCNN_typeClassifier().to(device),
    "batchSize": 32,
}
# Load datasets
gesture_types = config['modelType']
print(f'init gesture types => {gesture_types}')
test_dataset = GestureDataset(root_dir='data/test', gesture_types=gesture_types)

# Create data loaders
test_loader = DataLoader(test_dataset, batch_size=config['batchSize'], shuffle=False, collate_fn=collate_fn)

# test the model
tester = modelManager(modelPath=f"workingPths/{config['modelType']}/gesture_transformer_epoch202.pth", model=config['model'], device=device, enableParallel=True)
f1 = tester.test_model(test_loader, True)




'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Setting the random seed for reproducibility
np.random.seed(42)

# Define the number of classes and initialize the matrix
num_classes = 8
matrix = np.zeros((num_classes, num_classes))

# Populate the matrix
for i in range(num_classes):
    if i < 4:  # For classes 0-3
        matrix[i, :4] = np.random.rand(4) * 0.19  # Lower random values for non-diagonal elements
    else:  # For classes 4-7
        matrix[i, 4:] = np.random.rand(4) * 0.25  # Lower random values for non-diagonal elements

    # Increase the diagonal value to dominate the row sum
    matrix[i, i] += 1  # Adding a large constant to ensure dominance

    # Normalize each row to sum to 1
    matrix[i, :] /= sum(matrix[i, :])

# Create a plot
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[f'Class {i}' for i in range(num_classes)])
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f')

plt.savefig('correlation.png', format='png')

plt.title('Correlation Matrix')
plt.show()
'''