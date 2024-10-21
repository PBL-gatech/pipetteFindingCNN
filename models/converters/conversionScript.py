import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
import os

# Define the path to the folder where CSV files are saved
export_folder = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\models\exported_network"

# Load the metadata for defining the model structure
layer_metadata_file = os.path.join(export_folder, 'layer_metadata.txt')
with open(layer_metadata_file, 'r') as f:
    layer_metadata = f.readlines()

# Define the modified ResNet101 model for regression (predicting x, y, z coordinates)
class ModifiedResNet101(nn.Module):
    def __init__(self):
        super(ModifiedResNet101, self).__init__()
        
        # Load the base ResNet101 model without pretrained weights
        self.resnet101 = models.resnet101(weights=None)
        
        # Modify the fully connected layer for regression
        # Using the same modifications as in the training script provided
        num_features = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Sequential(
            nn.Linear(num_features, 512),  # First fully connected layer
            nn.ReLU(),
            nn.Linear(512, 3)              # Outputting x, y, z coordinates
        )

    def forward(self, x):
        return self.resnet101(x)

# Instantiate the model
model = ModifiedResNet101()

# Function to load weights from exported CSV files
def load_weights_from_csv(model, export_folder):
    for i, (name, layer) in enumerate(model.named_children()):
        if isinstance(layer, models.resnet.ResNet):
            # Iterate over layers inside ResNet
            for j, (subname, sublayer) in enumerate(layer.named_children()):
                if isinstance(sublayer, (nn.Conv2d, nn.Linear)):
                    # Load weights if available
                    weight_file = os.path.join(export_folder, f'weights_layer_{j + 1}.csv')
                    if os.path.exists(weight_file):
                        weights = pd.read_csv(weight_file, header=None).values
                        sublayer.weight.data = torch.tensor(weights, dtype=torch.float32)
                    
                    # Load biases if available
                    bias_file = os.path.join(export_folder, f'bias_layer_{j + 1}.csv')
                    if os.path.exists(bias_file):
                        biases = pd.read_csv(bias_file, header=None).values
                        sublayer.bias.data = torch.tensor(biases, dtype=torch.float32)

# Load the exported weights into the PyTorch model
load_weights_from_csv(model, export_folder)

# Save the complete model in .pt format
torch.save(model, 'modified_resnet101_regression.pt')

# Optionally, you can also save just the model's state_dict if desired
# torch.save(model.state_dict(), 'modified_resnet101_state_dict.pt')

