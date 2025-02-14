"""
model.py
---------
This module defines a ModelFactory class that offers multiple model options.
It now uses the updated torchvision API with the 'weights' parameter.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet101_Weights, ResNet50_Weights

class ModelFactory:
    @staticmethod
    def get_resnet101(pretrained=True, output_dim=1):
        # Use the new 'weights' parameter instead of 'pretrained'
        weights = ResNet101_Weights.DEFAULT if pretrained else None
        model = models.resnet101(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        return model

    @staticmethod
    def get_resnet50(pretrained=True, output_dim=1):
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        return model

    @staticmethod
    def get_model(model_name="resnet101", pretrained=True, output_dim=1):
        if model_name.lower() == "resnet101":
            return ModelFactory.get_resnet101(pretrained, output_dim)
        elif model_name.lower() == "resnet50":
            return ModelFactory.get_resnet50(pretrained, output_dim)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

