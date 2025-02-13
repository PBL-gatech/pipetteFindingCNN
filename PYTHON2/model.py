"""
model.py
---------
This module defines a ModelFactory class that offers multiple model options.
You can easily add additional architectures as needed.
"""

import torch.nn as nn
from torchvision import models


class ModelFactory:
    @staticmethod
    def get_resnet101(pretrained=True):
        """
        Returns a modified ResNet101 model for regression.
        The final fully-connected layer is replaced to output 3 values (x, y, z).
        """
        model = models.resnet101(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        return model

    @staticmethod
    def get_resnet50(pretrained=True):
        """
        Returns a modified ResNet50 model for regression.
        """
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        return model

    @staticmethod
    def get_model(model_name="resnet101", pretrained=True):
        """
        Returns a model based on the given model_name.
        Supported values: "resnet101", "resnet50"
        """
        if model_name.lower() == "resnet101":
            return ModelFactory.get_resnet101(pretrained)
        elif model_name.lower() == "resnet50":
            return ModelFactory.get_resnet50(pretrained)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")
