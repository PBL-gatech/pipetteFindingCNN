"""
model.py
---------
This module defines a ModelFactory class that offers multiple model options.
It now uses the updated torchvision API with the 'weights' parameter where applicable,
and TIMM for models not provided by torchvision.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet101_Weights, ResNet50_Weights

class ModelFactory:
    @staticmethod
    def get_resnet101(pretrained=True, output_dim=1):
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
    def get_efficientnet_lite(pretrained=True, output_dim=1):
        try:
            import timm
        except ImportError:
            raise ImportError("The timm library is required for efficientnet_lite. Please install it via 'pip install timm'.")
        # Create an EfficientNet-Lite model using timm.
        # Here we use "tf_efficientnet_lite0" as an example.
        model = timm.create_model("tf_efficientnet_lite0", pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        return model

    @staticmethod
    def get_mobilenetv3(pretrained=True, output_dim=1):
        # Using MobileNetV3 Large from torchvision.
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        # The classifier in MobileNetV3 is a Sequential; we replace it entirely.
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        return model

    @staticmethod
    def get_convnext(pretrained=True, output_dim=1):
        try:
            import timm
        except ImportError:
            raise ImportError("The timm library is required for convnext. Please install it via 'pip install timm'.")
        # Create a ConvNeXt Tiny model using timm.
        model = timm.create_model("convnext_tiny", pretrained=pretrained)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        return model

    @staticmethod
    def get_model(model_name="resnet101", pretrained=True, output_dim=1):
        model_name = model_name.lower()
        if model_name == "resnet101":
            return ModelFactory.get_resnet101(pretrained, output_dim)
        elif model_name == "resnet50":
            return ModelFactory.get_resnet50(pretrained, output_dim)
        elif model_name == "efficientnet_lite":
            return ModelFactory.get_efficientnet_lite(pretrained, output_dim)
        elif model_name == "mobilenetv3":
            return ModelFactory.get_mobilenetv3(pretrained, output_dim)
        elif model_name == "convnext":
            return ModelFactory.get_convnext(pretrained, output_dim)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")
