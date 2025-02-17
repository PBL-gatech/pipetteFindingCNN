# model_regression.py
import torch.nn as nn
import timm

def get_regression_model(model_name="mobilenetv3_large_100", pretrained=True, output_dim=1):
    # Create the base model with pretrained weights
    model = timm.create_model(model_name, pretrained=pretrained)

    # Replace the final classifier (the last 3 layers) with a regression head.
    # This example checks for common attributes.
    if hasattr(model, 'classifier'):
        # Get the in_features from the current classifier layer.
        # For example, if classifier is a Sequential block:
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
        else:
            in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    elif hasattr(model, 'head'):
        # For models where the classifier is stored in model.head
        in_features = model.head.in_features if hasattr(model.head, 'in_features') else model.head.fc.in_features
        model.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    else:
        # Fallback if neither attribute is found:
        model.reset_classifier(num_classes=output_dim)
    return model
