# model.py
import torch.nn as nn
import timm

def get_regression_model(model_name="mobilenetv3_large_100", pretrained=True, output_dim=1):
    """
    Build a timm model for scalar regression.
    Prefer letting timm create the head with num_classes=output_dim;
    if that fails, fall back to a simple MLP head replacement.
    """
    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=output_dim)
    except Exception:
        # Fallback path mirrors the previous behavior
        model = timm.create_model(model_name, pretrained=pretrained)

        if hasattr(model, "classifier"):
            in_features = model.classifier[-1].in_features if isinstance(model.classifier, nn.Sequential) else model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim),
            )
        elif hasattr(model, "head"):
            in_features = model.head.in_features if hasattr(model.head, "in_features") else model.head.fc.in_features
            model.head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim),
            )
        else:
            model.reset_classifier(num_classes=output_dim)
        return model
