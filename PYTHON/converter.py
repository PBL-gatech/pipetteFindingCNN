import os
import torch
from model import get_regression_model  # Your custom regression model builder

def convert_to_onnx(checkpoint_path, onnx_path, model_name="mobilenetv3_large_100", output_dim=1, opset=11):
    # Instantiate your regression model.
    # Note: We set pretrained=False because we'll load our custom checkpoint,
    # and we do not need any classifier postprocessing since this is a regression task.
    model = get_regression_model(model_name=model_name, pretrained=False, output_dim=output_dim)
    
    # Load the saved state dictionary from your checkpoint.
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode, as done in timm's exporter.

    # Create a dummy input tensor with the dimensions expected by the model.
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export the model to ONNX.
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,       # Export trained parameters.
        opset_version=opset,
        do_constant_folding=True, # Perform constant folding for optimization.
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},   # Allow variable batch sizes.
            'output': {0: 'batch_size'}
        }
    )
    print(f"Regression model successfully exported to {onnx_path}")

if __name__ == "__main__":
    # Update these paths as appropriate for your environment.
    checkpoint_path = os.path.join(r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\training\train-mobilenetv3_large_100-20250217_171822", "best_model_focus_epoch18.pth")
    onnx_path = os.path.join(r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\training\train-mobilenetv3_large_100-20250217_171822", "regression_model2.onnx")
    
    convert_to_onnx(checkpoint_path, onnx_path)
