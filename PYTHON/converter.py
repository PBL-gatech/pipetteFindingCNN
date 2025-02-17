import os
import torch
from model import ModelFactory  # Make sure this is in your PYTHON path


def convert_to_onnx(checkpoint_path, onnx_path, model_name="mobilenetv3", output_dim=1):
    # Instantiate the model exactly as used in training.
    model = ModelFactory.get_model(model_name, pretrained=False, output_dim=output_dim)
    
    # Load the saved state dictionary.
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode.

    # Create a dummy input tensor that matches the expected input dimensions.
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export the model to ONNX.
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,       # Export trained parameters
        opset_version=11,         # Adjust opset version if needed
        do_constant_folding=True, # Perform constant folding for optimization
        input_names=['input'],    # Name of the input node
        output_names=['output'],  # Name of the output node
        dynamic_axes={
            'input': {0: 'batch_size'},  # Allow variable batch sizes
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model successfully exported to {onnx_path}")





if __name__ == "__main__":
    # Path to the saved checkpoint (state dict) from your training script.
    checkpoint_path = os.path.join(r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\training\train-mobilenetv3-20250215_160856", "best_model_focus_22.pth")
    
    # Desired output path for the ONNX model.
    onnx_path = os.path.join(r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\training\train-mobilenetv3-20250215_160856", "pipette_focus_model.onnx")
    
    convert_to_onnx(checkpoint_path, onnx_path)
