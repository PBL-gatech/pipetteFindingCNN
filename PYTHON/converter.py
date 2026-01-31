import os
import torch
from model import get_regression_model  # Your custom regression model builder


def convert_to_onnx(checkpoint_path, onnx_path, model_name="mobilenetv3_large_100",
                    output_dim=1, opset=11, two_head=None, img_size=224):
    """
    Export a trained checkpoint to ONNX.

    Args:
        checkpoint_path: path to the .pth checkpoint
        onnx_path: destination .onnx path
        model_name: timm backbone name
        output_dim: number of regression outputs
        opset: ONNX opset version
        two_head: whether to use the multi-head architecture (default: output_dim>=3)
        img_size: input resolution expected by the model
    """
    use_two_head = two_head if two_head is not None else output_dim >= 3
    model = get_regression_model(model_name=model_name, pretrained=False, output_dim=output_dim, two_head=use_two_head)
    
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Regression model successfully exported to {onnx_path}")


if __name__ == "__main__":
    checkpoint_name = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\training\train-mobilenetv3_large_100-20260121_022553"
    model_name = "best_model_focus_epoch38.pth"
    onnx_name = "WaynesBoroPipetteFocuserNet.onnx"
    output_dim = 3  # update to match your training configuration
    # Update these paths as appropriate for your environment.
 
    checkpoint_path = os.path.join(checkpoint_name, model_name)
    onnx_path = os.path.join(os.path.dirname(checkpoint_path), onnx_name)
    
    convert_to_onnx(checkpoint_path, onnx_path, model_name="mobilenetv3_large_100", output_dim=output_dim)
