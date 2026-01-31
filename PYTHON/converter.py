import os
import torch
import torch.nn as nn
from model import build_model


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.forward_export(images)


def convert_to_onnx(
    checkpoint_path,
    onnx_path,
    model_name="mobilenetv3_large_100",
    heatmap_sigma=2.0,
    heatmap_stride=4,
    lambda_xy=1.0,
    lambda_z=1.0,
    huber_beta=1.0,
    opset=11,
    img_size=224,
    pretrained=False,
):
    """
    Export a trained checkpoint to ONNX with outputs [x_px, y_px, defocus_microns].
    """
    model = build_model(
        model_name=model_name,
        pretrained=pretrained,
        heatmap_sigma=heatmap_sigma,
        heatmap_stride=heatmap_stride,
        lambda_xy=lambda_xy,
        lambda_z=lambda_z,
        huber_beta=huber_beta,
    )

    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = ExportWrapper(model)
    dummy_input = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model successfully exported to {onnx_path}")


if __name__ == "__main__":
    checkpoint_name = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\training\train-mobilenetv3_large_100-20260121_022553"
    model_name = "best_model_focus_epoch38.pth"
    onnx_name = "WaynesBoroPipetteFocuserNet.onnx"

    checkpoint_path = os.path.join(checkpoint_name, model_name)
    onnx_path = os.path.join(os.path.dirname(checkpoint_path), onnx_name)

    convert_to_onnx(checkpoint_path, onnx_path, model_name="mobilenetv3_large_100")
