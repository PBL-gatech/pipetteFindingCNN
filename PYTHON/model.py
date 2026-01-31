# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class HeatmapDepthModel(nn.Module):
    """
    Timm backbone -> heatmap head for x/y localization + z regression head.
    - Generates Gaussian target heatmaps inside the forward pass when targets are provided.
    - Decodes heatmap peaks (with local subpixel refinement) to x_px / y_px.
    - Stores z normalization stats as buffers and returns denormalized microns.
    """

    def __init__(
        self,
        model_name: str = "mobilenetv3_large_100",
        pretrained: bool = True,
        heatmap_sigma: float = 2.0,
        heatmap_stride: int = 4,
        lambda_z: float = 1.0,
        huber_beta: float = 1.0,
        heatmap_head_channels: int = 256,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True, out_indices=(-1,)
        )
        self.heatmap_sigma = float(heatmap_sigma)
        self.heatmap_stride = max(int(heatmap_stride), 1)
        self.lambda_z = float(lambda_z)
        self.huber_beta = float(huber_beta)

        in_ch = self.backbone.feature_info[-1]["num_chs"]
        mid_ch = max(heatmap_head_channels, 32)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=1),
        )

        self.z_pool = nn.AdaptiveAvgPool2d(1)
        self.z_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        # Persist z normalization inside checkpoints
        self.register_buffer("z_mean", torch.tensor(0.0), persistent=True)
        self.register_buffer("z_std", torch.tensor(1.0), persistent=True)

        self.z_loss_fn = nn.SmoothL1Loss(beta=self.huber_beta)
        self.heatmap_loss_fn = nn.MSELoss()

    @torch.no_grad()
    def set_z_stats(self, mean: float, std: float):
        std = max(float(std), 1e-6)
        self.z_mean.fill_(float(mean))
        self.z_std.fill_(std)

    def _normalize_z(self, z_microns: torch.Tensor) -> torch.Tensor:
        return (z_microns - self.z_mean) / self.z_std

    def _denormalize_z(self, z_norm: torch.Tensor) -> torch.Tensor:
        return z_norm * self.z_std + self.z_mean

    @staticmethod
    def _strides_from_shapes(input_hw, heatmap_hw):
        h_img, w_img = input_hw
        h_hm, w_hm = heatmap_hw
        stride_y = h_img / float(h_hm)
        stride_x = w_img / float(w_hm)
        return stride_x, stride_y

    def _make_target_heatmap(self, xy_px: torch.Tensor, heatmap_shape, input_hw):
        """
        xy_px: (B, 2) in resized image pixel coordinates.
        Returns Gaussian target heatmap shaped (B, 1, Hh, Wh).
        """
        b = xy_px.shape[0]
        h_hm, w_hm = heatmap_shape
        stride_x, stride_y = self._strides_from_shapes(input_hw, (h_hm, w_hm))
        cx = xy_px[:, 0] / stride_x
        cy = xy_px[:, 1] / stride_y

        yy = torch.arange(h_hm, device=xy_px.device, dtype=xy_px.dtype).view(1, h_hm, 1)
        xx = torch.arange(w_hm, device=xy_px.device, dtype=xy_px.dtype).view(1, 1, w_hm)
        cx = cx.view(b, 1, 1)
        cy = cy.view(b, 1, 1)

        heatmap = torch.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / (self.heatmap_sigma ** 2))
        return heatmap.unsqueeze(1)

    @staticmethod
    def _subpixel_offsets(heatmap: torch.Tensor, peak_x: torch.Tensor, peak_y: torch.Tensor, radius: int = 1):
        """
        Local weighted-average refinement in a fixed window around the argmax.
        Returns dx, dy (each shaped [B]) relative to integer peak.
        """
        b, _, h, w = heatmap.shape
        dx_list, dy_list = [], []
        for i in range(b):
            x0 = int(peak_x[i].item())
            y0 = int(peak_y[i].item())
            x1, x2 = max(0, x0 - radius), min(w - 1, x0 + radius)
            y1, y2 = max(0, y0 - radius), min(h - 1, y0 + radius)
            patch = heatmap[i, 0, y1 : y2 + 1, x1 : x2 + 1]
            weights = patch
            if weights.numel() == 0:
                dx_list.append(torch.tensor(0.0, device=heatmap.device, dtype=heatmap.dtype))
                dy_list.append(torch.tensor(0.0, device=heatmap.device, dtype=heatmap.dtype))
                continue
            coords_x = torch.arange(x1, x2 + 1, device=heatmap.device, dtype=heatmap.dtype)
            coords_y = torch.arange(y1, y2 + 1, device=heatmap.device, dtype=heatmap.dtype)
            grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing="ij")
            w_sum = weights.sum()
            if w_sum <= 1e-6:
                dx_list.append(torch.tensor(0.0, device=heatmap.device, dtype=heatmap.dtype))
                dy_list.append(torch.tensor(0.0, device=heatmap.device, dtype=heatmap.dtype))
                continue
            dx_list.append(((weights * grid_x).sum() / w_sum) - peak_x[i])
            dy_list.append(((weights * grid_y).sum() / w_sum) - peak_y[i])
        return torch.stack(dx_list), torch.stack(dy_list)

    def _decode_heatmap(self, heatmap: torch.Tensor, input_hw):
        """
        heatmap: (B, 1, Hh, Wh) after sigmoid
        Returns x_px, y_px tensors in resized image pixel coordinates.
        """
        b, _, h_hm, w_hm = heatmap.shape
        flat = heatmap.view(b, -1)
        max_idx = flat.argmax(dim=1)
        peak_y = (max_idx // w_hm).float()
        peak_x = (max_idx % w_hm).float()

        dx, dy = self._subpixel_offsets(heatmap, peak_x, peak_y, radius=1)

        stride_x, stride_y = self._strides_from_shapes(input_hw, (h_hm, w_hm))
        x_px = (peak_x + dx) * stride_x
        y_px = (peak_y + dy) * stride_y
        return x_px, y_px

    def forward(self, images: torch.Tensor, targets: dict | None = None, compute_loss: bool = False):
        feats = self.backbone(images)[0]
        heatmap_logits = self.heatmap_head(feats)
        input_hw = images.shape[-2:]
        target_h = max(1, input_hw[0] // self.heatmap_stride)
        target_w = max(1, input_hw[1] // self.heatmap_stride)
        if (heatmap_logits.shape[-2], heatmap_logits.shape[-1]) != (target_h, target_w):
            heatmap_logits = F.interpolate(
                heatmap_logits, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        heatmap = torch.sigmoid(heatmap_logits)

        z_norm_pred = self.z_head(self.z_pool(feats))
        z_microns = self._denormalize_z(z_norm_pred)

        x_px, y_px = self._decode_heatmap(heatmap, input_hw)

        output = {
            "x_px": x_px,
            "y_px": y_px,
            "defocus_microns": z_microns.squeeze(-1),
        }

        if compute_loss and targets is not None:
            xy_px = targets["xy"]
            z_gt = targets["z"].unsqueeze(-1)
            b, _, h_hm, w_hm = heatmap.shape
            target_heatmap = self._make_target_heatmap(xy_px, (h_hm, w_hm), input_hw)
            heatmap_loss = self.heatmap_loss_fn(heatmap, target_heatmap)

            z_norm_gt = self._normalize_z(z_gt)
            z_loss = self.z_loss_fn(z_norm_pred, z_norm_gt)

            total_loss = heatmap_loss + self.lambda_z * z_loss
            output.update(
                heatmap=heatmap,
                z_norm_pred=z_norm_pred.squeeze(-1),
                loss=total_loss,
                loss_heatmap=heatmap_loss,
                loss_z=z_loss,
            )

        return output


def build_model(
    model_name: str = "mobilenetv3_large_100",
    pretrained: bool = True,
    heatmap_sigma: float = 2.0,
    heatmap_stride: int = 4,
    lambda_z: float = 1.0,
    huber_beta: float = 1.0,
    heatmap_head_channels: int = 256,
):
    return HeatmapDepthModel(
        model_name=model_name,
        pretrained=pretrained,
        heatmap_sigma=heatmap_sigma,
        heatmap_stride=heatmap_stride,
        lambda_z=lambda_z,
        huber_beta=huber_beta,
        heatmap_head_channels=heatmap_head_channels,
    )
