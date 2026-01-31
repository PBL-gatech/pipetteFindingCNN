#!/usr/bin/env python
import json
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


class Inferencer:
    def __init__(self, model_path=None, stats_path=None, img_size=224):
        cur_dir = Path(__file__).parent.absolute()
        self.model_path = model_path or os.path.join(cur_dir, "python_models", "regression_model2.onnx")

        # ONNX Runtime session
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        try:
            self.output_dim = self.session.get_outputs()[0].shape[1] or 1
        except Exception:
            self.output_dim = None

        # Image preprocessing parameters
        self.img_size = img_size
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Target normalization parameters (optional; saved during training)
        stats_candidate = stats_path or os.path.join(os.path.dirname(self.model_path), "target_normalization.json")
        self.target_stats = None
        self.target_cols = None
        if os.path.isfile(stats_candidate):
            with open(stats_candidate, "r") as f:
                self.target_stats = json.load(f)
            self.target_cols = self.target_stats.get("columns")

    def preprocess(self, img):
        """Resize, normalize, and convert BGR image to CHW tensor batch."""
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_normalized = (img_float - self.img_mean) / self.img_std
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        return img_batch

    def predict(self, img):
        """Run inference and return output vector (de-normalized if stats are available)."""
        input_tensor = self.preprocess(img)
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.4f} seconds")
        output_array = outputs[0].reshape(-1)
        if self.target_stats is not None:
            mean = np.array(self.target_stats["mean"], dtype=np.float32)
            std = np.array(self.target_stats["std"], dtype=np.float32)
            output_array = output_array * std + mean
        return output_array

    def get_pipette_focus_value(self, img):
        """Return only the defocus prediction (microns)."""
        preds = self.predict(img)
        if preds.size >= 3:
            return float(preds[2])
        return float(preds[0])

    def get_pipette_pose(self, img):
        """Return predicted pose dictionary using saved column order when available."""
        preds = self.predict(img)
        if self.target_cols:
            return {name: float(preds[i]) for i, name in enumerate(self.target_cols) if i < preds.size}
        if preds.size >= 3:
            return {"pipette_x_px": float(preds[0]), "pipette_y_px": float(preds[1]), "defocus_microns": float(preds[2])}
        if preds.size == 2:
            return {"pipette_x_px": float(preds[0]), "pipette_y_px": float(preds[1])}
        return {"defocus_microns": float(preds[0])}


if __name__ == '__main__':
    focuser = Inferencer()

    cur_dir = Path(__file__).parent.absolute()
    image_path = os.path.join(cur_dir, "example_images", "0_focus.png")

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        raise SystemExit(1)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        raise SystemExit(1)

    pose = focuser.get_pipette_pose(img)
    focus_value = pose.get("defocus_microns", 0.0)
    if "pipette_x_px" in pose and "pipette_y_px" in pose:
        print(f"Predicted pipette pose: x={pose['pipette_x_px']:.1f}px, y={pose['pipette_y_px']:.1f}px, z={focus_value:.2f} microns")
        label = f"Pose: ({pose['pipette_x_px']:.1f}, {pose['pipette_y_px']:.1f}), z={focus_value:.2f} um"
    else:
        print(f"Predicted pipette focus value: {focus_value:.2f} microns")
        label = f"Focus: {focus_value:.2f} um"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Pipette Focus Inference", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
