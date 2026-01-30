#!/usr/bin/env python
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


class Inferencer:
    def __init__(self):
        # Determine the model path
        cur_dir = Path(__file__).parent.absolute()
        model_path = os.path.join(cur_dir, 'python_models', 'regression_model2.onnx')

        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        try:
            self.output_dim = self.session.get_outputs()[0].shape[1] or 1
        except Exception:
            self.output_dim = None

        # Input image size (as used in training transforms)
        self.imgSize = 224

        # Mean and std used in training (from Albumentations normalization)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, img):
        """
        Preprocess the image:
          - Resize to self.imgSize x self.imgSize.
          - Convert BGR (OpenCV) to RGB.
          - Scale pixel values to [0, 1].
          - Normalize with mean and std.
          - Rearrange dimensions from HWC to CHW and add a batch dimension.
        """
        img_resized = cv2.resize(img, (self.imgSize, self.imgSize))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_normalized = (img_float - self.mean) / self.std
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        return img_batch

    def predict(self, img):
        """
        Runs the ONNX model on a preprocessed image and returns all outputs
        (supports 1–3 values: x_px, y_px, z_defocus).
        """
        input_tensor = self.preprocess(img)
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.4f} seconds")
        output_array = outputs[0]
        return output_array.reshape(-1)

    def get_pipette_focus_value(self, img):
        """
        Runs the ONNX model on a preprocessed image and returns the defocus value in microns.
        """
        preds = self.predict(img)
        return float(preds[0])

    def get_pipette_pose(self, img):
        """
        Returns a dict of predicted values. Keys depend on available outputs.
        """
        preds = self.predict(img)
        if preds.size >= 3:
            return {"x_px": float(preds[0]), "y_px": float(preds[1]), "defocus_microns": float(preds[2])}
        if preds.size == 2:
            return {"x_px": float(preds[0]), "y_px": float(preds[1])}
        return {"defocus_microns": float(preds[0])}


if __name__ == '__main__':
    focuser = Inferencer()

    # Adjust image path as needed
    cur_dir = Path(__file__).parent.absolute()
    image_path = os.path.join(cur_dir, "example_images", "0_focus.png")

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        exit(1)

    pose = focuser.get_pipette_pose(img)
    focus_value = pose.get("defocus_microns", 0.0)
    if "x_px" in pose and "y_px" in pose:
        print(f"Predicted pipette pose: x={pose['x_px']:.1f}px, y={pose['y_px']:.1f}px, z={focus_value:.2f} microns")
        label = f"Pose: ({pose['x_px']:.1f}, {pose['y_px']:.1f}), z={focus_value:.2f} Âµm"
    else:
        print(f"Predicted pipette focus value: {focus_value:.2f} microns")
        label = f"Focus: {focus_value:.2f} Âµm"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Pipette Focus Inference", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
