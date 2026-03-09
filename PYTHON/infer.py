#!/usr/bin/env python
import os
import json
import time
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from utils import contrast_stretch_mu_2sigma_uint8

class Inferencer:
    def __init__(self, model_path: str | None = None, enable_contrast_stretch: bool = False, preprocess_config_path: str | None = None):
        # Determine the model path
        cur_dir = Path(__file__).parent.absolute()
        resolved_model_path = model_path or os.path.join(cur_dir, 'python_models', 'regression_model2.onnx')
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(resolved_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Input image size (as used in training transforms)
        self.imgSize = 224
        
        # Mean and std used in training (from Albumentations normalization)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.enable_contrast_stretch = bool(enable_contrast_stretch)
        if preprocess_config_path:
            self.load_preprocess_config(preprocess_config_path)

    def load_preprocess_config(self, preprocess_config_path: str) -> None:
        """
        Load preprocessing flags from a training run's preprocess_config.json.
        """
        with open(preprocess_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.enable_contrast_stretch = bool(cfg.get("enable_contrast_stretch", self.enable_contrast_stretch))

    def preprocess(self, img):
        """
        Preprocess the image:
          - Optional contrast stretching (mu +/- 2*sigma).
          - Resize to self.imgSize x self.imgSize.
          - Convert BGR (OpenCV) to RGB.
          - Scale pixel values to [0, 1].
          - Normalize with mean and std.
          - Rearrange dimensions from HWC to CHW and add a batch dimension.
        """
        working_img = img
        if working_img.ndim == 2:
            working_img = cv2.cvtColor(working_img, cv2.COLOR_GRAY2BGR)
        if self.enable_contrast_stretch:
            working_img = contrast_stretch_mu_2sigma_uint8(working_img)

        img_resized = cv2.resize(working_img, (self.imgSize, self.imgSize))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_normalized = (img_float - self.mean) / self.std
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        return img_batch

    def get_pipette_focus_value(self, img):
        """
        Runs the ONNX model on a preprocessed image and returns the defocus value in microns.
        """
        input_tensor = self.preprocess(img)
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.4f} seconds")
        output_array = outputs[0]
        norm_pred = float(output_array.flatten()[0])
        # pred_microns = self.denormalize_z(norm_pred)
        pred_microns = norm_pred
        return pred_microns

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
    
    focus_value = focuser.get_pipette_focus_value(img)
    print(f"Predicted pipette focus value: {focus_value:.2f} microns")
    
    label = f"Focus: {focus_value:.2f} µm"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Pipette Focus Inference", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
