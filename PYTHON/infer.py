"""
infer.py
---------
This module defines an Inferencer class that encapsulates the inference logic.
It loads a trained model and performs predictions on test images.
"""

import os
import torch
from PIL import Image
from torchvision import transforms
from model import ModelFactory
from utils import Visualizer


class Inferencer:
    def __init__(self, model_path, model_name="resnet101", device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # Instantiate model architecture (without pretrained weights) and load state dict.
        self.model = ModelFactory.get_model(model_name, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Define transforms (same as used during training).
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def infer_image(self, image_path):
        """
        Preprocess the image, perform inference, and return predicted coordinates.
        """
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor).cpu().numpy().flatten()
        return prediction, image

    def run_inference(self, test_images_dir):
        """
        Runs inference on all images in the test directory.
        """
        for filename in os.listdir(test_images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(test_images_dir, filename)
                prediction, image = self.infer_image(image_path)
                print(f"Image: {filename} - Predicted coordinates: {prediction}")

                # For visualization: adjust predicted x,y if needed.
                # Here we assume the predicted x, y are in the coordinate space of the 224x224 image.
                pred_x = prediction[0] + 112
                pred_y = prediction[1] + 112

                # If ground truth is available, you can pass it as the first argument.
                Visualizer.visualize_prediction(image, (112, 112), (pred_x, pred_y))


if __name__ == "__main__":
    # Set paths (adjust these as needed).
    model_path = "best_model_epoch_10.pth"  # Replace with your trained model file.
    test_images_dir = "/path/to/test_images"  # Directory containing test images.

    inferencer = Inferencer(model_path=model_path, model_name="resnet101", device='cuda')
    inferencer.run_inference(test_images_dir)
