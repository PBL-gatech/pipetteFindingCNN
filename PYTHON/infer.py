"""
infer.py
---------
This module defines an Inferencer class for performing defocus regression inference.
It loads a specified model checkpoint using the ModelFactory, applies the proper image
transformations, and outputs the predicted defocus (in microns) for each image in a folder.
"""

import os
import torch
from PIL import Image
from torchvision import transforms
from model import ModelFactory

class Inferencer:
    def __init__(self, model_path, model_name="resnet101", device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        # Load the model architecture with pretrained=False (since we're loading a checkpoint)
        self.model = ModelFactory.get_model(model_name, pretrained=False, output_dim=1)
        # Load checkpoint (make sure the checkpoint exists)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Checkpoint '{model_path}' not found.")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def infer_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)  # shape: [1, 1]
            defocus_microns = output.item()   # single scalar value
        return defocus_microns

    def run_inference(self, test_images_dir):
        if not os.path.isdir(test_images_dir):
            raise NotADirectoryError(f"Directory '{test_images_dir}' not found.")
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        for filename in os.listdir(test_images_dir):
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(test_images_dir, filename)
                predicted_defocus = self.infer_image(image_path)
                if predicted_defocus is not None:
                    print(f"{filename}: predicted defocus = {predicted_defocus:.3f} microns")
                else:
                    print(f"Skipping {filename} due to error.")

if __name__ == "__main__":
    # Specify your checkpoint path and test images directory
    model_path = "best_model_focus_10.pth"
    test_images_dir = "/path/to/test_images"
    
    inferencer = Inferencer(model_path=model_path, model_name="resnet101", device='cuda')
    inferencer.run_inference(test_images_dir)
