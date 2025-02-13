# infer.py
import os
import torch
from PIL import Image
from torchvision import transforms
from model import ModelFactory

class Inferencer:
    def __init__(self, model_path, model_name="resnet101", device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ModelFactory.get_model(model_name, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def infer_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)       # shape: [1, 1]
            defocus_microns = output.item()         # single scalar
        return defocus_microns

    def run_inference(self, test_images_dir):
        for filename in os.listdir(test_images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(test_images_dir, filename)
                predicted_defocus = self.infer_image(image_path)
                print(f"{filename}: predicted defocus = {predicted_defocus:.3f} microns")


if __name__ == "__main__":
    model_path = "best_model_focus_10.pth"
    test_images_dir = "/path/to/test_images"

    inferencer = Inferencer(model_path=model_path, model_name="resnet101", device='cuda')
    inferencer.run_inference(test_images_dir)
