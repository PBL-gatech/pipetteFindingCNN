import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Step 1: Define the ModifiedResNet101 class
class ModifiedResNet101(nn.Module):
    def __init__(self):
        super(ModifiedResNet101, self).__init__()
        
        # Load the base ResNet101 model without pretrained weights
        self.resnet101 = models.resnet101(weights=None)
        
        # Modify the fully connected layer for regression
        num_features = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Sequential(
            nn.Linear(num_features, 512),  # First fully connected layer
            nn.ReLU(),
            nn.Linear(512, 3)              # Outputting x, y, z coordinates
        )

    def forward(self, x):
        return self.resnet101(x)

# Step 2: Load the model from the saved .pt file
model_path = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\models\models\modified_resnet101_regression.pt"  # Update to the correct path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = torch.load(model_path, map_location=device)
    print("Successfully loaded the full model.")
except Exception as e:
    print(f"Failed to load the model. Error: {e}")

# Set the model to evaluation mode
model = model.to(device)
model.eval()

# Step 3: Set up directories and ground truth files for testing
image_folder = r"C:\Users\sa-forest\GaTech Dropbox\Benjamin Magondu\YOLOretrainingdata\Pipette CNN Training Data\20191111"  # Update with the correct path
ground_truth_file = r"C:\Users\sa-forest\GaTech Dropbox\Benjamin Magondu\YOLOretrainingdata\Pipette CNN Training Data\20191111\20191111.txt"  # Update with the correct path

# New results directory
results_directory = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\models\test"
annotated_images_dir = os.path.join(results_directory, "annotated_images")

# Create results directories if they do not exist
os.makedirs(results_directory, exist_ok=True)
os.makedirs(annotated_images_dir, exist_ok=True)

# Load ground truth labels
ground_truth = pd.read_csv(ground_truth_file, sep="\t")
ground_truth_dict = {
    str(int(row['img'])): (row['x'], row['y'], row['z'])
    for _, row in ground_truth.iterrows()
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Step 4: Prediction, error calculation, and saving all annotated images
errors = {'dx': [], 'dy': [], 'dz': []}
predictions = []
ground_truths = []

for image_file in os.listdir(image_folder):
    if image_file.endswith('.png'):
        image_id = os.path.splitext(image_file)[0]
        
        if image_id in ground_truth_dict:
            # Load and preprocess the image
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path).convert('RGB')
            processed_image = transform(image).unsqueeze(0).to(device)
            
            # Predict position using the model
            with torch.no_grad():
                prediction = model(processed_image).cpu().numpy().flatten()

            # Get ground truth position
            gt_x, gt_y, gt_z = ground_truth_dict[image_id]
            ground_truth_values = np.array([gt_x, gt_y, gt_z])

            # Calculate errors (in microns)
            dx, dy, dz = (ground_truth_values - prediction) * 0.1 / 1.093
            errors['dx'].append(dx)
            errors['dy'].append(dy)
            errors['dz'].append(dz)

            # Store predictions and ground truth for further use
            predictions.append(prediction)
            ground_truths.append(ground_truth_values)

            # Annotate and save the image
            real_pos = ground_truth_values[:2]
            pred_pos = prediction[:2]
            
            # Transform coordinates to new image dimension (224 x 224)
            original_size = (1024, 1280)
            min_dimension = min(original_size)
            new_dimension = 224
            transformed_real_pos = [
                new_dimension * ((real_pos[0] + original_size[1] / 2) - (original_size[1] - min_dimension) / 2) / min_dimension,
                new_dimension * ((real_pos[1] + original_size[0] / 2) - (original_size[0] - min_dimension) / 2) / min_dimension
            ]
            transformed_pred_pos = [
                new_dimension * ((pred_pos[0] + original_size[1] / 2) - (original_size[1] - min_dimension) / 2) / min_dimension,
                new_dimension * ((pred_pos[1] + original_size[0] / 2) - (original_size[0] - min_dimension) / 2) / min_dimension
            ]

            # Draw markers
            marked_image = ImageDraw.Draw(image)
            marked_image.ellipse([transformed_real_pos[0] - 9, transformed_real_pos[1] - 9,
                                  transformed_real_pos[0] + 9, transformed_real_pos[1] + 9], outline='cyan', width=3)
            marked_image.ellipse([transformed_pred_pos[0] - 9, transformed_pred_pos[1] - 9,
                                  transformed_pred_pos[0] + 9, transformed_pred_pos[1] + 9], outline='red', width=3)

            # Save annotated image
            annotated_image_path = os.path.join(annotated_images_dir, f"annotated_{image_file}")
            image.save(annotated_image_path)

# Calculate metrics
x_errors = np.abs(errors['dx'])
y_errors = np.abs(errors['dy'])
z_errors = np.abs(errors['dz'])

x_error_mean = np.mean(x_errors)
y_error_mean = np.mean(y_errors)
z_error_mean = np.mean(z_errors)
x_std = np.std(x_errors)
y_std = np.std(y_errors)
z_std = np.std(z_errors)

print(f"Average Errors (microns):\ndx: {x_error_mean:.2f}\ndy: {y_error_mean:.2f}\ndz: {z_error_mean:.2f}")
print(f"Standard Deviations (microns):\ndx: {x_std:.2f}\ndy: {y_std:.2f}\ndz: {z_std:.2f}")

# Save summary statistics
summary_stats_path = os.path.join(results_directory, "summary_statistics.txt")
with open(summary_stats_path, "w") as f:
    f.write(f"Average Errors (microns):\ndx: {x_error_mean:.2f}\ndy: {y_error_mean:.2f}\ndz: {z_error_mean:.2f}\n")
    f.write(f"Standard Deviations (microns):\ndx: {x_std:.2f}\ndy: {y_std:.2f}\ndz: {z_std:.2f}\n")

# Step 6: Plot error histograms
plt.figure(figsize=(10, 12))
plt.subplot(3, 1, 1)
plt.hist(x_errors, bins=20, color='blue', alpha=0.7)
plt.xlabel('dx error (microns)')
plt.ylabel('Frequency')
plt.title('Histogram of dx Errors')

plt.subplot(3, 1, 2)
plt.hist(y_errors, bins=20, color='green', alpha=0.7)
plt.xlabel('dy error (microns)')
plt.ylabel('Frequency')
plt.title('Histogram of dy Errors')

plt.subplot(3, 1, 3)
plt.hist(z_errors, bins=20, color='red', alpha=0.7)
plt.xlabel('dz error (microns)')
plt.ylabel('Frequency')
plt.title('Histogram of dz Errors')

plt.tight_layout()

# Save histogram plot
histogram_path = os.path.join(results_directory, "error_histograms.png")
plt.savefig(histogram_path)

plt.show()
