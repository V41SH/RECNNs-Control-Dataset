# Re-import libraries due to code execution environment reset
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# Dataset parameters
IMAGE_SIZE = 128
NUM_SAMPLES = 100
ROTATION_ANGLES = np.linspace(0, 360, NUM_SAMPLES, endpoint=False)
ELLIPSE_AXES = (40, 20)  # Major and minor axis lengths
OUTPUT_DIR = "data/"

# Create directories
def ensure_dirs(base_path, shape_type):
    img_dir = os.path.join(base_path, f"{shape_type}_images")
    mask_dir = os.path.join(base_path, f"{shape_type}_masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    return img_dir, mask_dir

# Function to generate an ellipse or arrow image and mask
def generate_shape_image(angle_deg, image_size=128, shape="ellipse", axes=(40, 20)):
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    mask = np.zeros_like(img)
    center = (image_size // 2, image_size // 2)
    
    if shape == "ellipse":
        cv2.ellipse(mask, center, axes, angle_deg, 0, 360, 255, -1)
    elif shape == "arrow":
        arrow = np.array([
            [0, -30], [10, -10], [5, -10], [5, 30], [-5, 30], [-5, -10], [-10, -10]
        ], dtype=np.float32)
        theta = np.deg2rad(angle_deg)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotated_arrow = (arrow @ rotation_matrix.T).astype(np.int32) + np.array(center)
        cv2.fillPoly(mask, [rotated_arrow], 255)
    else:
        raise ValueError("Unsupported shape type. Use 'ellipse' or 'arrow'.")

    img = cv2.GaussianBlur(mask, (11, 11), 0)
    return img, mask

# Generate datasets for both ellipse and arrow
for shape_type in ["ellipse", "arrow"]:
    img_dir, mask_dir = ensure_dirs(OUTPUT_DIR, shape_type)
    for i, angle in enumerate(ROTATION_ANGLES):
        img, mask = generate_shape_image(angle, shape=shape_type)
        cv2.imwrite(os.path.join(img_dir, f"{shape_type}_{i:03d}.png"), img)
        cv2.imwrite(os.path.join(mask_dir, f"{shape_type}_mask_{i:03d}.png"), mask)

# Display a sample arrow image and mask
sample_idx = 10
sample_img = cv2.imread(os.path.join(OUTPUT_DIR, "arrow_images", f"arrow_{sample_idx:03d}.png"), cv2.IMREAD_GRAYSCALE)
sample_mask = cv2.imread(os.path.join(OUTPUT_DIR, "arrow_masks", f"arrow_mask_{sample_idx:03d}.png"), cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.title("Arrow Image")
plt.imshow(sample_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Segmentation Mask")
plt.imshow(sample_mask, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
