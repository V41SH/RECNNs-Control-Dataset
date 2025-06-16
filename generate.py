import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

IMAGE_SIZE = 128
NUM_SAMPLES = 100
ROTATION_ANGLES = np.linspace(0, 180, NUM_SAMPLES, endpoint=False)
ELLIPSE_AXES = (40, 20)  # Major and minor axis lengths
OUTPUT_DIR = "/data/ellipse_dataset"

# Create directories
images_dir = os.path.join(OUTPUT_DIR, "images")
masks_dir = os.path.join(OUTPUT_DIR, "masks")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

def generate_ellipse_image(angle_deg, image_size=128, axes=(40, 20)):
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    mask = np.zeros_like(img)

    center = (image_size // 2, image_size // 2)
    color = 255  # white ellipse

    # Draw ellipse on the mask
    cv2.ellipse(mask, center, axes, angle_deg, 0, 360, color, -1)

    # Optionally simulate intensity in image (slightly blurred version)
    img = cv2.GaussianBlur(mask, (11, 11), 0)

    return img, mask

# Generate dataset
for i, angle in enumerate(ROTATION_ANGLES):
    img, mask = generate_ellipse_image(angle)

    # Save files
    img_path = os.path.join(images_dir, f"ellipse_{i:03d}.png")
    mask_path = os.path.join(masks_dir, f"mask_{i:03d}.png")
    cv2.imwrite(img_path, img)
    cv2.imwrite(mask_path, mask)

# Show a sample
sample_idx = 10
sample_img = cv2.imread(os.path.join(images_dir, f"ellipse_{sample_idx:03d}.png"), cv2.IMREAD_GRAYSCALE)
sample_mask = cv2.imread(os.path.join(masks_dir, f"mask_{sample_idx:03d}.png"), cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.title("Ellipse Image")
plt.imshow(sample_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Segmentation Mask")
plt.imshow(sample_mask, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

