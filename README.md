# Rotation Equivariance Testing Dataset

A dataset generator for testing rotation equivariance in neural networks, specifically focusing on segmentation tasks using simple geometric shapes.

## Overview

This project generates a controlled dataset of rotated ellipses along with their corresponding segmentation masks. The dataset is designed to evaluate how well neural networks can handle rotational transformations in image segmentation tasks.

## Dataset Details

- **Image Size**: 128 x 128 pixels
- **Number of Samples**: 100 images
- **Rotation Range**: 0° to 180° (uniformly distributed)
- **Shape Type**: Ellipses with major axis 40 pixels and minor axis 20 pixels
- **Format**: Grayscale PNG images

## Dataset Structure

```
data/
├── images/          # Contains the ellipse images
│   └── ellipse_*.png
└── masks/           # Contains the segmentation masks
    └── mask_*.png
```

## Usage

To generate the dataset, run:

```bash
python generate.py
```

This will:

1. Create the necessary directories
2. Generate 100 rotated ellipse images with their masks
3. Save the images and masks in