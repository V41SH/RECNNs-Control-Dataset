# Rotation Equivariance Testing Dataset

A dataset generator for testing rotation equivariance in neural networks, specifically focusing on segmentation tasks using simple geometric shapes.

## Dataset Details

- **Image Size**: 128 x 128 pixels
- **Number of Samples**: 100 images
- **Rotation Range**: 0° to 360° (uniformly distributed)
- **Shape Type**: Ellipses with major axis 40 pixels and minor axis 20 pixels, Arrows

## Usage

```bash
python generate.py
```
