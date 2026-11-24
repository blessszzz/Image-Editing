# Performance Analysis of CPU and GPU Architectures for Image Editing: Color Grading, Masking, and Fusion

A comparative study of how CPU and GPU architectures perform on three common image-editing operations: **color grading**, **masking**, and **image fusion**. The project implements each operation for both architectures and benchmarks them on the same inputs, so the speed-up (or lack of one) can be measured and explained.

## Overview

Image-editing workloads are highly parallel, since the same operation is applied to millions of pixels, which makes them a natural fit for GPUs. In practice, the benefit depends on the operation, the image size, and the cost of moving data between host and device. This project measures that trade-off across three operations.

## Operations studied

| Operation | What it does | Typical cost profile |
|---|---|---|
| **Color grading** | Per-pixel adjustment of color (e.g. exposure, contrast, curves, LUTs, white balance) | Compute-light, fully parallel, memory-bound |
| **Masking** | Builds and applies a selection mask (e.g. thresholding, feathering, blur of mask edges) to restrict edits to a region | Mix of per-pixel and neighborhood operations |
| **Fusion** | Combines multiple images into one (e.g. alpha blending, exposure fusion, multi-resolution blending) | Several inputs, higher memory traffic |

## Run the code

Open the notebook directly in Google Colab (set the runtime to GPU):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blessszzz/Image-Editing/blob/main/CAO_Project.ipynb)

Or view the notebook on GitHub: [CAO_Project.ipynb](CAO_Project.ipynb)

**Input image:** the notebook loads `images.jpg` from the repo root. In Colab, upload it through the folder icon in the left sidebar before running, or add a cell at the top with `!wget -q https://raw.githubusercontent.com/blessszzz/Image-Editing/main/images.jpg`. To test your own photo, replace it with another image of the same name. The AI masking step looks for a cat (COCO class 17), so use a photo with a cat in it, or change the class ID in the notebook.
