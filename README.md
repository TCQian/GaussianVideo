# GaussianVideo: Video Representation Model Using Gaussian Splats for Efficient Storage and Rendering

This repository contains the implementation for **GaussianVideo**, a method for representing and compressing videos using 3D Gaussian splatting. Videos are modeled as continuous volumes in space and time using parameterized Gaussian functions.

While this initial method does not yet match the rate-distortion performance of neural codecs like HNeRV or traditional codecs like AV1, it demonstrates the feasibility of using 3D Gaussians as a unit of video representation. This is a first step toward understanding how splatting-based methods could evolve for efficient, temporally coherent video compression.

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/50c2f7a7-6c9b-4d06-9c95-44316a942ac9" width="400"/>
      <br/>
      <b>PSNR vs. BPP</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b8b31c76-d8f8-4492-a49b-97dfbf0db73f" width="400"/>
      <br/>
      <b>MS-SSIM vs. BPP</b>
    </td>
  </tr>
</table>

## Training Progression

The following images show how the output improves over training time with 10,000 Gaussians on the "HoneyBee" video in the UVG dataset:

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/3ef87eb0-088b-42b9-91ca-7602b62fda3b" width="360"/><br/>
      <code>HoneyBee_i50_g10000.png</code><br/>50 iterations
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/bd3c0cdf-4f01-449a-aa18-35a8335bd5fa" width="360"/><br/>
      <code>HoneyBee_i400_g10000.png</code><br/>400 iterations
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/445f6747-16d3-47bd-9aa1-d7f454fd5f84" width="360"/><br/>
      <code>HoneyBee_i800_g10000.png</code><br/>800 iterations
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/c8787bfa-636b-447c-9a6b-e896d2624a6a" width="360"/><br/>
      <code>HoneyBee_i1600_g10000.png</code><br/>1600 iterations
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/ffca371c-6c00-41b8-a275-055fe3140660" width="360"/><br/>
      <code>HoneyBee_i3200_g10000.png</code><br/>3200 iterations
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/10b6c41d-63dc-4b5f-8703-a2474665cb8b" width="360"/><br/>
      <code>HoneyBee_i6400_g10000.png</code><br/>6400 iterations
    </td>
  </tr>
</table>


## Overview

The pipeline consists of three main stages:

1. **Overfitting**: Learn a full-precision 3D Gaussian representation of a video segment.
2. **Quantization-aware fine-tuning**: Adapt the model to compressed (quantized) parameter formats.
3. **Evaluation**: Render and compare reconstructed frames using the quantized model.

## Requirements

```bash
pip install -r requirements.txt
cd gsplat
pip install .[dev]
```

## Dataset Format

Prepare your video dataset as a folder of PNG frames, named sequentially as:

```
dataset/
  ├── Beauty/
  │   ├── frame_0001.png
  │   ├── frame_0002.png
  │   ├── ...
  ├── HoneyBee/
  ├── Jockey/
```

This implementation assumes extracted frames from the [UVG dataset](https://ultravideo.fi/#testsequences), or your own videos converted into this format.

## Usage

### 1. Overfitting (Full-precision training)

```bash
python train_video.py \
  --dataset "dataset" \
  --data_name "Beauty" \
  --iterations 20000 \
  --model_name "beauty_3dgs" \
  --num_points 10000 \
  --start_frame 0 \
  --num_frames 10 \
  --lr 1e-2 \
  --save_imgs
```

### 2. Quantization-aware fine-tuning

```bash
python train_quantize_video.py \
  --dataset "dataset" \
  --data_name "Beauty" \
  --iterations 10000 \
  --model_name "beauty_quant" \
  --num_points 10000 \
  --model_path "checkpoints/Beauty/gaussian_model.pth.tar" \
  --start_frame 0 \
  --num_frames 10 \
  --lr 1e-2 \
  --save_imgs
```

### 3. Evaluation of quantized model

```bash
python test_quantize_video.py \
  --dataset "dataset" \
  --data_name "Beauty" \
  --iterations 10000 \
  --model_name "beauty_quant" \
  --num_points 10000 \
  --model_path "checkpoints_quant/Beauty/gaussian_model.best.pth.tar" \
  --start_frame 0 \
  --num_frames 10 \
  --lr 1e-2 \
  --save_imgs
```

## Notes

- `--num_points` controls the total number of 3D Gaussians.
- `--save_imgs` enables saving rendered frames to disk.
- All scripts require a CUDA-enabled GPU and use a modified version of `gsplat` for rasterization.
