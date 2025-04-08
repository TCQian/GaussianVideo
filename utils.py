import os
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
import torch
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
import cv2

class LogWriter:
    def __init__(self, file_path, train=True):
        os.makedirs(file_path, exist_ok=True)
        self.file_path = os.path.join(file_path, "train.txt" if train else "test.txt")

    def write(self, text):
        # 打印到控制台
        print(text)
        # 追加到文件
        with open(self.file_path, 'a') as file:
            file.write(text + '\n')


def loss_fn(pred, target, loss_type='L2', lambda_value=0.7):
    target = target.detach()
    pred = pred.float()
    target  = target.float()
    if loss_type == 'L2':
        loss = F.mse_loss(pred, target)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * F.l1_loss(pred, target)
    elif loss_type == 'Fusion4':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion_hinerv':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value)  * (1 - ms_ssim(pred, target, data_range=1, size_average=True, win_size=5))
    return loss

def strip_lowerdiag(L):
    if L.shape[1] == 3:
        uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]

    elif L.shape[1] == 2:
        uncertainty = torch.zeros((L.shape[0], 3), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 1, 1]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_rotation_2d(r):
    '''
    Build rotation matrix in 2D.
    '''
    R = torch.zeros((r.size(0), 2, 2), device='cuda')
    R[:, 0, 0] = torch.cos(r)[:, 0]
    R[:, 0, 1] = -torch.sin(r)[:, 0]
    R[:, 1, 0] = torch.sin(r)[:, 0]
    R[:, 1, 1] = torch.cos(r)[:, 0]
    return R

def build_scaling_rotation_2d(s, r, device):
    L = torch.zeros((s.shape[0], 2, 2), dtype=torch.float, device='cuda')
    R = build_rotation_2d(r, device)
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L = R @ L
    return L
    
def build_covariance_from_scaling_rotation_2d(scaling, scaling_modifier, rotation, device):
    '''
    Build covariance metrix from rotation and scale matricies.
    '''
    L = build_scaling_rotation_2d(scaling_modifier * scaling, rotation, device)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def build_triangular(r):
    R = torch.zeros((r.size(0), 2, 2), device=r.device)
    R[:, 0, 0] = r[:, 0]
    R[:, 1, 0] = r[:, 1]
    R[:, 1, 1] = r[:, 2]
    return R

def process_yuv_video(
    file_path: str, 
    width: int, 
    height: int, 
    start_frame: int = 0, 
    num_frames: int = None
) -> list[np.ndarray]:
    """
    Process a YUV video file into RGB frames, with support for frame range selection.
    
    Args:
        file_path: Path to the YUV video file
        width: Video width in pixels
        height: Video height in pixels
        start_frame: Index of the first frame to process (0-based)
        num_frames: Number of frames to process (None for all remaining frames)
    
    Returns:
        List of RGB frames as numpy arrays (dtype=np.uint8)
    """
    frame_size = width * height * 3 // 2  # Size of one frame in bytes (I420 format)
    file_size = os.path.getsize(file_path)
    total_frames_in_file = file_size // frame_size
    
    # Validate start_frame
    if start_frame < 0 or start_frame >= total_frames_in_file:
        raise ValueError(f"start_frame {start_frame} is out of range (0-{total_frames_in_file-1})")
    
    # Calculate actual number of frames to process
    if num_frames is None:
        num_frames = total_frames_in_file - start_frame
    else:
        num_frames = min(num_frames, total_frames_in_file - start_frame)
    
    video_frames = []
    
    with open(file_path, 'rb') as f:
        # Seek to the start frame
        f.seek(start_frame * frame_size)
        
        for _ in tqdm(range(num_frames), desc=f"Processing frames {start_frame}-{start_frame+num_frames-1}"):
            yuv_frame = f.read(frame_size)
            if not yuv_frame or len(yuv_frame) < frame_size:
                break  # End of file or incomplete frame
            
            # Convert YUV I420 to numpy array and reshape
            yuv = np.frombuffer(yuv_frame, dtype=np.uint8).reshape((height * 3 // 2, width))
            
            # Convert YUV to RGB
            rgb_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            
            video_frames.append(rgb_frame)
    
    return video_frames

def calculate_vmaf(gt_yuv_path: str, generated_yuv_path: str, width: int, height: int) -> float:
    """
    Calculate VMAF score between two YUV videos using FFmpeg's libvmaf.
    Simplified for FFmpeg 7.0.2 compatibility.
    """
    # First verify the files exist
    if not (os.path.exists(gt_yuv_path) and os.path.exists(generated_yuv_path)):
        print("Error: Input files don't exist")
        return 0.0

    cmd = [
        'ffmpeg',
        '-s', f'{width}x{height}',        # Input resolution
        '-pix_fmt', 'yuv420p',           # Must match your YUV format
        '-i', generated_yuv_path,         # Distorted video
        '-s', f'{width}x{height}',
        '-pix_fmt', 'yuv420p',
        '-i', gt_yuv_path,                # Reference video
        '-lavfi', 'libvmaf',             # Simple VMAF calculation
        '-f', 'null', '-'
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Parse FFmpeg output for VMAF score
        for line in result.stderr.split('\n'):
            if "VMAF score" in line:
                return float(line.split("VMAF score:")[1].strip())
    except subprocess.CalledProcessError as e:
        print(f"VMAF calculation failed. Command: {' '.join(cmd)}")
        print(f"FFmpeg error: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    
    return 0.0  # Fallback value