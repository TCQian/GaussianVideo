from gsplat.project_gaussians_video import project_gaussians_video
from gsplat.rasterize_sum_video import rasterize_gaussians_sum_video
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from quantize import *
from optimizer import Adan
from PIL import Image
from filelock import FileLock, Timeout
import torchvision.transforms as transforms

class GaussianVideo(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W, self.T = kwargs["H"], kwargs["W"], kwargs["T"]
        self.BLOCK_W, self.BLOCK_H, self.BLOCK_T = kwargs["BLOCK_W"], kwargs["BLOCK_H"], kwargs["BLOCK_T"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W, # (1920 + 16 - 1) // 16 = 120
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H, # (1080 + 16 - 1) // 16 = 68
            (self.T + self.BLOCK_T - 1) // self.BLOCK_T, # (50 + 1 - 1) // 1 = 50
        ) # tile_bounds (120, 68, 50)
        self.device = kwargs["device"]

        # Position
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 3) - 0.5)))

        # Covariance
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 6))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        
        # Increase L33 (the last element in each row) to boost temporal variance.
        with torch.no_grad():
            self._cholesky.data[:, 5] += self.T  # adjust the constant as needed
        
        # Color
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.last_size = (self.H, self.W, self.T)
        self.quantize = kwargs["quantize"]
        self.register_buffer('background', torch.ones(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        # self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5, 0.5, 0, 0.5]).view(1, 6))
        
        if self.quantize:
            self.xyz_quantizer = FakeQuantizationHalf.apply 
            self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            self.cholesky_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=6)

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        return self._opacity
    
    @property
    def get_cholesky_elements(self):
        return self._cholesky + self.cholesky_bound
    
    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_video(
            self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.T, self.tile_bounds
        )
        out_img = rasterize_gaussians_sum_video(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            self.get_features, self._opacity, self.H, self.W, self.T,
            self.BLOCK_H, self.BLOCK_W, self.BLOCK_T,
            background=self.background, return_alpha=False
        )
        print("out_img.shape", out_img.shape)
        out_img = torch.clamp(out_img, 0, 1)  # [T, H, W, 3]
        out_img = out_img.view(-1, self.T, self.H, self.W, 3).permute(0, 4, 2, 3, 1).contiguous()
        return {"render": out_img}


    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()

        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-8))

        # print(f"[Loss] {loss.item():.6f}, PSNR: {psnr:.2f} dB")
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.data.norm().item()
        #         print(f"[Gradient Norm] {name}: {grad_norm:.6e}")

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        
        return loss, psnr

    
    def forward_quantize(self):
        # For latent positions: using 16 bits per element; now 3 channels (x, y, t)
        l_vqm, m_bit = 0, 16 * self.init_num_points * 3
        # Quantize the latent positions and then apply tanh to clip into (-1, 1)
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        
        # Quantize the Cholesky parameters using the uniform quantizer.
        # Note: Assuming the quantizer can handle the 6-channel input appropriately.
        cholesky_elements, l_vqs, s_bit = self.cholesky_quantizer(self._cholesky)
        # Add a fixed bound to ensure proper scaling (for both spatial and temporal dimensions)
        cholesky_elements = cholesky_elements + self.cholesky_bound

        # For features (colors), quantize using the vector quantizer.
        l_vqr, r_bit = 0, 0  # No extra loss for opacity or similar here
        colors, l_vqc, c_bit = self.features_dc_quantizer(self.get_features)
        
        # Project Gaussians into video space (3D) using the provided project function.
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_video(
            means, cholesky_elements, self.H, self.W, self.T, self.tile_bounds
        )
        
        # Rasterize the projected Gaussians into a video volume.
        out_img = rasterize_gaussians_sum_video(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            colors, self._opacity, self.H, self.W, self.T,
            self.BLOCK_H, self.BLOCK_W, self.BLOCK_T,
            background=self.background, return_alpha=False
        )
        
        # Clamp the output image to [0, 1]
        out_img = torch.clamp(out_img, 0, 1)
        # Rearrange dimensions to a conventional [batch, channels, H, W, T] order
        out_img = out_img.view(-1, self.T, self.H, self.W, 3).permute(0, 4, 2, 3, 1).contiguous()
        
        # Combine all quantization losses from the various components.
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
        
        # Return the rendered video, the quantization loss, and a list of bit counts.
        return {"render": out_img, "vq_loss": vq_loss, "unit_bit": [m_bit, s_bit, r_bit, c_bit]}

    def train_iter_quantize(self, gt_video):
        # Run a forward pass with quantization enabled.
        render_pkg = self.forward_quantize()
        video = render_pkg["render"]
        
        # Compute the loss as a combination of the reconstruction loss and the vector quantization loss.
        loss = loss_fn(video, gt_video, self.loss_type, lambda_value=0.7) + render_pkg["vq_loss"]
        
        # Backpropagate the loss.
        loss.backward()

        # Optionally, you could snapshot parameters before updating if needed.
        # before_update = {name: param.clone().detach() for name, param in self.named_parameters()}
        
        # Update the parameters.
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        
        # Compute PSNR using the mean squared error.
        with torch.no_grad():
            mse_loss = F.mse_loss(video, gt_video)
            psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-8))
        
        # Log loss and PSNR
        # print(f"[Loss-Quantized] {loss.item():.6f}, PSNR: {psnr:.2f} dB")
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.data.norm().item()
        #         print(f"[Gradient Norm - Quantized] {name}: {grad_norm:.6e}")
        
        # Step the learning rate scheduler.
        self.scheduler.step()
        
        return loss, psnr
    
    def compress_wo_ec(self):
        # Quantize the latent positions using the xyz_quantizer, then apply tanh to clip into (-1, 1)
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        # We store the raw latent positions (in half precision) as our encoding
        xyz_half = self._xyz.half()
        
        # Quantize the Cholesky (covariance) parameters using the cholesky_quantizer.
        # The compress function returns both the quantized bitstream and the decompressed version.
        quant_cholesky_elements, cholesky_elements = self.cholesky_quantizer.compress(self._cholesky)
        # Adjust the decompressed Cholesky elements by adding the fixed bound.
        cholesky_elements = cholesky_elements + self.cholesky_bound
        
        # Compress the features (e.g., colors) using the features_dc_quantizer.
        # This returns the quantized feature values along with discrete indices.
        colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        
        return {
            "xyz": xyz_half,
            "feature_dc_index": feature_dc_index,
            "quant_cholesky_elements": quant_cholesky_elements,
        }
        
    def decompress_wo_ec(self, encoding_dict):
        xyz = encoding_dict["xyz"]
        feature_dc_index = encoding_dict["feature_dc_index"]
        quant_cholesky_elements = encoding_dict["quant_cholesky_elements"]
        
        # Recover latent positions and ensure they are within (-1, 1)
        means = torch.tanh(xyz.float())
        
        # Decompress the quantized Cholesky parameters and adjust with the fixed bound
        cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        
        # Decompress the feature indices into colors
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        
        # Project the Gaussians into the video (3D) space
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_video(
            means, cholesky_elements, self.H, self.W, self.T, self.tile_bounds
        )
        
        # Rasterize the projected Gaussians to obtain the rendered video
        out_img = rasterize_gaussians_sum_video(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            colors, self._opacity, self.H, self.W, self.T,
            self.BLOCK_H, self.BLOCK_W, self.BLOCK_T,
            background=self.background, return_alpha=False
        )
        
        # Clamp the rendered output to [0, 1]
        out_img = torch.clamp(out_img, 0, 1)
        
        # Reshape and permute to the final format: [batch, channels, H, W, T]
        out_img = out_img.view(-1, self.T, self.H, self.W, 3).permute(0, 4, 2, 3, 1).contiguous()
        
        return {"render": out_img}
    
    def analysis_wo_ec(self, encoding_dict):
        # Retrieve the quantized cholesky elements and feature indices from the encoding.
        quant_cholesky_elements = encoding_dict["quant_cholesky_elements"]
        feature_dc_index = encoding_dict["feature_dc_index"]

        total_bits = 0
        initial_bits = 0
        codebook_bits = 0

        # Calculate the bits for the feature quantizer codebooks.
        for layer in self.features_dc_quantizer.quantizer.layers:
            codebook_bits += layer._codebook.embed.numel() * torch.finfo(layer._codebook.embed.dtype).bits

        # Bits required for the Cholesky quantizer's auxiliary parameters.
        initial_bits += self.cholesky_quantizer.scale.numel() * torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        initial_bits += self.cholesky_quantizer.beta.numel() * torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        initial_bits += codebook_bits

        total_bits += initial_bits

        # Latent positions (_xyz) are assumed to use 16 bits per element.
        total_bits += self._xyz.numel() * 16

        # Process feature indices.
        feature_dc_index_np = feature_dc_index.int().cpu().numpy()
        # Calculate the maximum index to estimate the number of bits required per index.
        index_max = np.max(feature_dc_index_np)
        max_bit = np.ceil(np.log2(index_max + 1))  # +1 to cover the full range
        total_bits += feature_dc_index_np.size * max_bit

        # Process quantized Cholesky elements.
        quant_cholesky_elements_np = quant_cholesky_elements.cpu().numpy()
        # Each element is stored with 6 bits.
        total_bits += quant_cholesky_elements_np.size * 6

        # Breakdown of individual components (for analysis purposes).
        position_bits = self._xyz.numel() * 16

        cholesky_bits = (
            self.cholesky_quantizer.scale.numel() * torch.finfo(self.cholesky_quantizer.scale.dtype).bits +
            self.cholesky_quantizer.beta.numel() * torch.finfo(self.cholesky_quantizer.beta.dtype).bits +
            quant_cholesky_elements_np.size * 6
        )

        feature_dc_bits = codebook_bits + feature_dc_index_np.size * max_bit

        # Compute bits per pixel (here per voxel) using H * W * T as the total number of elements.
        bpp = total_bits / (self.H * self.W * self.T)
        position_bpp = position_bits / (self.H * self.W * self.T)
        cholesky_bpp = cholesky_bits / (self.H * self.W * self.T)
        feature_dc_bpp = feature_dc_bits / (self.H * self.W * self.T)

        return {
            "bpp": bpp,
            "position_bpp": position_bpp,
            "cholesky_bpp": cholesky_bpp,
            "feature_dc_bpp": feature_dc_bpp,
        }
        
    def compress(self):
        # Quantize the latent positions using xyz_quantizer and apply tanh,
        # even though for storage we use the raw half-precision parameters.
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        
        # Compress the Cholesky parameters using the cholesky_quantizer.
        # This returns both a quantized version (to be entropy-coded) and a decompressed version.
        quant_cholesky_elements, cholesky_elements = self.cholesky_quantizer.compress(self._cholesky)
        # Adjust the decompressed Cholesky elements with the fixed bound.
        cholesky_elements = cholesky_elements + self.cholesky_bound
        
        # Compress the features (e.g., color) into discrete indices.
        colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        
        # Now, perform entropy coding on the quantized Cholesky elements and feature indices.
        # The helper function compress_matrix_flatten_categorical converts the flattened list into
        # an entropy-coded bitstream along with its histogram and unique symbols.
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = compress_matrix_flatten_categorical(
            quant_cholesky_elements.int().flatten().tolist()
        )
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(
            feature_dc_index.int().flatten().tolist()
        )
        
        return {
            "xyz": self._xyz.half(),
            "feature_dc_index": feature_dc_index,
            "quant_cholesky_elements": quant_cholesky_elements,
            "feature_dc_bitstream": [feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique],
            "cholesky_bitstream": [cholesky_compressed, cholesky_histogram_table, cholesky_unique]
        }
        
    def decompress(self, encoding_dict):
        # Retrieve the stored latent positions (in half precision) and determine device and number of points.
        xyz = encoding_dict["xyz"]
        num_points, device = xyz.size(0), xyz.device

        # Retrieve the entropy-coded bitstreams for the features and the Cholesky parameters.
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = encoding_dict["feature_dc_bitstream"]
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = encoding_dict["cholesky_bitstream"]

        # Decompress the feature indices.
        # Note: The features are quantized with a vector quantizer using 2 quantizers,
        # so the expected shape is (num_points, 2).
        feature_dc_index = decompress_matrix_flatten_categorical(
            feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique, num_points * 2, (num_points, 2)
        )

        # Decompress the quantized Cholesky elements.
        # For video, the Cholesky parameters are of length 6 per point.
        quant_cholesky_elements = decompress_matrix_flatten_categorical(
            cholesky_compressed, cholesky_histogram_table, cholesky_unique, num_points * 6, (num_points, 6)
        )

        # Convert the decompressed arrays back to torch tensors.
        feature_dc_index = torch.from_numpy(feature_dc_index).to(device).int()
        quant_cholesky_elements = torch.from_numpy(quant_cholesky_elements).to(device).float()

        # Recover the latent positions using a tanh to bring values into (-1, 1).
        means = torch.tanh(xyz.float())

        # Decompress the quantized Cholesky parameters using the quantizer's decompress method,
        # then add the fixed bound to get the final Cholesky elements.
        cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
        cholesky_elements = cholesky_elements + self.cholesky_bound

        # Decompress the features (e.g., color) using the features quantizer.
        colors = self.features_dc_quantizer.decompress(feature_dc_index)

        # Project the Gaussians into the video space (3D) using the video projection function.
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_video(
            means, cholesky_elements, self.H, self.W, self.T, self.tile_bounds
        )

        # Rasterize the Gaussians into a video volume using the corresponding video rasterization function.
        out_img = rasterize_gaussians_sum_video(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            colors, self._opacity, self.H, self.W, self.T,
            self.BLOCK_H, self.BLOCK_W, self.BLOCK_T,
            background=self.background, return_alpha=False
        )

        # Clamp the output to [0, 1] and reshape the tensor into the conventional order: [batch, channels, H, W, T].
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.T, self.H, self.W, 3).permute(0, 4, 2, 3, 1).contiguous()

        return {"render": out_img}
    
    def analysis(self, encoding_dict):
        # Retrieve quantized Cholesky elements and feature indices from the encoding.
        quant_cholesky_elements = encoding_dict["quant_cholesky_elements"]
        feature_dc_index = encoding_dict["feature_dc_index"]
        
        # Apply entropy coding to the quantized values:
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = compress_matrix_flatten_categorical(
            quant_cholesky_elements.int().flatten().tolist()
        )
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(
            feature_dc_index.int().flatten().tolist()
        )
        
        # Calculate codebook bits from the feature quantizer:
        codebook_bits = 0
        for layer in self.features_dc_quantizer.quantizer.layers:
            codebook_bits += layer._codebook.embed.numel() * torch.finfo(layer._codebook.embed.dtype).bits

        # Compute initial bits for auxiliary parameters:
        initial_bits = 0
        initial_bits += self.cholesky_quantizer.scale.numel() * torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        initial_bits += self.cholesky_quantizer.beta.numel() * torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        initial_bits += get_np_size(cholesky_histogram_table) * 8
        initial_bits += get_np_size(cholesky_unique) * 8
        initial_bits += get_np_size(feature_dc_histogram_table) * 8
        initial_bits += get_np_size(feature_dc_unique) * 8
        initial_bits += codebook_bits

        # Total bits is the sum of initial bits, bits for latent positions, and the entropy-coded bitstreams.
        total_bits = initial_bits
        total_bits += self._xyz.numel() * 16  # 16 bits per element for latent positions.
        total_bits += get_np_size(cholesky_compressed) * 8
        total_bits += get_np_size(feature_dc_compressed) * 8

        # Breakdown for individual components:
        position_bits = self._xyz.numel() * 16

        cholesky_bits = (
            self.cholesky_quantizer.scale.numel() * torch.finfo(self.cholesky_quantizer.scale.dtype).bits +
            self.cholesky_quantizer.beta.numel() * torch.finfo(self.cholesky_quantizer.beta.dtype).bits +
            get_np_size(cholesky_histogram_table) * 8 +
            get_np_size(cholesky_unique) * 8 +
            get_np_size(cholesky_compressed) * 8
        )

        feature_dc_bits = codebook_bits + get_np_size(feature_dc_histogram_table) * 8 + get_np_size(feature_dc_unique) * 8 + get_np_size(feature_dc_compressed) * 8

        # Calculate bits per voxel (bpp) using the full video volume dimensions.
        bpp = total_bits / (self.H * self.W * self.T)
        position_bpp = position_bits / (self.H * self.W * self.T)
        cholesky_bpp = cholesky_bits / (self.H * self.W * self.T)
        feature_dc_bpp = feature_dc_bits / (self.H * self.W * self.T)
        
        return {
            "bpp": bpp,
            "position_bpp": position_bpp,
            "cholesky_bpp": cholesky_bpp,
            "feature_dc_bpp": feature_dc_bpp,
        }
