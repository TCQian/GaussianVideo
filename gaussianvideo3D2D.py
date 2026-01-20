from gsplat.project_gaussians_video import project_gaussians_video
from gsplat.rasterize_sum_video import rasterize_gaussians_sum_video
from utils import *
import torch
import torch.nn as nn
import math
from optimizer import Adan
import torch.nn.functional as F

from utils import *
from quantize import *

class GaussianVideo3D2D(nn.Module):
    def __init__(self, layer= 0, loss_type="L2", **kwargs):
        super().__init__()

        self.debug_mode = False 

        self.layer = int(layer)
        self.loss_type = loss_type
        self.iterations = kwargs["iterations"]
        self.opt_type = kwargs["opt_type"]
        self.lr = kwargs["lr"]
        self.quantize = kwargs["quantize"]

        self.H, self.W, self.T = kwargs["H"], kwargs["W"], kwargs["T"]
        self.BLOCK_W, self.BLOCK_H, self.BLOCK_T = kwargs["BLOCK_W"], kwargs["BLOCK_H"], kwargs["BLOCK_T"]
        
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W, # (1920 + 16 - 1) // 16 = 120
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H, # (1080 + 16 - 1) // 16 = 68
            (self.T + self.BLOCK_T - 1) // self.BLOCK_T, # (50 + 1 - 1) // 1 = 50
        ) # tile_bounds (120, 68, 50)
        self.device = kwargs["device"]

        self.num_points = kwargs["num_points"]
        
        self.register_buffer('background', torch.ones(3))
        self.register_buffer('cholesky_bound_3D', torch.tensor([0.5, 0, 0, 0.5, 0, 0.5]).view(1, 6))
        self.register_buffer('cholesky_bound_2D', torch.tensor([0.5, 0, 0, 0.5, 0, 0]).view(1, 6))

        self.opacity_activation = torch.sigmoid

        self._xyz_3D = None
        self._cholesky_3D = None
        self._features_dc_3D = None
        self._opacity_3D = None
        self._xyz_2D = None
        self._cholesky_2D = None
        self._features_dc_2D = None
        self._opacity_2D = None
        self.num_points_list = None
        self.optimizer = None
        self.scheduler = None

        if self.quantize:
            self.xyz_quantizer = FakeQuantizationHalf.apply 
            self.features_dc_quantizer_layer0 = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            self.features_dc_quantizer_layer1 = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) if self.layer == 1 else None
            self.cholesky_quantizer_layer0 = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=6)
            self.cholesky_quantizer_layer1 = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=3) if self.layer == 1 else None
            
            self.decoded_xyz_layer0 = None
            self.decoded_feature_dc_index_layer0 = None
            self.decoded_quant_cholesky_elements_layer0 = None

    def _create_data_from_checkpoint(self, checkpoint_path_layer0, checkpoint_path_layer1):
        if self.layer == 0:
            assert checkpoint_path_layer1 is None, "Layer 1 checkpoint is not required for Layer 0 training"
        elif self.layer == 1:
            assert checkpoint_path_layer0 is not None, "Layer 0 checkpoint is required for Layer 1 training"

        if checkpoint_path_layer0 is not None:
            print(f"Loading layer 0 checkpoint from: {checkpoint_path_layer0}")
            checkpoint_layer0 = torch.load(checkpoint_path_layer0)
            
            xyz = checkpoint_layer0['_xyz_3D']
            cholesky = checkpoint_layer0['_cholesky_3D']
            features_dc = checkpoint_layer0['_features_dc_3D']
            self._opacity_3D = nn.Parameter(self.opacity_activation(checkpoint_layer0['_opacity_3D']))

            if self.layer == 0:
                self._xyz_3D = nn.Parameter(xyz, requires_grad=True)
                self._cholesky_3D = nn.Parameter(cholesky, requires_grad=True)
                self._features_dc_3D = nn.Parameter(features_dc, requires_grad=True)
            else:
                self._xyz_3D = xyz.requires_grad_(False)
                self._cholesky_3D = cholesky.requires_grad_(False)
                self._features_dc_3D = features_dc.requires_grad_(False)
                print(f'Layer 0 gaussians are frozen for Layer {self.layer} training')

            if self.quantize:
                try:
                    self.cholesky_quantizer_layer0.load_state_dict(checkpoint_layer0['cholesky_quantizer_layer0'])
                    self.features_dc_quantizer_layer0.load_state_dict(checkpoint_layer0['features_dc_quantizer_layer0'])
                except:
                    print("Layer 0 quantization parameters not found, initialized new quantization parameters")

            print(f"Layer 0 checkpoint loaded successfully with {self._xyz_3D.shape[0]} gaussians")
        else:
            if self.layer == 0:
                self._init_layer0()

        if checkpoint_path_layer1 is not None:
            print(f"Loading layer 1 checkpoint from: {checkpoint_path_layer1}")
            checkpoint_layer1 = torch.load(checkpoint_path_layer1)
            self.num_points_list = checkpoint_layer1['gaussian_num_list']
            self._ckpt_xyz_2D = checkpoint_layer1['_xyz_2D']
            self._xyz_2D = nn.Parameter(self._ckpt_xyz_2D)

            self._ckpt_cholesky_2D = checkpoint_layer1['_cholesky_2D']
            self._cholesky_2D = nn.Parameter(self._ckpt_cholesky_2D)

            self._features_dc_2D = nn.Parameter(checkpoint_layer1['_features_dc_2D'])
            self._opacity_3D = nn.Parameter(checkpoint_layer1['_opacity_3D']) # loading opacity_3D tuned in layer 1 training
            self._opacity_2D = nn.Parameter(checkpoint_layer1['_opacity_2D']).requires_grad_(False)

            if self.quantize:
                try:
                    self.cholesky_quantizer_layer1.load_state_dict(checkpoint_layer1['cholesky_quantizer_layer1'])
                    self.features_dc_quantizer_layer1.load_state_dict(checkpoint_layer1['features_dc_quantizer_layer1'])
                except:
                    print("Layer 1 quantization parameters not found, initialized new quantization parameters")

            print(f"Layer 1 checkpoint loaded successfully with {self._xyz_2D.shape[0]} gaussians")
        else:
            if self.layer == 1:
                self._init_layer1()

        self.trainable_params = []
        if self.layer == 0:
            self.trainable_params.append(self._xyz_3D)
            self.trainable_params.append(self._cholesky_3D)
            self.trainable_params.append(self._features_dc_3D)
            self.trainable_params.append(self._opacity_3D)
            if self.quantize:
                self.trainable_params.append(self.cholesky_quantizer_layer0.scale)
                self.trainable_params.append(self.cholesky_quantizer_layer0.beta)
                self.trainable_params.extend(self.features_dc_quantizer_layer0.parameters())

        elif self.layer == 1:
            if self._xyz_3D.shape[0] > 0:
                self.trainable_params.append(self._opacity_3D)
            self.trainable_params.append(self._xyz_2D)
            self.trainable_params.append(self._cholesky_2D)
            self.trainable_params.append(self._features_dc_2D)
            # self.trainable_params.append(self._opacity_2D)
            if self.quantize:
                self.trainable_params.append(self.cholesky_quantizer_layer1.scale)
                self.trainable_params.append(self.cholesky_quantizer_layer1.beta)
                self.trainable_params.extend(self.features_dc_quantizer_layer1.parameters())

        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(self.trainable_params, lr=self.lr)
        else:
            self.optimizer = Adan(self.trainable_params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def _init_layer0(self):
        self.init_num_points = int(self.num_points * self.T / 2)
        self._xyz_3D = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 3) - 0.5)))
        self._cholesky_3D = nn.Parameter(torch.rand(self.init_num_points, 6))
        self._opacity_3D = nn.Parameter(torch.logit(0.1 * torch.ones(self.init_num_points, 1)))
        self._features_dc_3D = nn.Parameter(torch.rand(self.init_num_points, 3))

        if self.quantize:
            self.cholesky_quantizer_layer0._init_data(self._cholesky_3D)

        self.layer = 0
        print("Layer 0 initialized, number of gaussians: ", self._xyz_3D.shape[0])

    def _init_layer1(self):
        self.num_points_layer0 = self._xyz_3D.shape[0]
        self.init_num_points = int((self.num_points * self.T) - self.num_points_layer0)
        num_points_per_frame = int(self.init_num_points / self.T)
        self.num_points_list = []
        for t in range(self.T):
            start = t * num_points_per_frame
            if t == self.T - 1:
                end = self.init_num_points
            else:
                end = start + num_points_per_frame
            self.num_points_list.append((start, end))

        self._xyz_2D = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._cholesky_2D = nn.Parameter(torch.rand(self.init_num_points, 3))
        self._opacity_2D = nn.Parameter(torch.ones(self.init_num_points, 1)).requires_grad_(False)
        self._features_dc_2D = nn.Parameter(torch.rand(self.init_num_points, 3))

        if self.quantize:
            self.cholesky_quantizer_layer1._init_data(self._cholesky_2D)
            
        self.layer = 1
        print("GaussianVideo3D2D: Layer 1 initialized, number of gaussians: ", self._xyz_2D.shape[0])
        
    def save_checkpoint(self, path, best=False):
        if self.layer == 0:
            state = {
                '_xyz_3D': self._xyz_3D.data,
                '_cholesky_3D': self._cholesky_3D.data,
                '_features_dc_3D': self._features_dc_3D.data,
                '_opacity_3D': self._opacity_3D.data,
                'cholesky_quantizer_layer0': self.cholesky_quantizer_layer0.state_dict() if self.quantize else None,
                'features_dc_quantizer_layer0': self.features_dc_quantizer_layer0.state_dict() if self.quantize else None,
            }
        elif self.layer == 1:
            state = {
                '_xyz_2D': self._xyz_2D.data,
                '_cholesky_2D': self._cholesky_2D.data,
                '_features_dc_2D': self._features_dc_2D.data,
                '_opacity_2D': self._opacity_2D.data,
                '_opacity_3D': self._opacity_3D.data,
                'gaussian_num_list': self.num_points_list,
                'cholesky_quantizer_layer1': self.cholesky_quantizer_layer1.state_dict() if self.quantize else None,
                'features_dc_quantizer_layer1': self.features_dc_quantizer_layer1.state_dict() if self.quantize else None,
            }

        if best:
            torch.save(state, path / f"layer_{self.layer}_model.best.pth.tar")
        else:
            torch.save(state, path / f"layer_{self.layer}_model.pth.tar")
        print(f"Checkpoint saved to: {path / f'layer_{self.layer}_model.pth.tar'}")

    @property
    def get_xyz(self):
        if self.layer == 0:
            return torch.tanh(self._xyz_3D)
        elif self.layer == 1:
            xyz_3D_tanh = torch.tanh(self._xyz_3D)
            xyz_2D_spatial_tanh = torch.tanh(self._xyz_2D)
            xyz_2d_temporal = torch.zeros(self._xyz_2D.shape[0], 1, device=self._xyz_2D.device, dtype=self._xyz_2D.dtype)
            xyz_2d_full = torch.cat((xyz_2D_spatial_tanh, xyz_2d_temporal), dim=1)
            return torch.cat((xyz_3D_tanh, xyz_2d_full), dim=0)
        
    @property
    def get_xyz_quantize(self):
        assert self.quantize, "Quantization is not enabled"
        if self.layer == 0:
            xyz_quantized = self.xyz_quantizer(self._xyz_3D)
            return torch.tanh(xyz_quantized)
        elif self.layer == 1:
            assert self.decoded_xyz_layer0 is not None, "To get xyz of layer 1, decoded_xyz_layer0 is required for layer 1"
            xyz_2d_spatial_quantized = self.xyz_quantizer(self._xyz_2D)
            xyz_2d_temporal = torch.zeros(self._xyz_2D.shape[0], 1, device=self._xyz_2D.device, dtype=self._xyz_2D.dtype)
            xyz_2d_spatial_tanh = torch.tanh(xyz_2d_spatial_quantized)
            xyz_2d_quantized = torch.cat((xyz_2d_spatial_tanh, xyz_2d_temporal), dim=1)
            xyz_3D_tanh = torch.tanh(self.decoded_xyz_layer0)
            xyz_quantized = torch.cat((xyz_3D_tanh, xyz_2d_quantized), dim=0)
            return xyz_quantized

    @property
    def get_features(self):
        if self.layer == 0:
            return self._features_dc_3D
        elif self.layer == 1:
            return torch.cat((self._features_dc_3D, self._features_dc_2D), dim=0)
    
    @property
    def get_features_quantize(self):
        assert self.quantize, "Quantization is not enabled"
        if self.layer == 0:
            feature_quantized, self.l_vqc, self.c_bit = self.features_dc_quantizer_layer0(self._features_dc_3D)
        elif self.layer == 1:
            assert self.decoded_feature_dc_index_layer0 is not None, "To get features of layer 1, decoded_feature_dc_index_layer0 is required for layer 1"
            features_quantized_2D, self.l_vqc, self.c_bit  = self.features_dc_quantizer_layer1(self._features_dc_2D)
            feature_quantized = torch.cat((self.decoded_feature_dc_index_layer0, features_quantized_2D), dim=0)
        return feature_quantized
    
    @property
    def get_opacity(self):
        if self.layer == 0:
            return self.opacity_activation(self._opacity_3D)
        elif self.layer == 1:
            return torch.cat((self._opacity_3D, self._opacity_2D), dim=0)
    
    def get_cholesky_2d_full(self, cholesky_2d_to_be_extended):
        num_points_2d = cholesky_2d_to_be_extended.shape[0]
        device = cholesky_2d_to_be_extended.device
        dtype = cholesky_2d_to_be_extended.dtype
        
        zeros = torch.zeros((num_points_2d, 1), device=device, dtype=dtype)
        ones  = torch.ones((num_points_2d, 1), device=device, dtype=dtype)
        
        l11 = cholesky_2d_to_be_extended[:, 0:1]
        l12 = cholesky_2d_to_be_extended[:, 1:2]
        l22 = cholesky_2d_to_be_extended[:, 2:3]
        
        return torch.cat([l11, l12, zeros, l22, zeros, ones], dim=1)

    @property
    def get_cholesky_elements(self):
        if self.layer == 0:
            return self._cholesky_3D + self.cholesky_bound_3D
        elif self.layer == 1:
            merged_cholesky = torch.cat((self._cholesky_3D + self.cholesky_bound_3D, self.get_cholesky_2d_full(self._cholesky_2D) + self.cholesky_bound_2D), dim=0)
            return merged_cholesky

    @property
    def get_cholesky_elements_quantize(self):
        assert self.quantize, "Quantization is not enabled"
        if self.layer == 0:
            cholesky, self.l_vqs, self.s_bit = self.cholesky_quantizer_layer0(self._cholesky_3D)
            cholesky = cholesky + self.cholesky_bound_3D
        elif self.layer == 1:
            assert self.decoded_quant_cholesky_elements_layer0 is not None, "To get cholesky of layer 1, decoded_quant_cholesky_elements_layer0 is required for layer 1"
            cholesky_2d_used_quantized, self.l_vqs, self.s_bit = self.cholesky_quantizer_layer1(self._cholesky_2D)
            cholesky = torch.cat((self.decoded_quant_cholesky_elements_layer0 + self.cholesky_bound_3D, self.get_cholesky_2d_full(cholesky_2d_used_quantized) + self.cholesky_bound_2D), dim=0)
        return cholesky
    
    def prune(self, opac_threshold=0.2):
        if self.layer == 0:
            print(f"min and max opacity: {self.get_opacity.min().item()}, {self.get_opacity.max().item()}")
            mask = (self.get_opacity > opac_threshold).squeeze()
            
            self._xyz_3D = torch.nn.Parameter(self._xyz_3D[mask])
            self._cholesky_3D = torch.nn.Parameter(self._cholesky_3D[mask])
            self._features_dc_3D = torch.nn.Parameter(self._features_dc_3D[mask])
            self._opacity_3D = torch.nn.Parameter(self._opacity_3D[mask])
            for param_group in self.optimizer.param_groups:
                param_group['params'] = [p for p in self.parameters() if p.requires_grad]

            print(f"Pruned to {self._xyz_3D.shape[0]} Gaussians.")

        else:
            print(f"min and max opacity: {self.get_opacity[self.num_points_layer0:].min().item()}, {self.get_opacity[self.num_points_layer0:].max().item()}")
            mask = (self.get_opacity[self.num_points_layer0:] > opac_threshold).squeeze()

            self._xyz_2D = torch.nn.Parameter(self._xyz_2D[mask])
            self._cholesky_2D = torch.nn.Parameter(self._cholesky_2D[mask])
            self._features_dc_2D = torch.nn.Parameter(self._features_dc_2D[mask])
            self._opacity_2D = torch.nn.Parameter(self._opacity_2D[mask])
            for param_group in self.optimizer.param_groups:
                param_group['params'] = [p for p in self.parameters() if p.requires_grad]

            new_num_points_list = []
            next_start = 0
            for start, end in self.num_points_list:
                keep_mask = mask[start:end]
                end = next_start + keep_mask.sum()
                assert end > next_start, f"end={end} must be greater than next_start={next_start}"
                new_num_points_list.append((next_start, end))
                next_start = end

            self.num_points_list = new_num_points_list
            print(f"Pruned to {self._xyz_2D.shape[0]} Gaussians.")

    def densify(self, num_new_gaussians=100):
        assert self.layer == 1, "Densify is only supported for layer 1"
        device = self._xyz_2D.device
        num_new_per_frame = int(num_new_gaussians / self.T)
        # create a new tensor with the new total number of points
        new_total_num_points = self._xyz_2D.shape[0] + num_new_per_frame * self.T
        new_xyz = torch.zeros(new_total_num_points, 2, device=device)
        new_cholesky = torch.zeros(new_total_num_points, 3, device=device)
        new_opacity = torch.zeros(new_total_num_points, 1, device=device)
        new_features_dc = torch.zeros(new_total_num_points, 3, device=device)
        new_num_points_list = []

        # insert the new gaussians in between the existing gaussians
        next_start = 0
        for start, end in self.num_points_list:
            cur_start = next_start
            cur_end = cur_start + (end - start)

            new_xyz[cur_start:cur_end] = self._xyz_2D[start:end]
            new_xyz[cur_end:cur_end+num_new_per_frame] = torch.atanh(2 * (torch.rand(num_new_per_frame, 2, device=device) - 0.5))
            new_cholesky[cur_start:cur_end] = self._cholesky_2D[start:end]
            new_cholesky[cur_end:cur_end+num_new_per_frame] = torch.rand(num_new_per_frame, 3, device=device)
            new_opacity[cur_start:cur_end] = self._opacity_2D[start:end]
            new_opacity[cur_end:cur_end+num_new_per_frame] = torch.logit(0.1 * torch.ones(num_new_per_frame, 1, device=device))
            new_features_dc[cur_start:cur_end] = self._features_dc_2D[start:end]
            new_features_dc[cur_end:cur_end+num_new_per_frame] = torch.rand(num_new_per_frame, 3, device=device)

            new_num_points_list.append((cur_start, cur_end+num_new_per_frame))
            next_start = cur_end + num_new_per_frame

        self._xyz_2D = torch.nn.Parameter(new_xyz)
        self._cholesky_2D = torch.nn.Parameter(new_cholesky)
        self._opacity_2D = torch.nn.Parameter(new_opacity)
        self._features_dc_2D = torch.nn.Parameter(new_features_dc)
        self.num_points_list = new_num_points_list
        for param_group in self.optimizer.param_groups:
            param_group['params'] = [p for p in self.parameters() if p.requires_grad]
        print(f"Added {num_new_per_frame} new gaussians per frame. Total: {self._xyz_2D.shape[0]} Gaussians.")

    def _fix_temporal_coords_after_projection(self, xys):
        if self.layer != 1 or self.num_points_list is None:
            return xys
        
        xys_fixed = xys.clone()
        layer1_start = self._xyz_3D.shape[0] if self._xyz_3D.shape[0] > 0 else 0
        
        for t, (start, end) in enumerate(self.num_points_list):
            idx_start = layer1_start + start
            idx_end = layer1_start + end
            xys_fixed[idx_start:idx_end, 2] = float(t)
        
        return xys_fixed

    def forward(self):
        self.xys, depths, radii, conics, num_tiles_hit = project_gaussians_video(
            self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.T, self.tile_bounds
        )
        # Fix temporal coordinates to exact frame indices after projection
        self.xys = self._fix_temporal_coords_after_projection(self.xys)
    
        out_img = rasterize_gaussians_sum_video(
            self.xys, depths, radii, conics, num_tiles_hit,
            self.get_features, self.get_opacity, self.H, self.W, self.T,
            self.BLOCK_H, self.BLOCK_W, self.BLOCK_T,
            background=self.background, return_alpha=False
        )
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

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        
        return loss, psnr

    def forward_quantize(self):
        num_points = self.get_xyz.shape[0]
        num_xyz_dims = 3 if self.layer == 0 else 2
        self.l_vqm, self.m_bit = 0, 16 * num_points * num_xyz_dims
        self.l_vqr, self.r_bit = 0, 0 
        
        self.xys, depths, radii, conics, num_tiles_hit = project_gaussians_video(
            self.get_xyz_quantize, self.get_cholesky_elements_quantize, self.H, self.W, self.T, self.tile_bounds
        )
        self.xys = self._fix_temporal_coords_after_projection(self.xys)
        out_img = rasterize_gaussians_sum_video(
            self.xys, depths, radii, conics, num_tiles_hit,
            self.get_features_quantize, self.get_opacity, self.H, self.W, self.T,
            self.BLOCK_H, self.BLOCK_W, self.BLOCK_T,
            background=self.background, return_alpha=False
        )
        out_img = torch.clamp(out_img, 0, 1)  # [T, H, W, 3]
        out_img = out_img.view(-1, self.T, self.H, self.W, 3).permute(0, 4, 2, 3, 1).contiguous()
        
        vq_loss = self.l_vqm + self.l_vqs + self.l_vqr + self.l_vqc

        return {"render": out_img, "vq_loss": vq_loss, "unit_bit": [self.m_bit, self.s_bit, self.r_bit, self.c_bit]}


    def train_iter_quantize(self, gt_video):
        render_pkg = self.forward_quantize()
        video = render_pkg["render"]

        loss = loss_fn(video, gt_video, self.loss_type, lambda_value=0.7) + render_pkg["vq_loss"]

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        
        with torch.no_grad():
            mse_loss = F.mse_loss(video, gt_video)
            psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-8))
        self.scheduler.step()
        
        return loss, psnr

    def create_en_decoded_layer0(self):
        '''
        To stimulate the analysis of encoder and decoder for layer 1, we need to get the encoded and decoded attributes for layer 0
        '''
        encoded_xyz = self._xyz_3D.half()
        _, feature_dc_index_encoded = self.features_dc_quantizer_layer0.compress(self._features_dc_3D)
        quant_cholesky_elements_encoded, _ = self.cholesky_quantizer_layer0.compress(self._cholesky_3D)

        decoded_feature_dc_index = self.features_dc_quantizer_layer0.decompress(feature_dc_index_encoded)
        decoded_quant_cholesky_elements = self.cholesky_quantizer_layer0.decompress(quant_cholesky_elements_encoded)

        self.decoded_xyz_layer0 = encoded_xyz.float().detach()
        self.decoded_feature_dc_index_layer0 = decoded_feature_dc_index.detach()
        self.decoded_quant_cholesky_elements_layer0 = decoded_quant_cholesky_elements.detach()
    
    def compress_wo_ec(self):
        if self.layer == 0:
            xyz = self._xyz_3D.half()
            _, feature_dc_index = self.features_dc_quantizer_layer0.compress(self._features_dc_3D)
            quant_cholesky_elements, _ = self.cholesky_quantizer_layer0.compress(self._cholesky_3D)
            return {"xyz": xyz, "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements}
        elif self.layer == 1:
            xyz = self._xyz_2D.half()
            _, feature_dc_index = self.features_dc_quantizer_layer1.compress(self._features_dc_2D)
            quant_cholesky_elements, _ = self.cholesky_quantizer_layer1.compress(self._cholesky_2D)
            return {"xyz": xyz, "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements}

    def decompress_wo_ec(self, encoding_dict):
        xyz = encoding_dict["xyz"]
        feature_dc_index = encoding_dict["feature_dc_index"]
        quant_cholesky_elements = encoding_dict["quant_cholesky_elements"]
        
        if self.layer == 0:
            means = torch.tanh(xyz.float())
            cholesky_elements = self.cholesky_quantizer_layer0.decompress(quant_cholesky_elements) + self.cholesky_bound_3D
            colors = self.features_dc_quantizer_layer0.decompress(feature_dc_index)
        elif self.layer == 1:
            assert self.decoded_xyz_layer0 is not None, "decoded_xyz_layer0 is required for layer 1"
            assert self.decoded_feature_dc_index_layer0 is not None, "decoded_feature_dc_index_layer0 is required for layer 1"
            assert self.decoded_quant_cholesky_elements_layer0 is not None, "decoded_quant_cholesky_elements_layer0 is required for layer 1"
            xyz_2D_spatial_tanh = torch.tanh(xyz.float())
            xyz_2d_temporal = torch.zeros(xyz.shape[0], 1, device=xyz.device, dtype=xyz.dtype)
            xyz_2d_full = torch.cat((xyz_2D_spatial_tanh, xyz_2d_temporal), dim=1)
            xyz_3D_tanh = torch.tanh(self.decoded_xyz_layer0.float())
            means = torch.cat((xyz_3D_tanh, xyz_2d_full), dim=0)

            cholesky_2d_decoded = self.cholesky_quantizer_layer1.decompress(quant_cholesky_elements)
            cholesky_elements = torch.cat((self.decoded_quant_cholesky_elements_layer0 + self.cholesky_bound_3D, self.get_cholesky_2d_full(cholesky_2d_decoded) + self.cholesky_bound_2D), dim=0)

            features_dc_2d_decoded = self.features_dc_quantizer_layer1.decompress(feature_dc_index)
            colors = torch.cat((self.decoded_feature_dc_index_layer0, features_dc_2d_decoded), dim=0)
        
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_video(
            means, cholesky_elements, self.H, self.W, self.T, self.tile_bounds
        )
        self.xys = self._fix_temporal_coords_after_projection(self.xys)
        out_img = rasterize_gaussians_sum_video(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            colors, self.get_opacity, self.H, self.W, self.T,
            self.BLOCK_H, self.BLOCK_W, self.BLOCK_T,
            background=self.background, return_alpha=False
        )
        
        out_img = torch.clamp(out_img, 0, 1)
        
        out_img = out_img.view(-1, self.T, self.H, self.W, 3).permute(0, 4, 2, 3, 1).contiguous()
        
        return {"render": out_img}
    
    def analysis_wo_ec(self, encoding_dict):
        quant_cholesky_elements = encoding_dict["quant_cholesky_elements"]
        feature_dc_index = encoding_dict["feature_dc_index"]

        total_bits = 0
        initial_bits = 0
        codebook_bits = 0

        if self.layer == 0:
            xyz = self._xyz_3D
            features_dc_quantizer = self.features_dc_quantizer_layer0
            cholesky_quantizer = self.cholesky_quantizer_layer0 
        elif self.layer == 1:
            xyz = self._xyz_2D
            features_dc_quantizer = self.features_dc_quantizer_layer1
            cholesky_quantizer = self.cholesky_quantizer_layer1

        print(f"\n{'='*80}")
        print(f"ANALYSIS_WO_EC - Layer {self.layer} Bit Calculation Comparison")
        print(f"{'='*80}")

        # Calculate codebook bits
        for layer in features_dc_quantizer.quantizer.layers:
            codebook_bits += layer._codebook.embed.numel() * torch.finfo(layer._codebook.embed.dtype).bits

        # Calculate initial bits (overhead)
        cholesky_scale_bits = cholesky_quantizer.scale.numel() * torch.finfo(cholesky_quantizer.scale.dtype).bits
        cholesky_beta_bits = cholesky_quantizer.beta.numel() * torch.finfo(cholesky_quantizer.beta.dtype).bits
        initial_bits += cholesky_scale_bits
        initial_bits += cholesky_beta_bits
        initial_bits += codebook_bits

        print(f"\n[INITIAL BITS - Overhead]")
        print(f"  Cholesky scale: {cholesky_scale_bits:,} bits ({cholesky_quantizer.scale.numel()} elements × {torch.finfo(cholesky_quantizer.scale.dtype).bits} bits)")
        print(f"  Cholesky beta:  {cholesky_beta_bits:,} bits ({cholesky_quantizer.beta.numel()} elements × {torch.finfo(cholesky_quantizer.beta.dtype).bits} bits)")
        print(f"  Codebook:       {codebook_bits:,} bits")
        print(f"  Total initial:  {initial_bits:,} bits")

        total_bits += initial_bits

        # Position bits
        position_bits = xyz.numel() * 16
        total_bits += position_bits
        print(f"\n[POSITION BITS]")
        print(f"  Calculated: {position_bits:,} bits ({xyz.numel()} elements × 16 bits)")
        print(f"  (No quantizer.size() method for position - using fixed 16 bits/element)")

        # Feature DC index bits
        feature_dc_index_np = feature_dc_index.int().cpu().numpy()
        index_max = np.max(feature_dc_index_np)
        max_bit = np.ceil(np.log2(index_max + 1))
        feature_dc_index_bits_calculated = feature_dc_index_np.size * max_bit
        total_bits += feature_dc_index_bits_calculated

        # Get actual size from quantizer
        features_dc_actual_bits = features_dc_quantizer.size(feature_dc_index)
        features_dc_index_bits_actual = features_dc_actual_bits - codebook_bits  # Remove codebook from actual to compare index bits

        print(f"\n[FEATURE DC INDEX BITS]")
        print(f"  Index shape: {feature_dc_index_np.shape}")
        print(f"  Max index value: {index_max}")
        print(f"  Bits per index (calculated): {max_bit:.1f} bits (ceil(log2({index_max + 1})))")
        print(f"  Calculated (wo_ec): {feature_dc_index_bits_calculated:,} bits ({feature_dc_index_np.size} indices × {max_bit:.1f} bits)")
        print(f"  Actual (with EC):  {features_dc_index_bits_actual:,} bits (from quantizer.size() - codebook)")
        print(f"  Difference:        {feature_dc_index_bits_calculated - features_dc_index_bits_actual:,.0f} bits")
        print(f"  Compression ratio: {feature_dc_index_bits_calculated / features_dc_index_bits_actual:.2f}x" if features_dc_index_bits_actual > 0 else "  Compression ratio: N/A")
        print(f"  Full actual size:  {features_dc_actual_bits:,} bits (includes codebook)")

        # Cholesky quantized elements bits
        quant_cholesky_elements_np = quant_cholesky_elements.cpu().numpy()
        cholesky_elements_bits_calculated = quant_cholesky_elements_np.size * 6
        total_bits += cholesky_elements_bits_calculated

        # Get actual size from quantizer
        cholesky_actual_bits = cholesky_quantizer.size(quant_cholesky_elements)
        cholesky_elements_bits_actual = cholesky_actual_bits - cholesky_scale_bits - cholesky_beta_bits  # Remove overhead

        print(f"\n[CHOLESKY QUANTIZED ELEMENTS BITS]")
        print(f"  Elements shape: {quant_cholesky_elements_np.shape}")
        print(f"  Calculated (wo_ec): {cholesky_elements_bits_calculated:,} bits ({quant_cholesky_elements_np.size} elements × 6 bits)")
        print(f"  Actual (with EC):   {cholesky_elements_bits_actual:,} bits (from quantizer.size() - scale - beta)")
        print(f"  Difference:         {cholesky_elements_bits_calculated - cholesky_elements_bits_actual:,.0f} bits")
        print(f"  Compression ratio: {cholesky_elements_bits_calculated / cholesky_elements_bits_actual:.2f}x" if cholesky_elements_bits_actual > 0 else "  Compression ratio: N/A")
        print(f"  Full actual size:   {cholesky_actual_bits:,} bits (includes scale + beta)")

        # Component-wise breakdown
        cholesky_bits = (
            cholesky_quantizer.scale.numel() * torch.finfo(cholesky_quantizer.scale.dtype).bits +
            cholesky_quantizer.beta.numel() * torch.finfo(cholesky_quantizer.beta.dtype).bits +
            quant_cholesky_elements_np.size * 6
        )

        feature_dc_bits = codebook_bits + feature_dc_index_np.size * max_bit

        # Total comparison
        total_bits_actual = (
            initial_bits +
            position_bits +
            features_dc_actual_bits +
            cholesky_elements_bits_actual
        )

        print(f"\n[TOTAL BITS COMPARISON]")
        print(f"  Calculated (wo_ec): {total_bits:,} bits")
        print(f"  Actual (with EC):   {total_bits_actual:,} bits")
        print(f"  Difference:         {total_bits - total_bits_actual:,.0f} bits")
        print(f"  Compression ratio: {total_bits / total_bits_actual:.2f}x" if total_bits_actual > 0 else "  Compression ratio: N/A")

        # BPP calculation
        bpp = total_bits / (self.H * self.W * self.T)
        position_bpp = position_bits / (self.H * self.W * self.T)
        cholesky_bpp = cholesky_bits / (self.H * self.W * self.T)
        feature_dc_bpp = feature_dc_bits / (self.H * self.W * self.T)
        
        bpp_actual = total_bits_actual / (self.H * self.W * self.T)
        cholesky_bpp_actual = cholesky_actual_bits / (self.H * self.W * self.T)
        feature_dc_bpp_actual = features_dc_actual_bits / (self.H * self.W * self.T)

        print(f"\n[BITS PER PIXEL (BPP) COMPARISON]")
        print(f"  Video dimensions: {self.H} × {self.W} × {self.T} = {self.H * self.W * self.T:,} pixels")
        print(f"  Position BPP:      {position_bpp:.6f} (same for both)")
        print(f"  Cholesky BPP:")
        print(f"    Calculated:     {cholesky_bpp:.6f}")
        print(f"    Actual:         {cholesky_bpp_actual:.6f}")
        print(f"  Feature DC BPP:")
        print(f"    Calculated:     {feature_dc_bpp:.6f}")
        print(f"    Actual:         {feature_dc_bpp_actual:.6f}")
        print(f"  Total BPP:")
        print(f"    Calculated:     {bpp:.6f}")
        print(f"    Actual:         {bpp_actual:.6f}")
        print(f"    Difference:     {bpp - bpp_actual:.6f} ({((bpp - bpp_actual) / bpp * 100):.1f}% reduction)")

        print(f"\n{'='*80}\n")

        return {
            "bpp": bpp,
            "position_bpp": position_bpp,
            "cholesky_bpp": cholesky_bpp,
            "feature_dc_bpp": feature_dc_bpp,
        }