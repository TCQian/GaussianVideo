from gsplat.project_gaussians_video import project_gaussians_video
from gsplat.rasterize_sum_video import rasterize_gaussians_sum_video
from utils import *
import torch
import torch.nn as nn
import math
from optimizer import Adan
import torch.nn.functional as F
from utils import *

class GaussianVideo3D2D(nn.Module):
    def __init__(self, layer= 0, loss_type="L2", **kwargs):
        super().__init__()

        self.debug_mode = False 

        self.layer = int(layer)
        self.loss_type = loss_type
        self.iterations = kwargs["iterations"]
        self.opt_type = kwargs["opt_type"]
        self.lr = kwargs["lr"]

        self.H, self.W, self.T = kwargs["H"], kwargs["W"], kwargs["T"]
        self.BLOCK_W, self.BLOCK_H, self.BLOCK_T = kwargs["BLOCK_W"], kwargs["BLOCK_H"], kwargs["BLOCK_T"]
        
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W, # (1920 + 16 - 1) // 16 = 120
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H, # (1080 + 16 - 1) // 16 = 68
            (self.T + self.BLOCK_T - 1) // self.BLOCK_T, # (50 + 1 - 1) // 1 = 50
        ) # tile_bounds (120, 68, 50)
        self.device = kwargs["device"]

        self.init_num_points = kwargs["num_points"]
        
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

    def _create_data_from_checkpoint(self, checkpoint_path_layer0, checkpoint_path_layer1):
        if checkpoint_path_layer0 is not None:
            checkpoint_layer0 = torch.load(checkpoint_path_layer0, map_location=self.device)
        if checkpoint_path_layer1 is not None:
            checkpoint_layer1 = torch.load(checkpoint_path_layer1, map_location=self.device)

        if checkpoint_layer0 is not None:
            self._xyz_3D = checkpoint_layer0['_xyz_3D'].requires_grad_(False)
            self._cholesky_3D = checkpoint_layer0['_cholesky_3D'].requires_grad_(False)
            self._features_dc_3D = checkpoint_layer0['_features_dc_3D'].requires_grad_(False)
            self._opacity_3D = nn.Parameter(checkpoint_layer0['_opacity_3D'])
            print(f"Layer 0 checkpoint loaded successfully with {self._xyz_3D.shape[0]} gaussians")
        else:
            if self.layer == 0:
                self._init_layer0()
            elif self.layer == 1: # required in get func
                self._xyz_3D = torch.zeros(0, 3).requires_grad_(False)
                self._cholesky_3D = torch.zeros(0, 6).requires_grad_(False)
                self._features_dc_3D = torch.zeros(0, 3).requires_grad_(False)
                self._opacity_3D = nn.Parameter(torch.zeros(0, 1))
                print('GaussianVideo3D2D: Without layer 0 checkpoint, layer 1 will train with 0 3D gaussians')
            
        if checkpoint_layer1 is not None:
            self.num_points_list = checkpoint_layer1['gaussian_num_list']

            self._xyz_2D = checkpoint_layer1['_xyz_2D']
            z = []
            for t, (start, end) in enumerate(self.num_points_list):
                z.append(torch.atanh(torch.tensor((2 * (t / self.T)) - 1.0)).item())
            self._xyz_2D = torch.cat((self._xyz_2D.data[:, :2], torch.tensor(z).view(-1, 1)), dim=1)

            self._ckpt_cholesky_2D = checkpoint_layer1['_cholesky_2D']
            self._cholesky_2D = torch.rand(len(self._xyz_2D), 6)
            for t, (start, end) in enumerate(self.num_points_list):
                self._cholesky_2D[start:end, 0] = self._ckpt_cholesky_2D[start:end, 0] # l11
                self._cholesky_2D[start:end, 1] = self._ckpt_cholesky_2D[start:end, 1] # l12
                self._cholesky_2D[start:end, 3] = self._ckpt_cholesky_2D[start:end, 2] # l22
                self._cholesky_2D[start:end, 2] = 0 
                self._cholesky_2D[start:end, 4] = 0
                self._cholesky_2D[start:end, 5] = 1

            self._features_dc_2D = checkpoint_layer1['_features_dc_2D']
            self._opacity_3D = checkpoint_layer1['_opacity_3D']
            self._opacity_2D = checkpoint_layer1['_opacity_2D']
            print(f"Layer 1 checkpoint loaded successfully with {self._xyz_2D.shape[0]} gaussians")
        else:
            if self.layer == 1:
                self._init_layer1()

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def _init_layer0(self):
        self.init_num_points = int(self.num_points * self.T / 2)
        self._xyz_3D = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 3) - 0.5)))
        self._cholesky_3D = nn.Parameter(torch.rand(self.init_num_points, 6))
        self._opacity_3D = nn.Parameter(torch.logit(0.1 * torch.ones(self.init_num_points, 1)))
        self._features_dc_3D = nn.Parameter(torch.rand(self.init_num_points, 3))
        
        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            self.optimizer = Adan(self.parameters(), lr=self.lr)

        self.layer = 0
        print("Layer 0 initialized, number of gaussians: ", self._xyz_3D.shape[0])

    def _init_layer1(self):
        self.init_num_points = int((self.num_points * self.T) - self._xyz_3D.shape[0])
        num_points_per_frame = int(self.init_num_points / self.T)
        self.num_points_list = []
        for t in range(self.T):
            start = t * num_points_per_frame
            if t == self.T - 1:
                end = self.init_num_points
            else:
                end = start + num_points_per_frame
            self.num_points_list.append((start, end))

        self._xyz_2D = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 3) - 0.5)))
        for t, (start, end) in enumerate(self.num_points_list):
            self._xyz_2D.data[start:end, 2] = torch.atanh(torch.tensor((2 * (t / self.T)) - 1.0)).item()

        self._cholesky_2D = nn.Parameter(torch.rand(self.init_num_points, 6))
        with torch.no_grad():
            self._cholesky_2D.data[:, 2] = 0 # l31
            self._cholesky_2D.data[:, 4] = 0 # l32
            self._cholesky_2D.data[:, 5] = 1 # l33

        self._opacity_2D = nn.Parameter(torch.logit(0.1 * torch.ones(self.init_num_points, 1)))
        self._features_dc_2D = nn.Parameter(torch.rand(self.init_num_points, 3))

        trainable_params =[
            self._opacity_3D,
            self._xyz_2D,
            self._cholesky_2D,
            self._features_dc_2D,
            self._opacity_2D
        ]
        
        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr)
        else:
            self.optimizer = Adan(trainable_params, lr=self.lr)

        self.layer = 1
        print("GaussianVideo3D2D: Layer 1 initialized, number of gaussians: ", self._xyz_2D.shape[0])

    def _apply_2d_gaussian_constraints(self):
        """Apply gradient constraints to 2D gaussian parameters during training"""
        if self.layer == 1:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    if name == "_xyz_2D":
                        # Freeze temporal dimension (z-coordinate)
                        param.grad[:, 2] = 0
                        param.grad[:, 2].requires_grad_(False)
                    elif name == "_cholesky_2D":
                        # Freeze L13, L23, L33 elements (indices 2, 4, 5)
                        param.grad[:, [2, 4, 5]] = 0
                        param.grad[:, [2, 4, 5]].requires_grad_(False)

    def save_checkpoint(self, path):
        if self.layer == 0:
            layer0_state = {
                '_xyz_3D': self._xyz_3D.data,
                '_cholesky_3D': self._cholesky_3D.data,
                '_features_dc_3D': self._features_dc_3D.data,
                '_opacity_3D': self._opacity_3D.data,
                'layer': self.layer
            }
            torch.save(layer0_state, path / "layer_0_model.pth.tar")
            print(f"Layer 0 checkpoint saved to: {path / 'layer_0_model.pth.tar'}")
        elif self.layer == 1:
            layer1_state = {
                '_xyz_2D': self._xyz_2D.data[:, :2],
                '_cholesky_2D': self._cholesky_2D.data[:, [0, 1, 3]],
                '_features_dc_2D': self._features_dc_2D.data,
                '_opacity_2D': self._opacity_2D.data,
                '_opacity_3D': self._opacity_3D.data,
                'gaussian_num_list': self.num_points_list,
            }
            torch.save(layer1_state, path / "layer_1_model.pth.tar")
            print(f"Layer 1 checkpoint saved to: {path / f'layer_1_model.pth.tar'}")

    @property
    def get_xyz(self):
        if self.layer == 0:
            return torch.tanh(self._xyz_3D)
        elif self.layer == 1:
            return torch.tanh(torch.cat((self._xyz_3D, self._xyz_2D), dim=0))
    
    @property
    def get_features(self):
        if self.layer == 0:
            return self._features_dc_3D
        elif self.layer == 1:
            return torch.cat((self._features_dc_3D, self._features_dc_2D), dim=0)
    
    @property
    def get_opacity(self):
        if self.layer == 0:
            return self.opacity_activation(self._opacity_3D)
        elif self.layer == 1:
            return self.opacity_activation(torch.cat((self._opacity_3D, self._opacity_2D), dim=0))
    
    @property
    def get_cholesky_elements(self):
        if self.layer == 0:
            return self._cholesky_3D + self.cholesky_bound_3D
        elif self.layer == 1:
            merged_cholesky = torch.cat((self._cholesky_3D + self.cholesky_bound_3D, self._cholesky_2D + self.cholesky_bound_2D), dim=0)
            return merged_cholesky

    
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

            # update num_points_list
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
    
    def forward(self):
        self.xys, depths, radii, conics, num_tiles_hit = project_gaussians_video(
            self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.T, self.tile_bounds
        )
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

        if self.layer == 1:
            self._apply_2d_gaussian_constraints()

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        
        return loss, psnr

