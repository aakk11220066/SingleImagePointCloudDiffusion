import torch

import DiffRender.Rasterizer


def get_default_rasterizer(img_height, img_width, angle_x=0.25*torch.pi, angle_y=-0.15*torch.pi, angle_z=0.5*torch.pi):
    return torch.nn.Sequential(
        DiffRender.Rasterizer.RotateModule(angle_x, angle_y, angle_z),
        DiffRender.Rasterizer.DifferentiableRaster(height=img_height, width=img_width)
    )
        
        
        