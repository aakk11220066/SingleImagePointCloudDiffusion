import torch

from ModelDiffusion.diffusion.models.diffusion import DiffusionPoint, PointwiseNet, VarianceSchedule


class DiffusionPointWrapper(DiffusionPoint):
    def __init__(self, sample_num_points, ckpt):
        super().__init__(
            net = PointwiseNet(point_dim=3, context_dim=ckpt['args'].latent_dim, residual=ckpt['args'].residual),
            var_sched = VarianceSchedule(
                num_steps=ckpt['args'].num_steps,
                beta_1=ckpt['args'].beta_1,
                beta_T=ckpt['args'].beta_T,
                mode=ckpt['args'].sched_mode
            )
        )
        self.sample_num_points = sample_num_points
        self.flexibility = ckpt["args"].flexibility
        
    def sample(self, context):
        return super().sample(        
            num_points=self.sample_num_points, 
            context=context, 
            flexibility=self.flexibility
        )
        
    def forward(self, noised_pointcloud, context, denoising_step_number):
        # Performs one denoising step
        beta = self.var_sched.betas[denoising_step_number]
        alpha_bar = self.var_sched.alpha_bars[denoising_step_number]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        
        estimated_pointcloud_noise = self.net(noised_pointcloud, beta=beta, context=context)
        denoised_pointcloud = (noised_pointcloud - estimated_pointcloud_noise) / c0
        return denoised_pointcloud


def get_default_diffusion_model(sample_num_points, device, category="airplane"):
    print('Loading model...')
    ckpt = torch.load(f'./diffusion/pretrained/GEN_{category}.pt')
    model = DiffusionPointWrapper(sample_num_points, ckpt)
    diffusion_weight_specifier = "diffusion."
    model.load_state_dict({
        diffusion_key[len(diffusion_weight_specifier):]: ckpt['state_dict'][diffusion_key] 
        for diffusion_key in ckpt['state_dict'].keys() 
        if diffusion_key[:len(diffusion_weight_specifier)] == diffusion_weight_specifier
    })
    print("Loaded")
    return model.to(device)