import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray).to(device=ray_bundle.origins.device).view(1, -1, 1)

        # TODO (Q1.4): Sample points from z values
        sample_points = ray_bundle.origins.unsqueeze(1).expand(-1, self.n_pts_per_ray, 3) \
            + ray_bundle.directions.unsqueeze(1).expand(-1, self.n_pts_per_ray, 3) * z_vals

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )

    def hierarchical_sample(self, ray_bundle, weights, N_fine, det=False):
        z_vals = ray_bundle.sample_lengths
        B, N = weights.shape

        # Compute PDF (add small epsilon for numerical stability)
        weights = weights + 1e-5
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [B, N]

        # Compute CDF
        cdf = torch.cumsum(pdf, dim=-1)  # [B, N]
        cdf = torch.cat([torch.zeros(B, 1, device=weights.device), cdf], dim=-1)  # [B, N+1]

        # Construct bins using midpoints of z_vals
        mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])  # [B, N-1]
        bins = torch.cat([z_vals[..., :1], mid, z_vals[..., -1:]], dim=-1)  # [B, N+1]

        # Sample uniform numbers
        if det:
            u = torch.linspace(0., 1., steps=N_fine, device=weights.device)
            u = u.expand(B, N_fine)
        else:
            u = torch.rand(B, N_fine, device=weights.device)

        # Invert CDF
        inds = torch.searchsorted(cdf, u, right=True)  # [B, N_fine]
        below = torch.clamp(inds-1, min=0)
        above = torch.clamp(inds, max=N)
        inds_g = torch.stack([below, above], dim=-1)  # [B, N_fine, 2]

        # Gather cdf and bins
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(B, N_fine, N+1), 2, inds_g)  # [B, N_fine, 2]
        bins_g = torch.gather(bins.unsqueeze(1).expand(B, N_fine, N+1), 2, inds_g)  # [B, N_fine, 2]

        denom = cdf_g[...,1] - cdf_g[...,0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[...,0]) / denom

        z_samples = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0])  # [B, N_fine]

        z_vals = torch.cat([z_vals, z_samples], dim=-1)  # [B, N + N_fine]
        z_vals, _ = torch.sort(z_vals, dim=-1)          # optional: sort to keep increasing order along ray

        sample_points = ray_bundle.origins.unsqueeze(1).expand(-1, self.n_pts_per_ray + N_fine, 3) \
            + ray_bundle.directions.unsqueeze(1).expand(-1, self.n_pts_per_ray + N_fine, 3) * z_vals.unsqueeze(-1)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )

sampler_dict = {
    'stratified': StratifiedRaysampler
}