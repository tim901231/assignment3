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
        """
        Perform hierarchical sampling along rays using coarse weights.

        Args:
            ray_bundle: namedtuple with ray origins, directions, and coarse z_vals (sample_lengths)
            weights: coarse weights along each ray, shape [B, N_coarse]
            N_fine: number of fine samples to draw
            det: whether to use deterministic sampling (linspace) or random
        Returns:
            Updated ray_bundle with fine sample points and z_vals
        """
        z_vals = ray_bundle.sample_lengths.squeeze(-1)  # [B, N_coarse]
        B, N = weights.shape

        # --- Compute PDF and CDF ---
        weights = weights + 1e-5  # avoid zeros
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [B, N]
        cdf = torch.cumsum(pdf, dim=-1)  # [B, N]
        cdf = torch.cat([torch.zeros(B, 1, device=weights.device), cdf], dim=-1)  # [B, N+1]

        # --- Construct bins using midpoints ---
        mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])  # [B, N-1]
        bins = torch.cat([z_vals[..., :1], mid, z_vals[..., -1:]], dim=-1)  # [B, N+1]

        # --- Sample uniform numbers ---
        if det:
            u = torch.linspace(0., 1., steps=N_fine, device=weights.device)
            u = u.expand(B, N_fine)  # [B, N_fine]
        else:
            u = torch.rand(B, N_fine, device=weights.device)  # [B, N_fine]

        # --- Invert CDF to get sample indices ---
        inds = torch.searchsorted(cdf, u, right=True)  # [B, N_fine]
        below = torch.clamp(inds - 1, min=0)
        above = torch.clamp(inds, max=N)
        inds_g = torch.stack([below, above], dim=-1)  # [B, N_fine, 2]

        # --- Gather CDF and bins at sampled indices ---
        # Use unsqueeze + expand safely
        cdf_exp = cdf.unsqueeze(1).expand(-1, N_fine, -1)  # [B, N_fine, N+1]
        bins_exp = bins.unsqueeze(1).expand(-1, N_fine, -1)  # [B, N_fine, N+1]
        cdf_g = torch.gather(cdf_exp, 2, inds_g)  # [B, N_fine, 2]
        bins_g = torch.gather(bins_exp, 2, inds_g)  # [B, N_fine, 2]

        # --- Linear interpolation to get z_samples ---
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        z_samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])  # [B, N_fine]

        # --- Combine coarse and fine samples, sort along ray ---
        z_vals_combined = torch.cat([z_vals, z_samples], dim=-1)  # [B, N + N_fine]
        z_vals_combined, _ = torch.sort(z_vals_combined, dim=-1)

        # --- Compute sample points in 3D space ---
        sample_points = ray_bundle.origins.unsqueeze(1) + ray_bundle.directions.unsqueeze(1) * z_vals_combined.unsqueeze(-1)

        # --- Return updated ray bundle ---
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals_combined.unsqueeze(-1),  # keep shape [B, N_total, 1]
        )

sampler_dict = {
    'stratified': StratifiedRaysampler
}