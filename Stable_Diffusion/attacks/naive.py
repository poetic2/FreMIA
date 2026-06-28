import torch

from attacks.common import encode_latents, extract
from filter import Fourier_filter


@torch.no_grad()
def run(batch, models, cfg, filter_enabled, threshold, scale):
    unet = models["unet"]
    vae = models["vae"]
    text_encoder = models["text_encoder"]
    scheduler = models["scheduler"]
    device = cfg["device"]

    latents, embeddings = encode_latents(batch, vae, text_encoder, device)
    t = latents.new_ones([latents.shape[0]], dtype=torch.long) * 500

    betas = scheduler.betas.double().to(device)
    alphas_t = extract(torch.cumprod(1.0 - betas, dim=0), t=t, x_shape=latents.shape)
    noise = torch.randn_like(latents)
    noise_pred = unet(
        alphas_t.sqrt() * latents + (1 - alphas_t).sqrt() * noise,
        t,
        embeddings,
    ).sample

    eps_pred = (
        alphas_t.sqrt() * latents
        + (1 - alphas_t).sqrt() * noise
        - (1 - alphas_t).sqrt() * noise_pred
    ) / alphas_t.sqrt()
    eps = latents

    if filter_enabled:
        eps = Fourier_filter(eps, threshold=threshold, scale=scale)
        eps_pred = Fourier_filter(eps_pred, threshold=threshold, scale=scale)

    return eps, eps_pred
