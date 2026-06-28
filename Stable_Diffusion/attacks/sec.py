import torch

from attacks.common import ddim_multistep, ddim_singlestep, encode_latents
from filter import Fourier_filter


@torch.no_grad()
def run(batch, models, cfg, filter_enabled, threshold, scale):
    unet = models["unet"]
    vae = models["vae"]
    text_encoder = models["text_encoder"]
    scheduler = models["scheduler"]
    device = cfg["device"]

    latents, embeddings = encode_latents(batch, vae, text_encoder, device)
    target_steps = list(range(0, cfg["t_sec"] + cfg["timestep"], cfg["timestep"]))[1:]
    x_sec = ddim_multistep(
        unet,
        scheduler,
        latents,
        t_c=0,
        target_steps=target_steps,
        device=device,
        encoder_hidden_states=embeddings,
    )["x_t_target"]

    forward_t = cfg["t_sec"]
    backward_t = forward_t + cfg["stpsnumi"]
    x_sec_forw = ddim_singlestep(
        unet,
        scheduler,
        x_sec,
        t_c=forward_t,
        t_target=backward_t,
        device=device,
        encoder_hidden_states=embeddings,
    )["x_t_target"]
    x_sec_recon = ddim_singlestep(
        unet,
        scheduler,
        x_sec_forw,
        t_c=backward_t,
        t_target=forward_t,
        device=device,
        encoder_hidden_states=embeddings,
    )["x_t_target"]

    if filter_enabled:
        x_sec = Fourier_filter(x_sec, threshold=threshold, scale=scale)
        x_sec_recon = Fourier_filter(x_sec_recon, threshold=threshold, scale=scale)

    return x_sec, x_sec_recon
