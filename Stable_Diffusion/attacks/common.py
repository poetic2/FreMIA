import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def init_model(diff_path, device):
    vae = AutoencoderKL.from_pretrained(diff_path, subfolder="vae", use_auth_token=True)
    tokenizer = CLIPTokenizer.from_pretrained(diff_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(diff_path, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(diff_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(diff_path, subfolder="scheduler")

    vae = vae.to(device).eval()
    text_encoder = text_encoder.to(device).eval()
    unet = unet.to(device).eval()
    print("all model loaded.")

    return {
        "vae": vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "unet": unet,
        "scheduler": scheduler,
    }


@torch.no_grad()
def extract(v, t, x_shape):
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


@torch.no_grad()
def ddim_singlestep(model, scheduler, x, t_c, t_target, device, encoder_hidden_states):
    if encoder_hidden_states is None:
        raise ValueError("encoder_hidden_states must be provided.")

    x = x.to(device)
    t_c = x.new_ones([x.shape[0]], dtype=torch.long) * t_c
    t_target = x.new_ones([x.shape[0]], dtype=torch.long) * t_target

    betas = scheduler.betas.double().to(device)
    alphas = torch.cumprod(1.0 - betas, dim=0)
    alphas_t_c = extract(alphas, t=t_c, x_shape=x.shape)
    alphas_t_target = extract(alphas, t=t_target, x_shape=x.shape)

    epsilon = model(x, t_c, encoder_hidden_states).sample
    pred_x_0 = (x - ((1 - alphas_t_c).sqrt() * epsilon)) / alphas_t_c.sqrt()
    x_t_target = alphas_t_target.sqrt() * pred_x_0 + (1 - alphas_t_target).sqrt() * epsilon
    return {"x_t_target": x_t_target, "epsilon": epsilon}


@torch.no_grad()
def ddim_multistep(model, scheduler, x, t_c, target_steps, device, encoder_hidden_states, clip=False):
    result = None
    for t_target in target_steps:
        result = ddim_singlestep(model, scheduler, x, t_c, t_target, device, encoder_hidden_states)
        x = result["x_t_target"]
        t_c = t_target

    if clip and result is not None:
        result["x_t_target"] = torch.clip(result["x_t_target"], -1, 1)

    return result


def encode_latents(batch, vae, text_encoder, device):
    batch["pixel_values"] = batch["pixel_values"].to(device)
    latents = vae.encode(batch["pixel_values"].to(torch.float32)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    embeddings = text_encoder(batch["input_ids"].to(device))[0]
    return latents, embeddings


def decode_latents(latents, vae):
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample
    return (image / 2 + 0.5).clamp(0, 1)


@torch.no_grad()
def att_measure(diffusion, sample, metric, device):
    diffusion = diffusion.to(device).float()
    sample = sample.to(device).float()

    if len(diffusion.shape) == 5:
        num_timestep = diffusion.size(0)
        diffusion = diffusion.permute(1, 0, 2, 3, 4).reshape(-1, num_timestep * 3, 32, 32)
        sample = sample.permute(1, 0, 2, 3, 4).reshape(-1, num_timestep * 3, 32, 32)

    if metric == "l2":
        return ((diffusion - sample) ** 2).flatten(1).sum(dim=-1)
    if metric == "cos":
        return 1 - torch.nn.functional.cosine_similarity(diffusion.flatten(1), sample.flatten(1), dim=-1)
    if isinstance(metric, int):
        return (torch.abs(diffusion - sample) ** metric).flatten(1).sum(dim=-1)

    raise NotImplementedError(f"Unsupported metric: {metric}")
