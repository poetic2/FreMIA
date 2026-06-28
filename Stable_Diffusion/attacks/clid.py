import numpy as np
import torch
import torch.nn.functional as F

from filter import Fourier_filter


@torch.no_grad()
def run(batch, models, cfg, filter_enabled, threshold, scale):
    unet = models["unet"]
    vae = models["vae"]
    text_encoder = models["text_encoder"]
    scheduler = models["scheduler"]
    device = cfg["device"]

    batch["pixel_values"] = batch["pixel_values"].to(device)
    latents = vae.encode(batch["pixel_values"].to(torch.float32)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    t_to_eval, t_clid_to_eval, start_idx = _clid_timesteps(cfg)
    batch_loss = {"cond0": [], "cond1_dif": [], "cond2_dif": [], "cond3_dif": [], "condNull_dif": []}

    input_groups = zip(
        latents,
        batch["input_ids"],
        batch["input_ids_1"],
        batch["input_ids_2"],
        batch["input_ids_3"],
        batch["input_ids_null"],
    )
    for latent, input_ids, input_ids_1, input_ids_2, input_ids_3, input_ids_null in input_groups:
        cond0_loss, noise, noise_other = _cond0_loss(
            latent,
            input_ids,
            latents,
            t_to_eval,
            start_idx,
            models,
            cfg,
            filter_enabled,
            threshold,
            scale,
        )
        batch_loss["cond0"].append(cond0_loss)

        for other_ids, name in zip(
            [input_ids_1, input_ids_2, input_ids_3, input_ids_null],
            ["cond1_dif", "cond2_dif", "cond3_dif", "condNull_dif"],
        ):
            other_loss = _condition_loss(
                latent,
                other_ids,
                latents,
                t_clid_to_eval,
                noise_other,
                models,
                cfg,
                filter_enabled,
                threshold,
                scale,
            )
            batch_loss[name].append(other_loss - cond0_loss)

    return batch_loss


def _clid_timesteps(cfg):
    start = cfg["T"] // 2 - (cfg["even_num"] * cfg["max_n_samples"] // 2)
    t_to_eval = np.array(list(range(start, cfg["T"], cfg["even_num"]))[: cfg["max_n_samples"]])
    start_idx = len(t_to_eval) // 2 - cfg["max_clid_samples"] // 2
    t_clid_to_eval = t_to_eval[[start_idx + i for i in range(cfg["max_clid_samples"])]]
    return t_to_eval, list(t_clid_to_eval), start_idx


def _cond0_loss(latent, input_ids, latents, t_to_eval, start_idx, models, cfg, filter_enabled, threshold, scale):
    device = cfg["device"]
    noise = torch.randn(len(t_to_eval), 4, 64, 64, device=device)
    noise_other = noise[[start_idx + i for i in range(cfg["max_clid_samples"])]]
    loss = _condition_loss(
        latent,
        input_ids,
        latents,
        t_to_eval,
        noise,
        models,
        cfg,
        filter_enabled,
        threshold,
        scale,
    )
    return loss, noise, noise_other


def _condition_loss(latent, input_ids, latents, timesteps, noise, models, cfg, filter_enabled, threshold, scale):
    unet = models["unet"]
    text_encoder = models["text_encoder"]
    scheduler = models["scheduler"]
    device = cfg["device"]

    ts = torch.tensor(np.concatenate([timesteps] * cfg["trials_eacht"])).long()
    pixel_mtcl = latent.view(-1, 4, 64, 64).expand(len(timesteps), 4, 64, 64)
    x_mtcl = scheduler.add_noise(pixel_mtcl.to(device), noise.to(device), ts.to(device))
    input_id_mtcl = input_ids.expand(len(timesteps), -1)
    embeddings = text_encoder(input_id_mtcl.to(device))[0]

    noise_pred = unet(x_mtcl, ts.to(device), embeddings).sample
    alphas_t = scheduler.alphas_cumprod.to(device)[ts.to(device)].view(-1, 1, 1, 1)
    eps_pred = (
        torch.sqrt(alphas_t) * latents
        + torch.sqrt(1 - alphas_t) * noise
        - torch.sqrt(1 - alphas_t) * noise_pred
    ) / torch.sqrt(alphas_t)
    eps = latents

    if filter_enabled:
        eps = Fourier_filter(eps, threshold=threshold, scale=scale)
        eps_pred = Fourier_filter(eps_pred, threshold=threshold, scale=scale)

    return float(F.mse_loss(eps.float(), eps_pred.float().to(device), reduction="mean").detach().cpu())


def get_l_clidavg_last3(train, test):
    return _process_clid_samples(train), _process_clid_samples(test)


def _process_clid_samples(samples):
    outputs = []
    for sample in samples:
        sample_id = sample["cond0"][0]
        other_values = [val[0] for key, val in sample.items() if key != "id"]
        outputs.append([sample_id, -sum(other_values)])
    return outputs


def deal_data_weight_avg(train, test, alpha):
    assert len(train[0]) == 2
    train = [(1 - alpha) * e[0] + alpha * e[1] for e in train]
    test = [(1 - alpha) * e[0] + alpha * e[1] for e in test]
    return train, test
