# Enhancing Membership Inference Attacks on Diffusion Models from a Frequency-Domain Perspective

Code for the ICML 2026 paper **Enhancing Membership Inference Attacks on Diffusion Models from a Frequency-Domain Perspective**.

## Install

Create a Python environment:

```bash
conda create -n fremia python=3.10 -y
conda activate fremia
```

Install dependencies:

```bash
pip install -r requirements.txt
```


## Project Layout

```text
DDIM/                 DDIM attack implementation
Stable_Diffusion/     Stable Diffusion attack implementation
train/                DDPM training and evaluation code
```

## Run DDIM Attack

```bash
cd DDIM
python attack.py \
  --checkpoint ../train/experiments/TINY-IN/checkpoint.pt \
  --dataset TINY-IN \
  --attacker-name naive \
  --filter 0 \
  --t 5 \
  --s 0.2
```

Choices:

```text
--dataset: TINY-IN, CIFAR100, STL10-U
--attacker-name: naive, pia, sec
```

## Run Stable Diffusion Attack

```bash
cd Stable_Diffusion
python attack.py \
  --attack naive \
  --dataset coco \
  --diff-path /path/to/stable-diffusion-checkpoint \
  --dataset-root /path/to/datasets \
  --filter 1 \
  --t 5 \
  --s 0.2
```

Choices:

```text
--attack: naive, pia, sec, clid
--dataset: pokemon, coco, flickr
```


## Notes

- Fourier filtering is controlled by `--filter`, `--t`, and `--s`.
- To fine-tune a Stable Diffusion model, follow the official Diffusers `train_text_to_image.py` example.
- To train DDPM models, follow the training code and configuration under `train/`.
