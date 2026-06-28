import argparse
import torch
from tqdm.auto import tqdm

from attacks import ATTACKS, CLID_POSTPROCESS
from attacks.common import att_measure, init_model
from dataset import (
    clid_load_coco_datasets,
    clid_load_flickr_datasets,
    clid_load_pokemon_datasets,
    load_coco_datasets,
    load_flickr_datasets,
    load_pokemon_datasets,
)
from Metrics import result


DEFAULT_CONFIG = {
    "T": 1000,
    "train_batch_size": 1,
    "dataloader_num_workers": 0,
    "resolution": 512,
    "image_column": "image",
    "caption_column": "text",
    "t_sec": 100,
    "timestep": 10,
    "stpsnumi": 1,
    "even_num": 10,
    "max_n_samples": 3,
    "max_clid_samples": 3,
    "trials_eacht": 1,
    "outdir": "outputs",
    "dataset": "coco",
    "diff_path": "/home/storage/model/sd1.4_coco",
    "dataset_root": "/home/storage/datasets/",
}

NUM_SAMPLES = {"pokemon": 415, "coco": 25, "flickr": 1000}
DATASET_LOADERS = {
    False: {
        "pokemon": load_pokemon_datasets,
        "coco": load_coco_datasets,
        "flickr": load_flickr_datasets,
    },
    True: {
        "pokemon": clid_load_pokemon_datasets,
        "coco": clid_load_coco_datasets,
        "flickr": clid_load_flickr_datasets,
    },
}


def resolve_device(device=None):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def load_attack_dataloaders(cfg, tokenizer):
    loader = DATASET_LOADERS[cfg["attack"] == "clid"][cfg["dataset"]]
    loaded = loader(cfg["dataset_root"], num_samples=NUM_SAMPLES[cfg["dataset"]], tokenizer=tokenizer)
    return loaded[-2], loaded[-1]


def collect_scores(cfg, models, train_loader, test_loader, filter_enabled, threshold, scale):
    if cfg["attack"] == "clid":
        train = run_split(train_loader, cfg, models, filter_enabled, threshold, scale)
        test = run_split(test_loader, cfg, models, filter_enabled, threshold, scale)
        train_out, test_out = CLID_POSTPROCESS["get_l_clidavg_last3"](train, test)
        return CLID_POSTPROCESS["deal_data_weight_avg"](train_out, test_out, alpha=0.4)

    member_a, member_b = run_split(train_loader, cfg, models, filter_enabled, threshold, scale)
    nonmember_a, nonmember_b = run_split(test_loader, cfg, models, filter_enabled, threshold, scale)
    norm = "l2"
    member_scores = att_measure(torch.concat(member_a), torch.concat(member_b), norm, device=cfg["device"]).cpu()
    nonmember_scores = att_measure(torch.concat(nonmember_a), torch.concat(nonmember_b), norm, device=cfg["device"]).cpu()
    return member_scores, nonmember_scores


def run_split(loader, cfg, models, filter_enabled, threshold, scale):
    attack_fn = ATTACKS[cfg["attack"]]
    if cfg["attack"] == "clid":
        return [attack_fn(batch, models, cfg, filter_enabled, threshold, scale) for batch in tqdm(loader)]

    left = []
    right = []
    for batch in tqdm(loader):
        attack_left, attack_right = attack_fn(batch, models, cfg, filter_enabled, threshold, scale)
        left.append(attack_left)
        right.append(attack_right)
    return left, right


def run(filter=1, t=5, s=0.2, attack="naive", dataset=None, diff_path=None, dataset_root=None, device=None):
    cfg = DEFAULT_CONFIG.copy()
    cfg["device"] = resolve_device(device)
    cfg["attack"] = attack
    cfg["filter"] = filter

    if dataset is not None:
        cfg["dataset"] = dataset
    if diff_path is not None:
        cfg["diff_path"] = diff_path
    if dataset_root is not None:
        cfg["dataset_root"] = dataset_root

    models = init_model(cfg["diff_path"], cfg["device"])
    print("loading finish!")

    train_loader, test_loader = load_attack_dataloaders(cfg, models["tokenizer"])
    print("start attack!")
    print(cfg["attack"])
    print(cfg["dataset"])

    member_scores, nonmember_scores = collect_scores(
        cfg,
        models,
        train_loader,
        test_loader,
        filter_enabled=bool(filter),
        threshold=t,
        scale=s,
    )
    return result(member_scores, nonmember_scores)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stable Diffusion membership inference attack.")
    parser.add_argument("--attack", default="naive", choices=["naive", "pia", "sec", "clid"])
    parser.add_argument("--dataset", default="pokemon", choices=["pokemon", "coco", "flickr"])
    parser.add_argument("--diff-path", default="/home/storage/model/sd1.4_coco_aug")
    parser.add_argument("--dataset-root", default="/home/storage/datasets/")
    parser.add_argument("--filter", type=int, default=1, choices=[0, 1])
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--s", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run(
        filter=args.filter,
        t=args.t,
        s=args.s,
        attack=args.attack,
        dataset=args.dataset,
        diff_path=args.diff_path,
        dataset_root=args.dataset_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()
