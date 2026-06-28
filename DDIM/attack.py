import argparse
import numpy as np
import random
import torch
import components
from typing import Type, Dict
from model import UNet
from dataset_utils import load_member_data
from torchmetrics.classification import BinaryAUROC, BinaryROC
from tqdm import tqdm


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EpsGetter(components.EpsGetter):
    def __call__(self, xt: torch.Tensor, condition: torch.Tensor = None, noise_level=None, t: int = None) -> torch.Tensor:
        t = torch.ones([xt.shape[0]], device=xt.device).long() * t
        return self.model(xt, t=t)


attackers: Dict[str, Type[components.DDIMAttacker]] = {
    "sec": components.SecMIAttacker,
    "pia": components.PIA,
    "naive": components.NaiveAttacker,
}

MODEL_CONFIG = {
    "T": 1000,
    "ch": 128,
    "ch_mult": [1, 2, 2, 2],
    "attn": [1],
    "num_res_blocks": 2,
    "dropout": 0.1,
    "beta_1": 0.0001,
    "beta_T": 0.02,
}

DATASETS = {
    "TINY-IN": "TINY-IN",
    "CIFAR100": "CIFAR100",
    "STL10-U": "STL10-U",
}


def get_model(ckpt, WA=True, config=None):
    config = config or MODEL_CONFIG
    model = UNet(
        T=config["T"],
        ch=config["ch"],
        ch_mult=config["ch_mult"],
        attn=config["attn"],
        num_res_blocks=config["num_res_blocks"],
        dropout=config["dropout"],
    )
    ckpt = torch.load(ckpt)

    weights = ckpt["ema_model"] if WA else ckpt["net_model"]
    new_state_dict = {key[7:] if key.startswith("module.") else key: val for key, val in weights.items()}
    model.load_state_dict(new_state_dict)
    return model.eval()


def load_attack_data(dataset, batch_size=64):
    if dataset not in DATASETS:
        raise NotImplementedError(f"Unsupported dataset: {dataset}")
    return load_member_data(dataset_name=DATASETS[dataset], batch_size=batch_size, shuffle=False, randaugment=False)


def build_attacker(attacker_name, device, interval, attack_num, model, Filter, t, s):
    if attacker_name not in attackers:
        raise NotImplementedError(f"Unsupported attacker: {attacker_name}")

    betas = torch.from_numpy(
        np.linspace(MODEL_CONFIG["beta_1"], MODEL_CONFIG["beta_T"], MODEL_CONFIG["T"])
    ).to(device)
    return attackers[attacker_name](
        betas,
        interval,
        attack_num,
        EpsGetter(model),
        lambda x: x * 2 - 1,
        Filter=Filter,
        t=t,
        s=s,
    )


def evaluate_roc(member, nonmember, device):
    auroc_labels = torch.cat(
        [
            torch.zeros(member.shape[1], dtype=torch.long, device=device),
            torch.ones(nonmember.shape[1], dtype=torch.long, device=device),
        ]
    )
    roc_labels = torch.cat(
        [
            torch.zeros(nonmember.shape[1], dtype=torch.long, device=device),
            torch.ones(member.shape[1], dtype=torch.long, device=device),
        ]
    )

    auroc = []
    tpr_fpr = []
    for i in range(member.shape[0]):
        max_score = max(member[i].max().item(), nonmember[i].max().item())
        member_score = member[i] / max_score
        nonmember_score = nonmember[i] / max_score

        auroc.append(BinaryAUROC().to(device)(torch.cat([member_score, nonmember_score]), auroc_labels).item())
        tpr_fpr.append(BinaryROC().to(device)(torch.cat([1 - nonmember_score, 1 - member_score]), roc_labels))

    return auroc, tpr_fpr


@torch.no_grad()
def DDIM_Attack(
    checkpoint='../train/experiments/TINY-IN/checkpoint.pt',
    dataset='TINY-IN',
    attacker_name="PIA",
    Filter=0,
    t=5,
    s=0.2,
    attack_num=1,
    interval=100,
    seed=0,
    device=None,
    batch_size=64,
):
    set_seeds(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("loading model...")
    model = get_model(checkpoint, WA=True).to(device)
    model.eval()

    print("loading dataset...")
    _, _, train_loader, test_loader = load_attack_data(dataset, batch_size=batch_size)

    attacker = build_attacker(attacker_name, device, interval, attack_num, model, Filter, t, s)

    print("attack start...")
    members, nonmembers = [], []
    for member, nonmember in tqdm(zip(train_loader, test_loader), total=len(train_loader)):
        member, nonmember = member[0].to(device), nonmember[0].to(device)

        members.append(attacker(member))
        nonmembers.append(attacker(nonmember))

        members = [torch.cat(members, dim=-1)]
        nonmembers = [torch.cat(nonmembers, dim=-1)]

    member = members[0]
    nonmember = nonmembers[0]

    auroc, tpr_fpr = evaluate_roc(member, nonmember, device)
    tpr_fpr_1 = [i[1][(i[0] < 0.01).sum() - 1].item() for i in tpr_fpr]
    cp_tpr_fpr_1 = tpr_fpr_1[:]

    print('auc', auroc)
    print('tpr @ 1% fpr', cp_tpr_fpr_1)


    n = member.shape[0]
    asr_list = []

    for i in range(n):
        member_scores = member[i, :]
        nonmember_scores = nonmember[i, :]

        min_score = min(member_scores.min(), nonmember_scores.min()).item()
        max_score = max(member_scores.max(), nonmember_scores.max()).item()

        best_asr = 0
        for threshold in torch.arange(min_score, max_score, (max_score - min_score) / 2000, device=device):
        # threshold = torch.tensor(xxx, device=device)

            TP = (member_scores <= threshold).sum().item()
            TN = (nonmember_scores > threshold).sum().item()
            FP = (nonmember_scores <= threshold).sum().item()
            FN = (member_scores > threshold).sum().item()

            ASR = (TP + TN) / (TP + TN + FP + FN)
            if ASR > best_asr:
                best_asr = ASR

        asr_list.append(best_asr)

    print("ASR list:", asr_list)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DDIM membership inference attack.")
    parser.add_argument("--checkpoint", default="../train/experiments/TINY-IN/checkpoint.pt")
    parser.add_argument("--dataset", default="TINY-IN", choices=["TINY-IN", "CIFAR100", "STL10-U"])
    parser.add_argument("--attacker-name", default="naive", choices=["naive", "pia", "sec"])
    parser.add_argument("--filter", type=int, default=1, choices=[0, 1])
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--s", type=float, default=0.2)
    parser.add_argument("--attack-num", type=int, default=1)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    DDIM_Attack(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        attacker_name=args.attacker_name,
        Filter=args.filter,
        t=args.t,
        s=args.s,
        attack_num=args.attack_num,
        interval=args.interval,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
