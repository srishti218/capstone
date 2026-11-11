"""Train and evaluate DenseNet-121 (CXR) or MONAI 3D U-Net (CT)."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomAffine

from .models import (
    CLASS_VERSION, ROI, SEG_VERSION, class_tensor, classifier, segment_volume, segmenter,
)


class CxrDataset(Dataset):
    def __init__(self, root, cases, training=False):
        self.root, self.cases = root, cases
        self.augment = RandomAffine(7, translate=(0.04, 0.04), scale=(0.92, 1.08)) if training else None

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        with Image.open(self.root / case["image"]) as image:
            if self.augment is not None:
                image = self.augment(image)
            return class_tensor(image), np.float32(case["target"])


class CtDataset(Dataset):
    def __init__(self, root, cases, training=False):
        self.root, self.cases, self.training = root, cases, training

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        pixels = np.load(self.root / case["image"], mmap_mode="r", allow_pickle=False)
        mask = np.load(self.root / case["mask"], mmap_mode="r", allow_pickle=False)
        if self.training:
            pixels, mask = self.patch(pixels, mask)
        return (
            torch.from_numpy(pixels.astype(np.float32))[None],
            torch.from_numpy(mask.astype(np.float32))[None],
        )

    @staticmethod
    def patch(pixels, mask):
        pad = [(0, max(0, wanted - actual)) for wanted, actual in zip(ROI, pixels.shape)]
        if any(after for _, after in pad):
            pixels = np.pad(pixels, pad)
            mask = np.pad(mask, pad)
        foreground = np.flatnonzero(mask) if random.random() < 0.65 else []
        if len(foreground):
            center = np.unravel_index(random.choice(foreground), mask.shape)
            starts = [
                min(max(0, voxel - random.randint(size // 4, 3 * size // 4)), extent - size)
                for voxel, size, extent in zip(center, ROI, mask.shape)
            ]
        else:
            starts = [random.randint(0, extent - size) for size, extent in zip(ROI, mask.shape)]
        crop = tuple(slice(start, start + size) for start, size in zip(starts, ROI))
        pixels, mask = pixels[crop], mask[crop]
        if random.random() < 0.5:
            pixels, mask = pixels[:, :, ::-1], mask[:, :, ::-1]
        pixels = np.clip(pixels.astype(np.float32) + random.uniform(-0.05, 0.05), 0, 1)
        return pixels, mask


def class_metrics(targets, probabilities):
    predicted = probabilities >= 0.5
    positives, negatives = targets == 1, targets == 0
    return {
        "n": len(targets), "positives": int(positives.sum()),
        "roc_auc": float(roc_auc_score(targets, probabilities)) if positives.any() and negatives.any() else None,
        "average_precision": (
            float(average_precision_score(targets, probabilities))
            if positives.any() and negatives.any() else None
        ),
        "sensitivity": float(predicted[positives].mean()) if positives.any() else None,
        "specificity": float((~predicted[negatives]).mean()) if negatives.any() else None,
        "brier_score": float(np.mean((probabilities - targets) ** 2)),
    }


@torch.inference_mode()
def eval_cxr(model, loader, device):
    all_targets, all_scores = [], []
    for pixels, labels in loader:
        all_targets.extend(labels.tolist())
        all_scores.extend(model(pixels.to(device)).flatten().sigmoid().cpu().tolist())
    targets, scores = np.array(all_targets), np.array(all_scores)
    report = class_metrics(targets, scores)
    for attribute in ("view", "sex", "age"):
        groups = np.array([case[attribute] for case in loader.dataset.cases])
        report[f"by_{attribute}"] = {
            str(group): class_metrics(targets[groups == group], scores[groups == group])
            for group in np.unique(groups)
        }
    return report


def eval_ct(model, loader, device):
    scores = []
    for pixels, labels in loader:
        predicted = segment_volume(model, pixels, device) >= 0.5
        truth = labels.bool()
        total = int(predicted.sum()) + int(truth.sum())
        scores.append(2 * int((predicted & truth).sum()) / total if total else 1.0)
    return {"n": len(scores), "mean_dice": float(np.mean(scores)), "per_case_dice": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["cxr", "ct"])
    parser.add_argument("data", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--from-scratch", action="store_true", help="Skip ImageNet initialization for a CXR smoke test.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    manifest = json.loads((args.data / "split.json").read_text())
    if args.task == "cxr":
        dataset, model, evaluate = CxrDataset, classifier(pretrained=not args.from_scratch), eval_cxr
        version, metric = CLASS_VERSION, "roc_auc"
        positives = sum(item["target"] for item in manifest["train"])
        if not 0 < positives < len(manifest["train"]):
            parser.error("The CXR training split needs both classes.")
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor((len(manifest["train"]) - positives) / positives, device=device)
        )
    else:
        from monai.losses import DiceCELoss

        dataset, model, evaluate = CtDataset, segmenter(), eval_ct
        version, metric = SEG_VERSION, "mean_dice"
        loss_fn = DiceCELoss(sigmoid=True)
    model.to(device)
    train = DataLoader(dataset(args.data, manifest["train"], training=True),
                       batch_size=args.batch_size, shuffle=True)
    eval_batch = args.batch_size if args.task == "cxr" else 1
    validation, test = (
        DataLoader(dataset(args.data, manifest[split]), batch_size=eval_batch)
        for split in ("validation", "test")
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best, history = -float("inf"), []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for pixels, labels in train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixels.to(device))
            loss = loss_fn(logits, labels.to(device).reshape_as(logits))
            loss.backward()
            optimizer.step()
            running += loss.item()
        model.eval()
        metrics = evaluate(model, validation, device)
        value = metrics[metric]
        if value is None or not np.isfinite(value):
            raise ValueError(f"Validation metric {metric} is unavailable.")
        history.append({"epoch": epoch + 1, "loss": running / len(train), "validation": metrics})
        print(f"Epoch {epoch + 1}: loss={running / len(train):.4f} validation {metric}={value:.4f}", flush=True)
        if value > best:
            best = value
            temporary = args.checkpoint.with_suffix(".tmp")
            torch.save({
                "architecture": version, "state_dict": model.state_dict(), "epoch": epoch + 1,
                "imagenet_initialization": args.task == "cxr" and not args.from_scratch,
            }, temporary)
            temporary.replace(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"architecture": version, "selected_epoch": checkpoint["epoch"], "seed": args.seed,
              "history": history, "test": evaluate(model, test, device)}
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
