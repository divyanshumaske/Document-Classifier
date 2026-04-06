"""
Document Image Classifier — EfficientNet-B4
============================================
Stage 1 baseline. Swap backbone string to upgrade to ViT/DiT later.

Usage:
    python train.py
"""

import os
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
import timm
from datasets import load_dataset
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — only edit this block
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    # Paths
    cache_dir   = "C:/Users/divya/OneDrive/Documents/Document classifier/data",
    ckpt_dir    = "C:/Users/divya/OneDrive/Documents/Document classifier/checkpoints",
    output_dir  = "C:/Users/divya/OneDrive/Documents/Document classifier/outputs",

    # Model — change this one string to swap backbone later
    backbone    = "efficientnet_b4",   # next: "swin_base_patch4_window7_224"
    img_size    = 224,
    num_classes = 16,
    dropout     = 0.40,

    # Training
    epochs      = 30,
    batch_size  = 16,       # safe for 8 GB VRAM with EfficientNet-B4
    num_workers = 4,

    # Optimiser
    lr_head     = 1e-3,     # phase 1: head only
    lr_full     = 1e-4,     # phase 2: full fine-tune
    weight_decay= 1e-4,
    label_smoothing = 0.10,

    # Scheduler
    phase1_epochs = 8,      # freeze backbone, train head only
    phase2_epochs = 22,     # unfreeze all, lower LR

    # Regularisation
    mixup_alpha = 0.4,

    # Early stopping
    patience    = 6,

    # Inference
    conf_threshold = 0.70,
)

CLASSES = [
    'letter', 'form', 'email', 'handwritten', 'advertisement',
    'scientific report', 'scientific publication', 'specification',
    'file folder', 'news article', 'budget', 'invoice',
    'presentation', 'questionnaire', 'resume', 'memo'
]

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

def build_transforms(split, img_size):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32),
                     interpolation=InterpolationMode.BICUBIC),
            T.RandomCrop(img_size),
            T.RandomRotation(degrees=5,
                             interpolation=InterpolationMode.BILINEAR),
            T.RandomApply([T.GaussianBlur(kernel_size=3,
                                          sigma=(0.1, 1.0))], p=0.25),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            T.RandomGrayscale(p=0.10),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size),
                     interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


class RVLCDIPDataset(Dataset):
    def __init__(self, hf_dataset, split, img_size):
        self.data      = hf_dataset
        self.transform = build_transforms(split, img_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img    = sample["image"].convert("RGB")
        label  = sample["label"]
        return self.transform(img), label

    def get_sample_weights(self):
        labels  = [self.data[i]["label"] for i in range(len(self.data))]
        counts  = np.bincount(labels, minlength=16).astype(float)
        counts  = np.where(counts == 0, 1, counts)
        weights = 1.0 / counts
        return torch.tensor([weights[l] for l in labels], dtype=torch.float)


def build_loaders(cfg, hf):
    train_ds = RVLCDIPDataset(hf["train"],      "train", cfg["img_size"])
    val_ds   = RVLCDIPDataset(hf["validation"], "val",   cfg["img_size"])
    test_ds  = RVLCDIPDataset(hf["test"],       "test",  cfg["img_size"])

    sampler  = WeightedRandomSampler(
        weights     = train_ds.get_sample_weights(),
        num_samples = len(train_ds),
        replacement = True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["batch_size"],
        sampler     = sampler,
        num_workers = cfg["num_workers"],
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = cfg["num_workers"],
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = cfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = cfg["num_workers"],
        pin_memory  = True,
    )
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class DocumentClassifier(nn.Module):
    def __init__(self, backbone, num_classes, dropout):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained   = True,
            num_classes  = 0,
            global_pool  = "avg",
        )
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("Backbone frozen — training head only.")

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("Backbone unfrozen — full fine-tune.")

    def forward(self, x):
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# MIXUP
# ─────────────────────────────────────────────────────────────────────────────

def mixup(x, y, alpha, device):
    if alpha <= 0:
        return x, y, y, 1.0
    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(x.size(0)).to(device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_loss(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device,
                use_mixup=False, alpha=0.4, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            if use_mixup:
                imgs_m, ya, yb, lam = mixup(imgs, labels, alpha, device)
                logits = model(imgs_m)
                loss   = mixup_loss(criterion, logits, ya, yb, lam)
            else:
                logits = model(imgs)
                loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  eval ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)

        preds       = logits.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"],   label="val")
    ax1.set_title("Loss"); ax1.legend(); ax1.set_xlabel("Epoch")

    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["val_acc"],   label="val")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150)
    plt.close()
    print(f"Training curves saved.")


def plot_confusion(labels, preds, classes, output_dir):
    cm  = confusion_matrix(labels, preds)
    fig = plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
    )
    plt.title("Confusion matrix — test set")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} — {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(CFG["ckpt_dir"],   exist_ok=True)
    os.makedirs(CFG["output_dir"], exist_ok=True)

    # Data
    print("\nLoading dataset...")
    hf = load_dataset("aharley/rvl_cdip", cache_dir=CFG["cache_dir"])
    train_loader, val_loader, test_loader = build_loaders(CFG, hf)
    print(f"Train: {len(train_loader.dataset):,}  "
          f"Val: {len(val_loader.dataset):,}  "
          f"Test: {len(test_loader.dataset):,}")

    # Model
    print(f"\nBuilding {CFG['backbone']}...")
    model     = DocumentClassifier(
        CFG["backbone"], CFG["num_classes"], CFG["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=CFG["label_smoothing"]
    )
    scaler    = torch.cuda.amp.GradScaler()

    history   = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
    best_acc  = 0.0
    patience_count = 0
    best_ckpt = f"{CFG['ckpt_dir']}/best_model.pt"

    # ── Phase 1: train head only ──────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"Phase 1 — head warm-up ({CFG['phase1_epochs']} epochs, backbone frozen)")
    print(f"{'─'*55}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr_head"], weight_decay=CFG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["phase1_epochs"], eta_min=1e-6
    )

    for epoch in range(1, CFG["phase1_epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            device, use_mixup=False, scaler=scaler
        )
        vl_loss, vl_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"Ep {epoch:02d}/{CFG['phase1_epochs']}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  "
              f"({time.time()-t0:.0f}s)")

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_acc": best_acc, "cfg": CFG,
            }, best_ckpt)
            print(f"  --> Best model saved (val_acc={best_acc:.4f})")

    # ── Phase 2: full fine-tune ───────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"Phase 2 — full fine-tune ({CFG['phase2_epochs']} epochs, all layers)")
    print(f"{'─'*55}")

    model.unfreeze_all()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["lr_full"], weight_decay=CFG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["phase2_epochs"], eta_min=1e-7
    )

    for epoch in range(1, CFG["phase2_epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            device, use_mixup=True,
            alpha=CFG["mixup_alpha"], scaler=scaler
        )
        vl_loss, vl_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"Ep {epoch:02d}/{CFG['phase2_epochs']}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  "
              f"({time.time()-t0:.0f}s)")

        if vl_acc > best_acc:
            best_acc = vl_acc
            patience_count = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_acc": best_acc, "cfg": CFG,
            }, best_ckpt)
            print(f"  --> Best model saved (val_acc={best_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= CFG["patience"]:
                print(f"\nEarly stopping — no improvement for "
                      f"{CFG['patience']} epochs.")
                break

    # ── Final evaluation on test set ─────────────────────────────────────
    print(f"\n{'─'*55}")
    print("Final evaluation on test set")
    print(f"{'─'*55}")

    ckpt = torch.load(best_ckpt)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded best checkpoint (val_acc={ckpt['val_acc']:.4f} "
          f"from epoch {ckpt['epoch']})")

    _, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\nTest accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\nPer-class report:")
    print(classification_report(test_labels, test_preds,
                                 target_names=CLASSES, digits=3))

    # Save plots
    plot_history(history, CFG["output_dir"])
    plot_confusion(test_labels, test_preds, CLASSES, CFG["output_dir"])

    print(f"\nDone. Best val_acc = {best_acc:.4f}")
    print(f"Outputs saved to: {CFG['output_dir']}")


if __name__ == "__main__":
    main()