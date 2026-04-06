import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import timm
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, top_k_accuracy_score,
    precision_recall_fscore_support, cohen_kappa_score
)
from PIL import Image, UnidentifiedImageError
from io import BytesIO

sys.stdout.reconfigure(encoding="utf-8")

# config
CACHE_DIR  = "C:/Users/divya/OneDrive/Documents/Document classifier/data"
CKPT_PATH  = "C:/Users/divya/OneDrive/Documents/Document classifier/checkpoints/best_model.pt"
OUTPUT_DIR = "C:/Users/divya/OneDrive/Documents/Document classifier/outputs/evaluation"

CLASSES = [
    'letter', 'form', 'email', 'handwritten', 'advertisement',
    'scientific report', 'scientific publication', 'specification',
    'file folder', 'news article', 'budget', 'invoice',
    'presentation', 'questionnaire', 'resume', 'memo'
]
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_WORKERS = 4


# model definition
class DocumentClassifier(nn.Module):
    def __init__(self, backbone, num_classes, dropout):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=False, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# dataset with corrupted image handling
class RVLCDIPDataset(Dataset):
    def __init__(self, hf_dataset, img_size):
        self.data = hf_dataset
        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.blank = Image.new("RGB", (img_size, img_size), color=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            s   = self.data[idx]
            img = s["image"].convert("RGB")
            lbl = s["label"]
        except (UnidentifiedImageError, Exception):
            img = self.blank
            lbl = -1
        return self.transform(img), lbl


# inference loop
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in tqdm(loader, desc="  running inference"):
        imgs = imgs.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    probs  = np.array(all_probs)

    # drop corrupted samples (label == -1)
    valid  = labels != -1
    skipped = (~valid).sum()
    if skipped > 0:
        print(f"  Skipped {skipped} corrupted image(s).")
    return labels[valid], preds[valid], probs[valid]


# overall and per-class metrics
def print_all_metrics(labels, preds, probs):
    sep = "=" * 68

    print(f"\n{sep}")
    print("  OVERALL METRICS")
    print(sep)

    acc   = accuracy_score(labels, preds)
    top2  = top_k_accuracy_score(labels, probs, k=2)
    top3  = top_k_accuracy_score(labels, probs, k=3)
    kappa = cohen_kappa_score(labels, preds)
    pm, rm, fm, _ = precision_recall_fscore_support(labels, preds, average="macro",    zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)

    print(f"  Top-1 accuracy        : {acc*100:.2f}%")
    print(f"  Top-2 accuracy        : {top2*100:.2f}%")
    print(f"  Top-3 accuracy        : {top3*100:.2f}%")
    print(f"  Cohen kappa           : {kappa:.4f}")
    print(f"  Macro  precision      : {pm:.4f}")
    print(f"  Macro  recall         : {rm:.4f}")
    print(f"  Macro  F1             : {fm:.4f}")
    print(f"  Weighted precision    : {pw:.4f}")
    print(f"  Weighted recall       : {rw:.4f}")
    print(f"  Weighted F1           : {fw:.4f}")

    # per-class report
    print(f"\n{sep}")
    print("  PER-CLASS CLASSIFICATION REPORT")
    print(sep)
    print(classification_report(labels, preds, target_names=CLASSES, digits=4, zero_division=0))

    # per-class confidence
    print(f"{sep}")
    print("  PER-CLASS CONFIDENCE")
    print(sep)
    for i, cls in enumerate(CLASSES):
        mask      = labels == i
        cls_probs = probs[mask, i]
        n_correct = (preds[mask] == i).sum()
        n_total   = mask.sum()
        print(f"  {cls:<26}  mean={cls_probs.mean():.3f}  "
              f"min={cls_probs.min():.3f}  max={cls_probs.max():.3f}  "
              f"correct={n_correct}/{n_total}")

    # confidence threshold stats
    print(f"\n{sep}")
    print("  CONFIDENCE GATING  (threshold = 0.70)")
    print(sep)
    max_probs = probs.max(axis=1)
    below     = (max_probs < 0.70).sum()
    above     = (max_probs >= 0.70).sum()
    acc_above = accuracy_score(labels[max_probs >= 0.70], preds[max_probs >= 0.70])
    print(f"  Samples below threshold : {below:,}  ({below/len(labels)*100:.1f}%)")
    print(f"  Samples above threshold : {above:,}  ({above/len(labels)*100:.1f}%)")
    print(f"  Accuracy on kept samples: {acc_above*100:.2f}%")
    print(sep)


# 16x16 raw count confusion matrix
def plot_cm_16x16_counts(labels, preds, output_dir):
    cm = confusion_matrix(labels, preds)
    short = ['letter','form','email','handwrit.','advert.','sci.rep.',
             'sci.pub.','spec.','folder','news','budget','invoice',
             'present.','questn.','resume','memo']
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=short, yticklabels=short,
                ax=ax, linewidths=0.3, annot_kws={"size": 7})
    ax.set_title("Confusion matrix - raw counts (16x16)", fontsize=13, pad=14)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    plt.tight_layout()
    path = f"{output_dir}/confusion_matrix_16x16_counts.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# 16x16 normalised confusion matrix
def plot_cm_16x16_norm(labels, preds, output_dir):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    short = ['letter','form','email','handwrit.','advert.','sci.rep.',
             'sci.pub.','spec.','folder','news','budget','invoice',
             'present.','questn.','resume','memo']
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=short, yticklabels=short,
                ax=ax, linewidths=0.4, vmin=0, vmax=100,
                annot_kws={"size": 7})
    ax.set_title("Confusion matrix - normalised % of true class (16x16)", fontsize=13, pad=14)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    plt.tight_layout()
    path = f"{output_dir}/confusion_matrix_16x16_normalised.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# 2x2 binary confusion matrix per class
def plot_cm_2x2_per_class(labels, preds, output_dir):
    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    fig.suptitle(
        "Per-class binary confusion matrices (2x2)\nPositive = that class, Negative = all others",
        fontsize=14, y=1.01
    )
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#2ecc71", "#e74c3c"])

    for idx, cls in enumerate(CLASSES):
        ax = axes[idx // 4][idx % 4]
        y_true_bin = (labels == idx).astype(int)
        y_pred_bin = (preds  == idx).astype(int)
        cm2        = confusion_matrix(y_true_bin, y_pred_bin)
        tn, fp, fn, tp = cm2.ravel()

        annot      = np.array([[f"TN\n{tn:,}", f"FP\n{fp:,}"],
                                [f"FN\n{fn:,}", f"TP\n{tp:,}"]])
        colour_vals = np.array([[0.0, 1.0], [1.0, 0.0]])

        sns.heatmap(colour_vals, annot=annot, fmt="", cmap=cmap,
                    vmin=0, vmax=1,
                    xticklabels=["Pred: NEG", "Pred: POS"],
                    yticklabels=["True: NEG", "True: POS"],
                    ax=ax, linewidths=1, linecolor="white",
                    cbar=False, annot_kws={"size": 11, "weight": "bold"})

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        ax.set_title(f"{cls}\nP={prec:.2f}  R={rec:.2f}  F1={f1:.2f}", fontsize=9, pad=4)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = f"{output_dir}/confusion_matrix_2x2_per_class.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# per-class f1 bar chart
def plot_per_class_f1(labels, preds, output_dir):
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    order  = np.argsort(f1)
    s_cls  = [CLASSES[i] for i in order]
    s_f1   = f1[order]
    colors = ["#e74c3c" if v < 0.80 else "#f39c12" if v < 0.90 else "#2ecc71" for v in s_f1]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(s_cls, s_f1, color=colors, edgecolor="white", height=0.65)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 score", fontsize=11)
    ax.set_title("Per-class F1 score", fontsize=12)
    ax.axvline(np.mean(f1), color="steelblue", linewidth=1.5,
               linestyle="--", label=f"Mean F1 = {np.mean(f1):.3f}")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, s_f1):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    path = f"{output_dir}/per_class_f1.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# confidence distribution correct vs incorrect
def plot_confidence_distribution(labels, preds, probs, output_dir):
    max_probs = probs.max(axis=1)
    correct   = labels == preds

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(max_probs[correct],  bins=50, alpha=0.65, color="steelblue",
            label="Correct predictions",  density=True)
    ax.hist(max_probs[~correct], bins=50, alpha=0.65, color="tomato",
            label="Incorrect predictions", density=True)
    ax.axvline(0.70, color="black", linewidth=1.5, linestyle="--", label="Threshold (0.70)")
    ax.set_xlabel("Max softmax probability", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Confidence distribution - correct vs incorrect", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = f"{output_dir}/confidence_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# top misclassification pairs
def plot_top_confusions(labels, preds, output_dir, top_n=15):
    cm = confusion_matrix(labels, preds)
    np.fill_diagonal(cm, 0)
    pairs, counts = [], []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j] > 0:
                pairs.append(f"{CLASSES[i]}  ->  {CLASSES[j]}")
                counts.append(cm[i, j])
    order      = np.argsort(counts)[-top_n:]
    top_pairs  = [pairs[i]  for i in order]
    top_counts = [counts[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_pairs, top_counts, color="steelblue", edgecolor="white", height=0.65)
    ax.set_xlabel("Number of misclassifications", fontsize=11)
    ax.set_title(f"Top {top_n} misclassification pairs", fontsize=12)
    for i, v in enumerate(top_counts):
        ax.text(v + 2, i, str(v), va="center", fontsize=8)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    path = f"{output_dir}/top_confusions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load checkpoint
    print(f"\nLoading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    cfg  = ckpt["cfg"]
    print(f"  Backbone   : {cfg['backbone']}")
    print(f"  Best epoch : {ckpt['epoch']}")
    print(f"  Val acc    : {ckpt['val_acc']*100:.2f}%")

    # build model and load weights
    model = DocumentClassifier(cfg["backbone"], cfg["num_classes"], cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print("  Model loaded.")

    # load test set
    print("\nLoading test set...")
    hf = load_dataset("aharley/rvl_cdip", cache_dir=CACHE_DIR)
    test_ds = RVLCDIPDataset(hf["test"], IMG_SIZE)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    print(f"  Test samples : {len(test_ds):,}")

    # run inference
    print()
    labels, preds, probs = run_inference(model, test_loader, device)

    # print all metrics
    print_all_metrics(labels, preds, probs)

    # save all plots
    print("\n" + "=" * 68)
    print("  SAVING PLOTS")
    print("=" * 68)
    plot_cm_16x16_counts(labels, preds, OUTPUT_DIR)
    plot_cm_16x16_norm(labels, preds, OUTPUT_DIR)
    plot_cm_2x2_per_class(labels, preds, OUTPUT_DIR)
    plot_per_class_f1(labels, preds, OUTPUT_DIR)
    plot_confidence_distribution(labels, preds, probs, OUTPUT_DIR)
    plot_top_confusions(labels, preds, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()