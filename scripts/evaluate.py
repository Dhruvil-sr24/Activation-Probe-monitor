 
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score,
)

from config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = {"harmful": "#E63946", "benign": "#457B9D", "best": "#2A9D8F"}


def load_artifacts(probe_dir: Path) -> tuple:
    """Load probe, scaler, and metadata."""
    with open(probe_dir / "probe_metadata.json") as f:
        meta = json.load(f)
    probe = joblib.load(probe_dir / "probe_best.joblib")
    scaler = joblib.load(probe_dir / "scaler.joblib")
    return probe, scaler, meta


def _save(fig, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {out}")
    plt.close(fig)


#  Figure generators 

def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PALETTE["harmful"], lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["harmful"])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Activation Probe", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    _save(fig, out_dir, "roc_curve.png")


def plot_precision_recall(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=PALETTE["benign"], lw=2, label=f"AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.08, color=PALETTE["benign"])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    _save(fig, out_dir, "pr_curve.png")


def plot_layer_auc(layer_metrics: list[dict], best_layer: int, out_dir: Path) -> None: 
    layers = [r["layer"] for r in layer_metrics]
    aucs = [r["roc_auc"] for r in layer_metrics]
    colors = [PALETTE["best"] if l == best_layer else "#ADB5BD" for l in layers]

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.5), 5))
    bars = ax.bar(layers, aucs, color=colors, edgecolor="white", linewidth=0.8)

    # Annotate best layer
    ax.axhline(aucs[best_layer], color=PALETTE["best"], linestyle="--",
               lw=1.2, alpha=0.7, label=f"Best layer {best_layer}")
    ax.set_xlabel("Transformer Layer Index", fontsize=12)
    ax.set_ylabel("Validation AUC", fontsize=12)
    ax.set_title("Layer-Wise AUC — Which Layer Carries the Most Signal?",
                 fontsize=13, fontweight="bold")
    ax.set_ylim([0.4, 1.02])
    ax.legend(fontsize=10)
    ax.tick_params(axis="x", labelsize=8 if len(layers) > 20 else 10)
    _save(fig, out_dir, "layer_auc.png")


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> None: 
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", color=PALETTE["harmful"],
            lw=2, markersize=7, label="Probe calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Plot (Reliability Diagram)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    _save(fig, out_dir, "calibration.png")


def plot_threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> None: 
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = [
        f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_t = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, f1_scores, color=PALETTE["benign"], lw=2)
    ax.axvline(best_t, color=PALETTE["harmful"], linestyle="--",
               lw=1.5, label=f"Best threshold = {best_t:.2f} (F1={best_f1:.4f})")
    ax.axvline(cfg.probe.threshold, color="gray", linestyle=":",
               lw=1.5, label=f"Current threshold = {cfg.probe.threshold}")
    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("F1 Score vs Classification Threshold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _save(fig, out_dir, "threshold_sweep.png")


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray,
                   threshold: float, out_dir: Path) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    classes = ["Benign", "Harmful"]
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           xlabel="Predicted label", ylabel="True label",
           title="Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=16, color="white" if cm[i, j] > thresh else "black")

    _save(fig, out_dir, "confusion_matrix.png")
 

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained probe and generate figures")
    parser.add_argument("--output-dir", type=str, default="artifacts/figures")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_dir = cfg.probe.probe_dir
    logger.info(f"Loading artifacts from {probe_dir}/")
    probe, scaler, meta = load_artifacts(probe_dir)
 
    logger.info(
        f"Probe metadata: best_layer={meta['best_layer']} | "
        f"test AUC={meta['test_metrics']['roc_auc']:.4f}"
    )  
    
    plot_layer_auc(meta["all_layer_val_metrics"], meta["best_layer"], out_dir)
    logger.info("Layer AUC chart saved. Run evaluate.py AFTER train.py to see all figures.")
    logger.info(
        "Note: ROC, PR, calibration, and threshold plots require test activations.\n"
        "      Extend this script to save and reload test-set hidden states for full evaluation."
    )

    print("\n" + "=" * 55)
    print("  EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Best layer : {meta['best_layer']}")
    print(f"  Val  AUC   : {meta['val_metrics']['roc_auc']:.4f}")
    print(f"  Val  F1    : {meta['val_metrics']['f1']:.4f}")
    print(f"  Test AUC   : {meta['test_metrics']['roc_auc']:.4f}")
    print(f"  Test F1    : {meta['test_metrics']['f1']:.4f}")
    print(f"  Test Acc   : {meta['test_metrics']['accuracy']:.4f}")
    print(f"  Figures    : {out_dir}/")
    print("=" * 55 + "\n")

    print("Layer-wise Validation AUC:")
    for r in meta["all_layer_val_metrics"]:
        marker = " ← best" if r["layer"] == meta["best_layer"] else ""
        print(f"  Layer {r['layer']:2d}: {r['roc_auc']:.4f}{marker}")


if __name__ == "__main__":
    main()
