
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.hooks import ActivationCollector, _detect_layers
from data.dataset import LabeledExample, get_texts_and_labels
from config import cfg

logger = logging.getLogger(__name__)
 

def collect_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[LabeledExample],
) -> tuple[dict[int, np.ndarray], np.ndarray]:
 
    collector = ActivationCollector(model)
    
    n_layers = collector.n_layers

    texts, label_list = get_texts_and_labels(examples)
    labels = np.array(label_list)
 
    layer_activations: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}

    logger.info(f"Collecting activations for {len(texts)} examples "
                f"across {n_layers} layers...")
    t0 = time.time()

    model.eval()
    for idx, text in enumerate(texts):
        if idx % 10 == 0:
            logger.info(f"  [{idx}/{len(texts)}]")
 
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.model.max_length,
        ).to(model.device)

        with collector:
            acts = collector.collect_batch(model, enc.input_ids)

        for layer_idx, vec in acts.items():
            layer_activations[layer_idx].append(vec.numpy())

    elapsed = time.time() - t0
    logger.info(f"Collection complete in {elapsed:.1f}s")
    for i, vecs in layer_activations.items():
        print(i, len(vecs)) 
    stacked = {i: np.vstack(vecs) for i, vecs in layer_activations.items()}
    return stacked, labels


# Per-layer probe training 

def _train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler: StandardScaler,
) -> tuple[LogisticRegression, dict[str, float]]: 
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    probe = LogisticRegression(
        C=cfg.probe.C,
        max_iter=cfg.probe.max_iter,
        solver="lbfgs",
        class_weight="balanced",   
        random_state=cfg.probe.random_seed,
    )
    probe.fit(X_train_s, y_train)

    val_probs = probe.predict_proba(X_val_s)[:, 1]
    val_preds = (val_probs >= cfg.probe.threshold).astype(int)

    metrics = {
        "roc_auc":  float(roc_auc_score(y_val, val_probs)),
        "f1":       float(f1_score(y_val, val_preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_val, val_preds)),
    }
    return probe, metrics


#  Main training entry point 

def train_probes(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[LabeledExample],
) -> dict:  
    layer_activations, labels = collect_activations(model, tokenizer, examples)
 
    indices = np.arange(len(labels))
 
    idx_trainval, idx_test = train_test_split(
        indices,
        test_size=cfg.probe.test_frac,
        stratify=labels,
        random_state=cfg.probe.random_seed,
    )  
    val_frac_adjusted = cfg.probe.val_frac / (1 - cfg.probe.test_frac)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_frac_adjusted,
        stratify=labels[idx_trainval],
        random_state=cfg.probe.random_seed,
    )

    y_train = labels[idx_train]
    y_val = labels[idx_val]
    y_test = labels[idx_test]

    logger.info(
        f"Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}"
    )
 
    layer_results: list[dict] = []
    probes: dict[int, LogisticRegression] = {}
    scalers: dict[int, StandardScaler] = {}

    n_layers = len(layer_activations)
    logger.info(f"Training probes on {n_layers} layers...")

    for layer_idx in range(n_layers):
        X = layer_activations[layer_idx]
        X_train, X_val = X[idx_train], X[idx_val]

        scaler = StandardScaler().fit(X_train)
        probe, val_metrics = _train_probe(X_train, y_train, X_val, y_val, scaler)

        probes[layer_idx] = probe
        scalers[layer_idx] = scaler
        layer_results.append({"layer": layer_idx, **val_metrics})

        logger.info(
            f"  Layer {layer_idx:2d} | "
            f"AUC={val_metrics['roc_auc']:.4f} | "
            f"F1={val_metrics['f1']:.4f} | "
            f"Acc={val_metrics['accuracy']:.4f}"
        )
 
    best = max(layer_results, key=lambda r: r[cfg.probe.selection_metric])
    best_layer = best["layer"]
    logger.info(
        f"\nBest layer: {best_layer} | "
        f"Val AUC: {best['roc_auc']:.4f}"
    )
 
    best_probe = probes[best_layer]
    best_scaler = scalers[best_layer]

    X_test = layer_activations[best_layer][idx_test]
    X_test_s = best_scaler.transform(X_test)
    test_probs = best_probe.predict_proba(X_test_s)[:, 1]
    test_preds = (test_probs >= cfg.probe.threshold).astype(int)

    test_metrics = {
        "roc_auc":  float(roc_auc_score(y_test, test_probs)),
        "f1":       float(f1_score(y_test, test_preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, test_preds)),
    }
    logger.info(
        f"Test set metrics: AUC={test_metrics['roc_auc']:.4f} | "
        f"F1={test_metrics['f1']:.4f} | Acc={test_metrics['accuracy']:.4f}"
    )
 
    probe_dir = cfg.probe.probe_dir
    probe_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_probe, probe_dir / "probe_best.joblib")
    joblib.dump(best_scaler, probe_dir / "scaler.joblib")

    metadata = {
        "model_name": cfg.model.name,
        "n_examples": len(labels),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "n_layers": n_layers,
        "best_layer": int(best_layer),
        "selection_metric": cfg.probe.selection_metric,
        "threshold": cfg.probe.threshold,
        "val_metrics": {k: v for k, v in best.items() if k != "layer"},
        "test_metrics": test_metrics,
        "all_layer_val_metrics": layer_results,
    }
    with open(probe_dir / "probe_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Artifacts saved to {probe_dir}/")
    return metadata
