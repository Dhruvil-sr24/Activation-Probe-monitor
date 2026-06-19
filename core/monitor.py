 
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.hooks import ActivationCollector
from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class HarmScore: 
    text: str
    harm_probability: float
    is_harmful: bool
    threshold: float
    best_layer: int 
    layer_probabilities: dict[int, float] | None = None

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "harm_probability": round(self.harm_probability, 6),
            "is_harmful": self.is_harmful,
            "threshold": self.threshold,
            "best_layer": self.best_layer,
            "layer_probabilities": (
                {str(k): round(v, 6) for k, v in self.layer_probabilities.items()}
                if self.layer_probabilities else None
            ),
        }


class SafetyMonitor: 

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        probe_dir: Path | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        probe_dir = probe_dir or cfg.probe.probe_dir
        self._load_artifacts(probe_dir)
        self.collector = ActivationCollector(model)

    def _load_artifacts(self, probe_dir: Path) -> None: 
        meta_path = probe_dir / "probe_metadata.json"
        probe_path = probe_dir / "probe_best.joblib"
        scaler_path = probe_dir / "scaler.joblib"

        for p in (meta_path, probe_path, scaler_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Artifact not found: {p}\n"
                    f"Run 'python scripts/train.py' first to train the probe."
                )

        with open(meta_path) as f:
            self.metadata: dict = json.load(f)

        self.probe: LogisticRegression = joblib.load(probe_path)
        self.scaler: StandardScaler = joblib.load(scaler_path)
        self.best_layer: int = self.metadata["best_layer"]
        self.threshold: float = self.metadata["threshold"]
        self.n_layers: int = self.metadata["n_layers"]

        logger.info(
            f"SafetyMonitor loaded | model={self.metadata['model_name']} | "
            f"best_layer={self.best_layer} | "
            f"val_AUC={self.metadata['val_metrics']['roc_auc']:.4f} | "
            f"test_AUC={self.metadata['test_metrics']['roc_auc']:.4f}"
        )

    # Single classification 

    def classify(self, text: str, explain: bool = False) -> HarmScore: 
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.model.max_length,
        ).to(self.model.device)

        self.model.eval()
        # acts = self.collector.collect_batch(self.model, enc.input_ids)
        acts = self.collector.collect_batch(self.model, enc.input_ids)
        print(acts.keys())  
        hidden = acts[self.best_layer].numpy().reshape(1, -1)
        scaled = self.scaler.transform(hidden)
        harm_prob = float(self.probe.predict_proba(scaled)[0, 1])
        is_harmful = harm_prob >= self.threshold 
        layer_probs: dict[int, float] | None = None
        if explain:
            layer_probs = {}
            for layer_idx, vec in acts.items():
                h = vec.numpy().reshape(1, -1) 
                try:
                    s = self.scaler.transform(h)
                    p = float(self.probe.predict_proba(s)[0, 1])
                except Exception:
                    p = 0.0
                layer_probs[layer_idx] = p

        return HarmScore(
            text=text,
            harm_probability=harm_prob,
            is_harmful=is_harmful,
            threshold=self.threshold,
            best_layer=self.best_layer,
            layer_probabilities=layer_probs,
        ) 

    def classify_batch(self, texts: list[str]) -> list[HarmScore]: 
        return [self.classify(text) for text in texts]
 

    def get_info(self) -> dict: 
        return {
            "model_name": self.metadata["model_name"],
            "best_layer": self.best_layer,
            "n_layers": self.n_layers,
            "threshold": self.threshold,
            "n_train_examples": self.metadata["n_train"],
            "val_metrics": self.metadata["val_metrics"],
            "test_metrics": self.metadata["test_metrics"],
        }
