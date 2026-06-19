 
import argparse
import logging
import sys
import time
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.trainer import train_probes
from data.dataset import load_dataset
from config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train activation probe")
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limit dataset to first N examples (for quick smoke tests)"
    )
    args = parser.parse_args()

    logger.info(f"=== Activation Probe Training ===")
    logger.info(f"Model  : {cfg.model.name}")
    logger.info(f"Device : {cfg.model.device}")
 
    logger.info("Loading model and tokenizer...")
    t0 = time.time()

    dtype = torch.float16 if cfg.model.torch_dtype == "float16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=dtype,
        device_map=cfg.model.device if cfg.model.device != "cpu" else None,
    )
    if cfg.model.device == "cpu":
        model = model.to("cpu")

    model.eval()
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")
 
    examples = load_dataset(shuffle=True, seed=cfg.probe.random_seed)
    if args.subset:
        examples = examples[: args.subset]
        logger.info(f"Using subset: {len(examples)} examples")
    else:
        logger.info(f"Full dataset: {len(examples)} examples")

    label_counts = {0: 0, 1: 0}
    for e in examples:
        label_counts[e.label] += 1
    logger.info(f"Class balance: benign={label_counts[0]}, harmful={label_counts[1]}")
 
    metadata = train_probes(model=model, tokenizer=tokenizer, examples=examples)
 
    print("\n" + "=" * 55)
    print("  PROBE TRAINING COMPLETE")
    print("=" * 55)
    print(f"  Model           : {metadata['model_name']}")
    print(f"  Best layer      : {metadata['best_layer']} / {metadata['n_layers'] - 1}")
    print(f"  Val  AUC        : {metadata['val_metrics']['roc_auc']:.4f}")
    print(f"  Val  F1         : {metadata['val_metrics']['f1']:.4f}")
    print(f"  Test AUC        : {metadata['test_metrics']['roc_auc']:.4f}")
    print(f"  Test F1         : {metadata['test_metrics']['f1']:.4f}")
    print(f"  Test Accuracy   : {metadata['test_metrics']['accuracy']:.4f}")
    print(f"  Artifacts saved : {cfg.probe.probe_dir}/")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
