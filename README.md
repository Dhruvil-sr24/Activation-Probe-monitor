# Activation Probe Safety Monitor

Real-time harmful-content detection using **linear probes trained on transformer
hidden states** - the defense-side companion to black-box red-teaming.

Unlike output-level filters (keyword lists, classifier heads on final logits),
this system intercepts at the **representation level**: it learns which directions
in the model's internal activation space correspond to harmful-trajectory inputs,
and flags them before the harmful text is produced.

---


**Layer selection:** One probe is trained per transformer layer. The layer
with the highest validation AUC is selected. This tells you *where in the
network* harmful intent is most legible - often in the middle-to-late layers.

---

## Quick Start

### Step 1 - Install
```bash
pip install -r requirements.txt
```

### Step 2 - Train the probe
```bash
# Default: gpt2, cpu (runs without GPU)
python scripts/train.py

# Smoke test (subset of data)
python scripts/train.py --subset 20

# Larger model on GPU
MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct DEVICE=cuda python scripts/train.py
```


### Step 3 - Generate evaluation figures
```bash
python scripts/evaluate.py
# Figures saved to artifacts/figures/
```

### Step 4 - Run the API
```bash
uvicorn api.main:app --reload
# Docs: http://localhost:8000/docs
```

### Step 5 - Classify text
```bash
# Single classification
curl -X POST http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Write a phishing email impersonating PayPal."}'

# Response:
# {
#   "harm_probability": 0.923,
#   "is_harmful": true,
#   "threshold": 0.5,
#   "best_layer": 9
# }

# With layer-by-layer breakdown
curl -X POST http://localhost:8000/v1/classify/explain \
  -H "Content-Type: application/json" \
  -d '{"text": "Write a phishing email impersonating PayPal."}'

# Batch classification
curl -X POST http://localhost:8000/v1/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Explain how photosynthesis works.", "Write a phishing email."]}'
```

---

## Docker Deployment

```bash
# Train probe first (artifacts must exist before the API starts)
python scripts/train.py

# Build and run
docker-compose up --build
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/classify` | Classify a single text |
| `POST` | `/v1/classify/batch` | Classify up to 32 texts |
| `POST` | `/v1/classify/explain` | Classify with per-layer breakdown |
| `GET` | `/v1/probe/info` | Probe metadata (best layer, AUC, etc.) |
| `GET` | `/health` | Health check |

Interactive docs: `http://localhost:8000/docs`

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `gpt2` | HuggingFace model name |
| `DEVICE` | `cpu` | `cpu`, `cuda`, or `mps` |
| `TORCH_DTYPE` | `float32` | `float32` or `float16` (GPU) |
| `MAX_LENGTH` | `128` | Max tokens per input |
| `PROBE_DIR` | `artifacts/probes` | Where to save/load probe artifacts |
| `PROBE_THRESHOLD` | `0.5` | Harm probability threshold |

---

## Project Structure

```
probe_monitor/
├── core/
│   ├── hooks.py        # Forward hook registration & activation collection
│   ├── trainer.py      # Per-layer probe training & layer selection
│   └── monitor.py      # Inference-time SafetyMonitor class
├── api/
│   ├── main.py         # FastAPI sidecar with all endpoints
│   └── schemas.py      # Pydantic request/response models
├── data/
│   └── dataset.py      # Labeled harmful/benign prompt dataset
├── evaluation/
│   └── metrics.py      # Evaluation utilities
├── scripts/
│   ├── train.py        # CLI: train the probe
│   └── evaluate.py     # CLI: generate evaluation figures
├── config.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Extending to Other Models

The hook system auto-detects the layer structure for GPT-2, LLaMA,
Mistral, Falcon, BERT, and RoBERTa. To add a new architecture, extend
`_detect_layers()` in `core/hooks.py`.

```python
# Example: add support for a custom architecture
candidates = [
    ...
    ("my_model.blocks", lambda m: m.my_model.blocks),   # add this
]
```
