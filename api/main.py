
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM

from api.schemas import (
    ClassifyRequest, BatchClassifyRequest,
    ClassifyResponse, ProbeInfoResponse,
)
from core.monitor import SafetyMonitor
from config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
 
_monitor: Optional[SafetyMonitor] = None


@asynccontextmanager
async def lifespan(app: FastAPI): 
    global _monitor
    logger.info(f"Loading model: {cfg.model.name} ...")
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

    _monitor = SafetyMonitor(model=model, tokenizer=tokenizer)
    logger.info("SafetyMonitor ready.")

    yield
 
    del _monitor
    del model


app = FastAPI(
    title="Activation Probe Safety Monitor",
    description=(
        "Real-time harmful-content detection using linear probes trained "
        "on transformer hidden states. Defense-side companion to PAIR red-teaming."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_monitor() -> SafetyMonitor:
    if _monitor is None:
        raise HTTPException(
            status_code=503,
            detail="Monitor not ready. Check startup logs.",
        )
    return _monitor


# Endpoints  

@app.get("/health", tags=["System"])
async def health(): 
    return {
        "status": "ok",
        "probe_loaded": _monitor is not None,
        "model": cfg.model.name,
    }


@app.post("/v1/classify", response_model=ClassifyResponse, tags=["Classification"])
async def classify(request: ClassifyRequest): 
    monitor = _get_monitor()
    t0 = time.time()
    result = monitor.classify(request.text, explain=request.explain)
    elapsed = time.time() - t0

    logger.info(
        f"classify | harm={result.harm_probability:.4f} | "
        f"is_harmful={result.is_harmful} | latency={elapsed*1000:.1f}ms"
    )

    return ClassifyResponse(**result.as_dict())


@app.post(
    "/v1/classify/explain",
    response_model=ClassifyResponse,
    tags=["Classification"],
)
async def classify_explain(request: ClassifyRequest): 
    monitor = _get_monitor() 
    result = monitor.classify(request.text, explain=True)
    return ClassifyResponse(**result.as_dict())


@app.post(
    "/v1/classify/batch",
    response_model=list[ClassifyResponse],
    tags=["Classification"],
)
async def classify_batch(request: BatchClassifyRequest): 
    monitor = _get_monitor()
    if len(request.texts) > 32:
        raise HTTPException(
            status_code=422,
            detail="Batch size must be <= 32.",
        )

    t0 = time.time()
    results = monitor.classify_batch(request.texts)
    elapsed = time.time() - t0

    logger.info(
        f"classify_batch | n={len(results)} | "
        f"harmful={sum(r.is_harmful for r in results)} | "
        f"latency={elapsed*1000:.1f}ms"
    )

    return [ClassifyResponse(**r.as_dict()) for r in results]


@app.get(
    "/v1/probe/info",
    response_model=ProbeInfoResponse,
    tags=["Probe"],
)
async def probe_info(): 
    monitor = _get_monitor()
    return ProbeInfoResponse(**monitor.get_info())
