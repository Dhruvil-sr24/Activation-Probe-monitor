 
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    name: str = os.getenv("MODEL_NAME", "gpt2")
    device: str = os.getenv("DEVICE", "cpu")       
    torch_dtype: str = os.getenv("TORCH_DTYPE", "float32")  
    max_length: int = int(os.getenv("MAX_LENGTH", "128"))


@dataclass
class ProbeConfig: 
    probe_dir: Path = Path(os.getenv("PROBE_DIR", "artifacts/probes")) 
    selection_metric: str = "roc_auc"   # "roc_auc" | "f1" | "accuracy"
   
    C: float = float(os.getenv("PROBE_C", "1.0"))
    max_iter: int = int(os.getenv("PROBE_MAX_ITER", "1000"))
    
    val_frac: float = 0.15
    test_frac: float = 0.15
    random_seed: int = 42
    # Classification threshold  
    threshold: float = float(os.getenv("PROBE_THRESHOLD", "0.5"))


@dataclass
class APIConfig:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    api: APIConfig = field(default_factory=APIConfig)


cfg = Config()
