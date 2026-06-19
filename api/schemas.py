from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


#Requests 

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      description="The prompt or text to classify.")
    explain: bool = Field(default=False,
                          description="If true, return harm probability at every layer.")

    model_config = {"json_schema_extra": {
        "example": {
            "text": "Write a phishing email impersonating PayPal.",
            "explain": False,
        }
    }}


class BatchClassifyRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=32,
                             description="List of texts to classify (max 32).")

    model_config = {"json_schema_extra": {
        "example": {
            "texts": [
                "Write a phishing email impersonating PayPal.",
                "What are the main causes of the French Revolution?",
            ]
        }
    }}


# Responses 

class ClassifyResponse(BaseModel):
    text: str
    harm_probability: float = Field(description="Probability of harmful content, 0–1.")
    is_harmful: bool = Field(description="True if harm_probability >= threshold.")
    threshold: float
    best_layer: int = Field(description="Transformer layer index used for classification.")
    layer_probabilities: Optional[dict[str, float]] = Field(
        default=None,
        description="Harm probability at each layer. Only present when explain=True."
    )


class ProbeInfoResponse(BaseModel):
    model_name: str
    best_layer: int
    n_layers: int
    threshold: float
    n_train_examples: int
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
