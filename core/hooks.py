 
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any


def _detect_layers(model: nn.Module) -> list[nn.Module]:
    """
      - GPT-2 style:           model.transformer.h
      - LLaMA / Mistral style: model.model.layers
      - Falcon style:          model.transformer.h (same as GPT-2)
      - BERT / RoBERTa style:  model.encoder.layer  (for encoder-only)
      - Bloom style:           model.transformer.h
    """ 
    candidates = [
        ("transformer.h",    lambda m: m.transformer.h),
        ("model.layers",     lambda m: m.model.layers),
        ("encoder.layer",    lambda m: m.encoder.layer),
        ("bert.encoder.layer", lambda m: m.bert.encoder.layer),
    ]
    for name, accessor in candidates:
        try:
            layers = list(accessor(model))
            # print(layers[:5])
            if layers:
                return layers
        except AttributeError:
            continue
 
    for _, module in model.named_modules():
        children = list(module.children())
        if len(children) >= 2:
            first = children[0]
            if any(hasattr(first, attr) for attr in ("attn", "attention", "self_attention", "self_attn")):
                return children

    raise ValueError(
        "Could not auto-detect transformer layer structure. "
        "Set model architecture explicitly or extend _detect_layers()."
    )


class ActivationCollector: 

    def __init__(self, model: nn.Module):
        self.model = model
        self.layers = _detect_layers(model)
        self.n_layers = len(self.layers) 
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []

    def _make_hook(self, layer_idx: int): 
        def hook(module, input, output):  
            hidden = output[0] if isinstance(output, tuple) else output 
            last_token = hidden[:, -1, :].detach().cpu() 
            self.activations[layer_idx] = last_token.squeeze(0)
        return hook

    def __enter__(self):
        self.activations = {}
        self._handles = []
        for i, layer in enumerate(self.layers):
            handle = layer.register_forward_hook(self._make_hook(i))
            self._handles.append(handle)
        # print(f"Found {len(self.layers)} layers")
        # for i, layer in enumerate(self.layers):
        #     print(i, type(layer))
            
        return self

    def __exit__(self, *args):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # def collect_batch(
    #     self,
    #     model: nn.Module,
    #     input_ids: torch.Tensor,
    # ) -> dict[int, torch.Tensor]: 
    #     self.activations = {}
    #     with torch.no_grad():
    #         model(input_ids=input_ids)
    #     return dict(self.activations)
    def collect_batch(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ):
        with self:

            self.activations = {}

            with torch.no_grad():
                model(input_ids=input_ids)

            return dict(self.activations)
