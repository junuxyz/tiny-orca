from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


@dataclass(slots=True)
class RequestSpan:
    """One request slice inside the flattened [sum_S, Hidden] tensor."""

    request_id: str
    start: int
    end: int


class Qwen3SelectiveModel(nn.Module):
    """Qwen3 forward path that owns execution block-by-block."""

    def __init__(self, model: Qwen3ForCausalLM):
        super().__init__()
        self.model = model
        self.layers = model.model.layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        spans: list[RequestSpan],
        position_ids: list[torch.Tensor],
        cache_position: list[torch.Tensor],
        request_caches: dict[str, DynamicCache],
    ) -> torch.Tensor:
        split_hidden_states = self.split_hidden_states(hidden_states, spans)
        position_embeddings: list[tuple[torch.Tensor, torch.Tensor]] = []
        attention_masks: list[torch.Tensor | None] = []

        for req_hidden, span, req_position_ids, req_cache_position in zip(
            split_hidden_states, spans, position_ids, cache_position, strict=True
        ):
            req_id = span.request_id
            # apply RoPE
            req_position_ids = req_position_ids.unsqueeze(0)
            position_embeddings.append(self.model.model.rotary_emb(req_hidden, req_position_ids))
            attention_masks.append(
                create_causal_mask(
                    config=self.model.config,
                    inputs_embeds=req_hidden,
                    attention_mask=None,
                    cache_position=req_cache_position,
                    past_key_values=request_caches[req_id],
                    position_ids=req_position_ids,
                )
            )

        # for each Transformer Layer
        for layer in self.layers:
            residual = hidden_states
            # RMSNorm 1
            hidden_states = layer.input_layernorm(hidden_states)

            # Split
            split_hidden_states = self.split_hidden_states(hidden_states, spans)
            request_outputs: list[torch.Tensor] = []

            # GQA (Grouped-Query Attention)
            # Split the flat tensor back into per-request slices so each
            # request can attend with its own KV cache state.
            for req_hidden, span, req_position_embeddings, req_cache_position, attention_mask in zip(
                split_hidden_states,
                spans,
                position_embeddings,
                cache_position,
                attention_masks,
                strict=True,
            ):
                req_id = span.request_id
                attn_out, _attn_weights = layer.self_attn(
                    hidden_states=req_hidden,
                    position_embeddings=req_position_embeddings,
                    cache_position=req_cache_position,
                    attention_mask=attention_mask,
                    past_key_values=request_caches[req_id],
                )
                request_outputs.append(attn_out)

            # Merge per-request attention outputs back into the flat tensor.
            attn_output = self.merge_request_outputs(hidden_states, spans, request_outputs)

            # residual connection
            hidden_states = residual + attn_output
            residual = hidden_states
            # RMSNorm 2
            hidden_states = layer.post_attention_layernorm(hidden_states)
            # MLP
            hidden_states = layer.mlp(hidden_states)
            # residual connection
            hidden_states = residual + hidden_states
        # final RMSNorm
        return self.model.model.norm(hidden_states)

    # Split
    @staticmethod
    def split_hidden_states(hidden_states: torch.Tensor, spans: list[RequestSpan]) -> list[torch.Tensor]:
        return [hidden_states[span.start : span.end].unsqueeze(0) for span in spans]

    # Merge
    @staticmethod
    def merge_request_outputs(
        hidden_states: torch.Tensor, spans: list[RequestSpan], request_outputs: list[torch.Tensor]
    ) -> torch.Tensor:
        merged_output = torch.empty_like(hidden_states)
        for span, request_output in zip(spans, request_outputs, strict=True):
            merged_output[span.start : span.end] = request_output.squeeze(0)
        return merged_output
