# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Full-layer shift-1 feedback with independent first/second-pass KV caches.

The decoder and output head are shared. Cache state also retains the preceding
token's normalized first-pass output: the second pass at t must use h1[t-1],
including across prefill/decode and multi-token chunk boundaries.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3RMSNorm

from .configuration_puro_feedback import PuroFeedbackConfig


class PuroFeedbackCache(DynamicCache):
    """The inherited layers hold pass 2; pass 1 has a separate cache."""

    def __init__(self, config, feedback_passes):
        super().__init__(config=config)
        self.first_pass = DynamicCache(config=config)
        self.previous_hidden = None
        self.previous_valid = None
        self.feedback_passes = feedback_passes
        self._feedback_config = config

    def _select_feedback(self, indices):
        for name in ("previous_hidden", "previous_valid"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.index_select(0, indices.to(value.device)))

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        self.first_pass.reorder_cache(beam_idx)
        self._select_feedback(beam_idx)

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        self.first_pass.batch_select_indices(indices)
        self._select_feedback(indices)

    def batch_repeat_interleave(self, repeats):
        super().batch_repeat_interleave(repeats)
        self.first_pass.batch_repeat_interleave(repeats)
        for name in ("previous_hidden", "previous_valid"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.repeat_interleave(repeats, dim=0))

    def reset(self):
        self.__init__(self._feedback_config, self.feedback_passes)

    def crop(self, max_length):
        length = self.get_seq_length()
        target = max_length if max_length >= 0 else length + max_length
        if target >= length:
            return
        if target == 0:
            self.reset()
            return
        raise ValueError("Feedback cache rollback is unsupported; start a fresh document cache")

    def to_legacy_cache(self):
        raise ValueError("A legacy KV tuple cannot preserve both passes and the feedback state")


class ResidualGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.token_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.embedding_norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.input_norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, previous_hidden, embeddings):
        value = self.hidden_projection(self.input_norm(previous_hidden))
        gate = torch.sigmoid(self.token_projection(self.embedding_norm(embeddings)))
        return embeddings + value * gate


@dataclass
class PuroFeedbackOutput(CausalLMOutputWithPast):
    k1_loss: Optional[torch.Tensor] = None
    k2_loss: Optional[torch.Tensor] = None


class PuroFeedbackForCausalLM(Qwen3ForCausalLM):
    config_class = PuroFeedbackConfig
    # Distributed HF TP/PP and static/quantized caches are not part of this adapter.
    _tp_plan = None
    _pp_plan = None

    def __init__(self, config):
        super().__init__(config)
        self.latent_feedback = ResidualGLU(config)
        self.post_init()
        with torch.no_grad():
            self.latent_feedback.hidden_projection.weight.zero_()

    @classmethod
    def _supports_default_dynamic_cache(cls):
        # GenerationMixin must let forward initialize the dual cache.
        return False

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        if generation_config.cache_implementation is not None:
            raise ValueError("Leave cache_implementation unset to use the Puro dual dynamic cache")
        return super()._prepare_cache_for_generation(
            generation_config, model_kwargs, *args, **kwargs
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PuroFeedbackCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        feedback_passes: Optional[int] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        passes = self.config.feedback_passes if feedback_passes is None else feedback_passes
        if passes not in (1, 2):
            raise ValueError("feedback_passes must be 1 or 2")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("Use a 2D attention mask and one independent document per batch row")
        use_cache = self.config.use_cache and labels is None if use_cache is None else use_cache
        if past_key_values is not None:
            if not isinstance(past_key_values, PuroFeedbackCache):
                raise ValueError("past_key_values must be a PuroFeedbackCache")
            if past_key_values.feedback_passes != passes:
                raise ValueError("Cannot change K1/K2 while reusing a cache")
            if not use_cache:
                raise ValueError("Do not supply past_key_values when use_cache=False")
        if labels is not None and (
            not isinstance(logits_to_keep, int)
            or logits_to_keep != 0
            or past_key_values is not None
        ):
            raise ValueError("Label scoring requires a full uncached sequence and logits_to_keep=0")
        if use_cache and past_key_values is None:
            past_key_values = PuroFeedbackCache(self.config, passes)
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        embeddings = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length, _ = embeddings.shape
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + length, device=embeddings.device
            )
        if attention_mask is None:
            current_valid = torch.ones((batch, length), dtype=torch.bool, device=embeddings.device)
        else:
            if attention_mask.shape != (batch, past_length + length):
                raise ValueError("attention_mask must cover cached plus current tokens")
            current_valid = attention_mask[:, -length:].bool()
        if position_ids is None:
            if attention_mask is None:
                position_ids = cache_position.unsqueeze(0)
            else:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)[:, -length:]
        model_kwargs = dict(
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )
        first_cache = past_key_values.first_pass if passes == 2 and use_cache else past_key_values
        first = self.model(inputs_embeds=embeddings, past_key_values=first_cache, **model_kwargs)
        output = first
        if passes == 2:
            preceding = torch.zeros_like(first.last_hidden_state[:, :1])
            preceding_valid = torch.zeros_like(current_valid[:, :1])
            if past_length:
                if (
                    past_key_values.previous_hidden is None
                    or first_cache.get_seq_length() != past_length + length
                ):
                    raise ValueError("Incomplete first-pass feedback cache")
                preceding = past_key_values.previous_hidden
                preceding_valid = past_key_values.previous_valid
            shifted = torch.cat((preceding, first.last_hidden_state[:, :-1]), dim=1)
            valid = current_valid & torch.cat((preceding_valid, current_valid[:, :-1]), dim=1)
            valid = valid & (position_ids != 0)
            fused = self.latent_feedback(shifted, embeddings)
            fused = torch.where(valid.unsqueeze(-1), fused, embeddings)
            output = self.model(
                inputs_embeds=fused, past_key_values=past_key_values, **model_kwargs
            )
            if use_cache:
                past_key_values.previous_hidden = first.last_hidden_state[:, -1:].detach()
                past_key_values.previous_valid = current_valid[:, -1:].detach()
        selection = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(output.last_hidden_state[:, selection])
        loss = k1_loss = k2_loss = None
        if labels is not None:
            final_loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs
            )
            k1_loss = final_loss
            if passes == 2:
                first_logits = self.lm_head(first.last_hidden_state)
                k1_loss = self.loss_function(
                    logits=first_logits, labels=labels, vocab_size=self.vocab_size, **kwargs
                )
                k2_loss = final_loss
                loss = (k1_loss + k2_loss) / 2
            else:
                loss = k1_loss
        result = PuroFeedbackOutput(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            k1_loss=k1_loss,
            k2_loss=k2_loss,
        )
        return (
            result
            if (self.config.use_return_dict if return_dict is None else return_dict)
            else result.to_tuple()
        )


PuroFeedbackForCausalLM.register_for_auto_class("AutoModelForCausalLM")
