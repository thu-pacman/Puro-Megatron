# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""The fixed shift-1, shared-decoder Puro feedback architecture."""

from transformers import Qwen3Config


class PuroFeedbackConfig(Qwen3Config):
    model_type = "puro_feedback"

    def __init__(
        self,
        feedback_passes=2,
        feedback_fusion="residual_glu",
        feedback_shift=1,
        feedback_loss_reduction="mean",
        **kwargs,
    ):
        if feedback_passes not in (1, 2):
            raise ValueError("feedback_passes must be 1 or 2")
        if (feedback_fusion, feedback_shift, feedback_loss_reduction) != (
            "residual_glu",
            1,
            "mean",
        ):
            raise ValueError("Only shift-1 Residual-GLU with mean K1/K2 loss is supported")
        rope_parameters = kwargs.get("rope_parameters") or {}
        if "rope_theta" in rope_parameters:
            if (
                kwargs.get("rope_theta", rope_parameters["rope_theta"])
                != rope_parameters["rope_theta"]
            ):
                raise ValueError("Conflicting rope_theta and rope_parameters")
            kwargs["rope_theta"] = rope_parameters["rope_theta"]
        super().__init__(**kwargs)
        if self.use_sliding_window or any(kind != "full_attention" for kind in self.layer_types):
            raise ValueError("This release supports full-attention decoder layers")
        self.feedback_passes = feedback_passes
        self.feedback_fusion = feedback_fusion
        self.feedback_shift = feedback_shift
        self.feedback_loss_reduction = feedback_loss_reduction


PuroFeedbackConfig.register_for_auto_class()
