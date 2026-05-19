from __future__ import annotations

import torch
import torch.nn as nn


def interleave_evidence_and_thought(
    evidence_tokens: torch.Tensor,
    thought_tokens: torch.Tensor,
) -> torch.Tensor:
    if evidence_tokens.shape != thought_tokens.shape:
        raise ValueError(
            "evidence_tokens and thought_tokens must have the same shape, "
            f"got {tuple(evidence_tokens.shape)} and {tuple(thought_tokens.shape)}"
        )
    batch_size, num_tokens, hidden_size = evidence_tokens.shape
    stacked = torch.stack((evidence_tokens, thought_tokens), dim=2)
    return stacked.reshape(batch_size, num_tokens * 2, hidden_size)


class LatentClinicalThoughtRefiner(nn.Module):
    """
    Implements latent clinical thought refinement from the paper.

    Given graph-conditioned evidence tokens Z, the doctor model first produces
    initial thought states H^0. At each refinement step, [z_i, h_i^k] pairs are
    processed by the doctor model and only thought positions are updated.
    """

    def __init__(self, doctor_model: nn.Module, refinement_steps: int = 5) -> None:
        super().__init__()
        self.doctor_model = doctor_model
        self.refinement_steps = refinement_steps

    def _last_hidden_state(self, outputs) -> torch.Tensor:
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        raise RuntimeError("Doctor model must return hidden states for latent refinement.")

    def _doctor_hidden(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        outputs = self.doctor_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return self._last_hidden_state(outputs)

    def forward(self, evidence_tokens: torch.Tensor) -> torch.Tensor:
        if evidence_tokens.ndim != 3:
            raise ValueError(f"Expected evidence tokens with shape (B, m, d), got {tuple(evidence_tokens.shape)}")

        thought_tokens = self._doctor_hidden(evidence_tokens)
        refined_sequence = interleave_evidence_and_thought(evidence_tokens, thought_tokens)

        for _ in range(self.refinement_steps):
            hidden = self._doctor_hidden(refined_sequence)
            thought_tokens = hidden[:, 1::2, :]
            refined_sequence = interleave_evidence_and_thought(evidence_tokens, thought_tokens)

        return refined_sequence

