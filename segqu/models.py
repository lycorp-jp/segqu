from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForMaskedLM


class SessionPretrainingModel(nn.Module):
    """ModernBERT MLM with either binary SSQP or contrastive SSQP."""

    def __init__(
        self,
        model_name_or_path: str,
        objective: str,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if objective not in {"segqu", "contrastive", "mlm"}:
            raise ValueError(f"Unsupported objective: {objective}")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        self.objective = objective
        self.temperature = temperature
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.mlm_model.config.model_type != "modernbert":
            raise ValueError("SessionPretrainingModel supports only ModernBERT.")
        self.ssqp_classifier = (
            nn.Linear(self.mlm_model.config.hidden_size, 2) if objective == "segqu" else None
        )

    @property
    def encoder(self) -> nn.Module:
        return self.mlm_model.base_model

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        ssqp_label: torch.Tensor | None = None,
        anchor_input_ids: torch.Tensor | None = None,
        anchor_attention_mask: torch.Tensor | None = None,
        positive_input_ids: torch.Tensor | None = None,
        positive_attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | None]:
        kwargs.pop("num_items_in_batch", None)
        need_hidden_states = self.objective == "segqu"
        mlm_output = self.mlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )
        loss = mlm_output.loss

        if self.objective == "segqu":
            if ssqp_label is None or self.ssqp_classifier is None:
                raise ValueError("SEGQU requires ssqp_label.")
            pooled = mlm_output.hidden_states[-1][:, 0]
            ssqp_logits = self.ssqp_classifier(pooled)
            loss = loss + F.cross_entropy(ssqp_logits, ssqp_label.long())

        elif self.objective == "contrastive":
            if anchor_input_ids is None or positive_input_ids is None:
                raise ValueError("Contrastive training requires anchor and positive inputs.")
            anchor_output = self.encoder(
                input_ids=anchor_input_ids,
                attention_mask=anchor_attention_mask,
                return_dict=True,
            )
            positive_output = self.encoder(
                input_ids=positive_input_ids,
                attention_mask=positive_attention_mask,
                return_dict=True,
            )
            anchor = F.normalize(anchor_output.last_hidden_state[:, 0], dim=-1)
            positive = F.normalize(positive_output.last_hidden_state[:, 0], dim=-1)
            # Cross-rank all_gather is not supported; negatives are local to each process.
            similarity = anchor @ positive.transpose(0, 1) / self.temperature
            targets = torch.arange(similarity.shape[0], device=similarity.device)
            loss = loss + F.cross_entropy(similarity, targets)

        return {"loss": loss, "logits": mlm_output.logits}
