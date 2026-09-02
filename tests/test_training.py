from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from segqu.models import SessionPretrainingModel
from segqu.preprocess import preprocess
from segqu.train import DataArguments, ModelArguments, SegquTrainingArguments, train


def test_one_step_training_and_encoder_export(
    tiny_model: Path, query_jsonl: Path, tmp_path: Path
) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "output"
    preprocess(
        argparse.Namespace(
            input=query_jsonl,
            output_dir=processed,
            model_name_or_path=str(tiny_model),
            objective="segqu",
            max_length=32,
            validation_size=2,
            seed=42,
        )
    )
    train(
        ModelArguments(model_name_or_path=str(tiny_model)),
        DataArguments(processed_data_path=str(processed), mlm_probability=0.9, temperature=0.05),
        SegquTrainingArguments(
            output_dir=str(output),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1.0,
            logging_steps=1,
            eval_steps=1,
            save_strategy="no",
            dataloader_num_workers=0,
            report_to="none",
            max_steps=1,
            use_cpu=True,
        ),
    )

    model = AutoModel.from_pretrained(output)
    tokenizer = AutoTokenizer.from_pretrained(output)
    assert model.config.model_type == "modernbert"
    assert tokenizer.mask_token_id is not None


@pytest.mark.parametrize("objective", ["segqu", "contrastive", "mlm"])
def test_objective_specific_loss(tiny_model: Path, objective: str) -> None:
    model = SessionPretrainingModel(str(tiny_model), objective).eval()
    input_ids = torch.tensor([[2, 5, 4, 3, 6], [2, 7, 4, 3, 8]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, 5, -100, -100], [-100, -100, 7, -100, -100]])
    arguments = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    with torch.no_grad():
        mlm_loss = model.mlm_model(**arguments).loss
        if objective == "segqu":
            arguments["ssqp_label"] = torch.tensor([0, 1])
        elif objective == "contrastive":
            auxiliary = input_ids[:, :2].repeat(2, 1)
            arguments.update(
                anchor_input_ids=auxiliary,
                anchor_attention_mask=torch.ones_like(auxiliary),
                positive_input_ids=auxiliary,
                positive_attention_mask=torch.ones_like(auxiliary),
            )
        loss = model(**arguments)["loss"]

    assert torch.isfinite(loss)
    assert (loss > mlm_loss) == (objective != "mlm")
