from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from segqu.models import SessionPretrainingModel
from segqu.preprocess import METADATA_FILE, OBJECTIVES


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "ModernBERT checkpoint."})


@dataclass
class DataArguments:
    processed_data_path: str = field(metadata={"help": "Dataset created by segqu-preprocess."})
    mlm_probability: float = 0.30
    temperature: float = 0.05


@dataclass
class SegquTrainingArguments(TrainingArguments):
    output_dir: str = field(metadata={"help": "Model and checkpoint output directory."})
    num_train_epochs: float = 5.0
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.01
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: float = 4_000
    eval_strategy: str = "steps"
    eval_steps: float = 4_000
    save_strategy: str = "steps"
    save_steps: float = 4_000
    save_total_limit: int | None = 1
    dataloader_num_workers: int = 8
    report_to: list[str] | None = field(default_factory=lambda: ["none"])
    metric_for_best_model: str | None = "eval_loss"
    greater_is_better: bool | None = False
    remove_unused_columns: bool = False
    label_names: list[str] | None = field(default_factory=lambda: ["labels"])
    save_safetensors: bool = False


class ObjectiveDataCollator:
    def __init__(self, tokenizer: Any, objective: str, mlm_probability: float) -> None:
        self.objective = objective
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
        )
        self.padding_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def _auxiliary_batch(self, features: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
        auxiliary = [
            {
                "input_ids": feature[f"{prefix}_input_ids"],
                "attention_mask": feature[f"{prefix}_attention_mask"],
            }
            for feature in features
        ]
        padded = self.padding_collator(auxiliary)
        return {
            f"{prefix}_input_ids": padded["input_ids"],
            f"{prefix}_attention_mask": padded["attention_mask"],
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        main_features = [
            {
                key: value
                for key, value in feature.items()
                if not key.startswith(("anchor_", "positive_")) and key != "ssqp_label"
            }
            for feature in features
        ]
        batch = self.mlm_collator(main_features)
        if self.objective == "segqu":
            batch["ssqp_label"] = torch.tensor(
                [feature["ssqp_label"] for feature in features], dtype=torch.long
            )
        elif self.objective == "contrastive":
            batch.update(self._auxiliary_batch(features, "anchor"))
            batch.update(self._auxiliary_batch(features, "positive"))
        return batch


def _load_metadata(processed_data_path: Path) -> dict[str, Any]:
    metadata_path = processed_data_path / METADATA_FILE
    if not metadata_path.is_file():
        raise ValueError(f"Preprocessing metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    objective = metadata.get("objective", metadata.get("mode"))
    if objective not in OBJECTIVES:
        raise ValueError(f"Invalid objective in metadata: {objective!r}")
    metadata["objective"] = objective
    return metadata


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: SegquTrainingArguments,
) -> None:
    if training_args.per_device_train_batch_size <= 0:
        raise ValueError("--per_device_train_batch_size must be greater than zero.")
    if not 0 < data_args.mlm_probability < 1:
        raise ValueError("--mlm_probability must be between zero and one.")

    processed_data_path = Path(data_args.processed_data_path)
    metadata = _load_metadata(processed_data_path)
    if metadata["model_name_or_path"] != model_args.model_name_or_path:
        raise ValueError(
            "--model_name_or_path must match the model used by segqu-preprocess: "
            f"{metadata['model_name_or_path']}"
        )
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if config.model_type != "modernbert":
        raise ValueError(f"Only ModernBERT is supported, got {config.model_type!r}.")

    dataset = load_from_disk(str(processed_data_path))
    if not isinstance(dataset, DatasetDict) or not {"train", "validation"} <= set(dataset):
        raise ValueError("Processed data must contain train and validation splits.")

    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    objective = metadata["objective"]
    print(f"Training objective: {objective}")
    model = SessionPretrainingModel(
        model_args.model_name_or_path,
        objective=objective,
        temperature=data_args.temperature,
    )
    collator = ObjectiveDataCollator(tokenizer, objective, data_args.mlm_probability)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        processing_class=tokenizer,
    )
    # Resuming from a Trainer checkpoint is not supported.
    trainer.train()

    if trainer.is_world_process_zero():
        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.encoder.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        (output_dir / "training_metadata.json").write_text(
            json.dumps(
                {
                    "preprocessing": metadata,
                    "objective_arguments": {
                        "mlm_probability": data_args.mlm_probability,
                        "temperature": data_args.temperature,
                    },
                    "training_arguments": training_args.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved the encoder and tokenizer to {output_dir}.")


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, SegquTrainingArguments))
    train(*parser.parse_args_into_dataclasses())


if __name__ == "__main__":
    main()
