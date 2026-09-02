from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from segqu.esci import ID_TO_LABEL, LABEL_TO_ID
from segqu.esci.metrics import compute_macro_f1


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Encoder checkpoint."})


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Training JSONL created by preprocessing."})
    dev_file: str = field(metadata={"help": "Development JSONL created by preprocessing."})
    max_length: int = 128


@dataclass
class EsciTrainingArguments(TrainingArguments):
    output_dir: str = field(metadata={"help": "Model output directory."})
    num_train_epochs: float = 3.0
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str | None = "eval_macro_f1"
    greater_is_better: bool | None = True
    save_total_limit: int | None = 1
    dataloader_num_workers: int = 4
    report_to: list[str] | None = field(default_factory=lambda: ["none"])


def tokenize(dataset: DatasetDict, tokenizer: Any, max_length: int) -> DatasetDict:
    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["query"], batch["product_title"], truncation=True, max_length=max_length
        )
        encoded["labels"] = batch["label"]
        return encoded

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing query-title pairs",
    )


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: EsciTrainingArguments,
) -> None:
    set_seed(training_args.seed)
    dataset = load_dataset(
        "json", data_files={"train": data_args.train_file, "dev": data_args.dev_file}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenized = tokenize(dataset, tokenizer, data_args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_macro_f1,
    )
    trainer.train()
    trainer.save_model()


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, EsciTrainingArguments))
    train(*parser.parse_args_into_dataclasses())


if __name__ == "__main__":
    main()
