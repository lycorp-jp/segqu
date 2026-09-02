from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from segqu.esci.metrics import compute_macro_f1


def evaluate(
    data_file: Path,
    model_dir: Path,
    output_dir: Path,
    max_length: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
    use_cpu: bool = False,
) -> dict[str, float]:
    dataset = load_dataset("json", data_files=str(data_file), split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["query"], batch["product_title"], truncation=True, max_length=max_length
        )
        encoded["labels"] = batch["label"]
        return encoded

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing query-title pairs",
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=num_workers,
            report_to="none",
            use_cpu=use_cpu,
        ),
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_macro_f1,
    )
    result = trainer.predict(tokenized)
    metrics = {"macro_f1": float(result.metrics["test_macro_f1"])}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ESCI relevance classifier.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-cpu", action="store_true")
    args = parser.parse_args()
    evaluate(
        args.data_file,
        args.model_dir,
        args.output_dir,
        args.max_length,
        args.batch_size,
        args.num_workers,
        args.use_cpu,
    )


if __name__ == "__main__":
    main()
