from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

OBJECTIVES = ("segqu", "contrastive", "mlm")
METADATA_FILE = "metadata.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess query pairs for SEGQU training.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL file.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--objective", required=True, choices=OBJECTIVES)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--validation-size", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _validate_model(model_name_or_path: str) -> None:
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if config.model_type != "modernbert":
        raise ValueError(f"Only ModernBERT is supported, but model_type={config.model_type!r}.")


def read_queries(path: Path, objective: str) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Line {line_number}: invalid JSON: {error.msg}") from error

            queries = item.get("queries") if isinstance(item, dict) else None
            if not isinstance(queries, list) or not queries:
                raise ValueError(f"Line {line_number}: 'queries' must be a non-empty list.")
            if objective != "mlm" and len(queries) != 2:
                raise ValueError(f"Line {line_number}: exactly two queries are required.")
            if not all(isinstance(query, str) and query.strip() for query in queries):
                raise ValueError(f"Line {line_number}: queries must be non-empty strings.")
            rows.append([query.strip() for query in queries])

    if not rows:
        raise ValueError("Input JSONL is empty.")
    return rows


def split_rows(
    rows: list[list[str]], validation_size: int, seed: int
) -> tuple[list[list[str]], list[list[str]]]:
    if validation_size <= 0:
        raise ValueError("--validation-size must be greater than zero.")
    if len(rows) <= validation_size:
        raise ValueError(
            f"Input has {len(rows)} rows, but --validation-size is {validation_size}. "
            "Provide a smaller explicit value."
        )

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    validation_indices = set(indices[:validation_size])
    train = [row for index, row in enumerate(rows) if index not in validation_indices]
    validation = [row for index, row in enumerate(rows) if index in validation_indices]
    return train, validation


def _encode_queries(
    tokenizer: PreTrainedTokenizerBase, queries: list[str], max_length: int
) -> dict[str, Any]:
    text = f"{tokenizer.sep_token}".join(queries)
    return tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
        return_token_type_ids=False,
    )


def _negative_query(query1: str, query2: str, candidates: list[str], rng: random.Random) -> str:
    valid = [candidate for candidate in candidates if candidate not in {query1, query2}]
    if not valid:
        raise ValueError("Cannot construct a random negative from this split.")
    return rng.choice(valid)


def encode_split(
    rows: list[list[str]],
    objective: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    seed: int,
) -> Dataset:
    records: list[dict[str, Any]] = []
    rng = random.Random(seed)

    if objective == "mlm":
        records = [_encode_queries(tokenizer, row, max_length) for row in rows]
    elif objective == "segqu":
        candidates = [row[1] for row in rows]
        for query1, query2 in rows:
            label = int(rng.random() < 0.5 or len(rows) < 2)
            paired_query = query2 if label else _negative_query(query1, query2, candidates, rng)
            record = _encode_queries(tokenizer, [query1, paired_query], max_length)
            record["ssqp_label"] = label
            records.append(record)
    else:
        for query1, query2 in rows:
            record = _encode_queries(tokenizer, [query1, query2], max_length)
            anchor = _encode_queries(tokenizer, [query1], max_length)
            positive = _encode_queries(tokenizer, [query2], max_length)
            record.update(
                {
                    "anchor_input_ids": anchor["input_ids"],
                    "anchor_attention_mask": anchor["attention_mask"],
                    "positive_input_ids": positive["input_ids"],
                    "positive_attention_mask": positive["attention_mask"],
                }
            )
            records.append(record)

    return Dataset.from_list(records)


def preprocess(args: argparse.Namespace) -> None:
    if args.max_length <= 0:
        raise ValueError("--max-length must be greater than zero.")
    if args.output_dir.exists():
        raise ValueError(f"Output path already exists: {args.output_dir}")

    _validate_model(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    rows = read_queries(args.input, args.objective)
    train_rows, validation_rows = split_rows(rows, args.validation_size, args.seed)

    dataset = DatasetDict(
        {
            "train": encode_split(
                train_rows,
                args.objective,
                tokenizer,
                args.max_length,
                args.seed,
            ),
            "validation": encode_split(
                validation_rows,
                args.objective,
                tokenizer,
                args.max_length,
                args.seed + 1,
            ),
        }
    )
    dataset.save_to_disk(str(args.output_dir))
    metadata = {
        "objective": args.objective,
        "model_name_or_path": args.model_name_or_path,
        "max_length": args.max_length,
        "validation_size": args.validation_size,
        "seed": args.seed,
    }
    (args.output_dir / METADATA_FILE).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {len(dataset['train'])} train and {len(dataset['validation'])} "
        f"validation examples to {args.output_dir}."
    )


def main() -> None:
    preprocess(build_parser().parse_args())


if __name__ == "__main__":
    main()
