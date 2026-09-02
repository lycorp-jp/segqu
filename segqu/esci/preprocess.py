from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import polars as pl

from segqu.esci import LABEL_TO_ID


def split_query_ids(query_ids: list[int], seed: int) -> tuple[set[int], set[int]]:
    query_ids = sorted(set(query_ids))
    random.Random(seed).shuffle(query_ids)
    development_size = min(max(round(len(query_ids) * 0.1), 1), len(query_ids) - 1)
    return set(query_ids[development_size:]), set(query_ids[:development_size])


def preprocess(examples_path: Path, products_path: Path, output_dir: Path, seed: int) -> None:
    examples = pl.read_parquet(
        examples_path,
        columns=[
            "example_id",
            "query_id",
            "query",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
            "split",
        ],
    ).filter((pl.col("small_version") == 1) & (pl.col("product_locale") == "jp"))
    products = pl.read_parquet(
        products_path,
        columns=["product_id", "product_locale", "product_title"],
    ).filter(pl.col("product_locale") == "jp")

    data = (
        examples.join(products, on=["product_id", "product_locale"], how="left")
        .with_columns(
            pl.col("query").fill_null(""),
            pl.col("product_title").fill_null(""),
            pl.col("esci_label").replace_strict(LABEL_TO_ID).alias("label"),
        )
        .select(
            "example_id",
            "query_id",
            "query",
            "product_id",
            "product_title",
            "esci_label",
            "label",
            "split",
        )
    )

    official_train = data.filter(pl.col("split") == "train")
    train_ids, development_ids = split_query_ids(official_train["query_id"].to_list(), seed)
    splits = {
        "train": official_train.filter(pl.col("query_id").is_in(train_ids)),
        "dev": official_train.filter(pl.col("query_id").is_in(development_ids)),
        "test": data.filter(pl.col("split") == "test"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, frame in splits.items():
        frame.drop("split").sort(["query_id", "example_id"]).write_ndjson(
            output_dir / f"{name}.jsonl"
        )
        counts[name] = {"rows": frame.height, "queries": frame["query_id"].n_unique()}
    print(json.dumps(counts, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Japanese reduced ESCI dataset.")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    preprocess(args.examples, args.products, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
