from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from transformers import EvalPrediction

from segqu.esci.evaluate import evaluate
from segqu.esci.metrics import compute_macro_f1
from segqu.esci.preprocess import preprocess
from segqu.esci.train import DataArguments, EsciTrainingArguments, ModelArguments, train


def test_esci_preprocessing_filters_joins_and_splits_by_query(tmp_path: Path) -> None:
    examples_path = tmp_path / "examples.parquet"
    products_path = tmp_path / "products.parquet"
    output_dir = tmp_path / "processed"
    examples = []
    products = []
    example_id = 0
    for query_id in range(10):
        for label in ("E", "S", "C", "I"):
            product_id = f"p-{example_id}"
            examples.append(
                {
                    "example_id": example_id,
                    "query_id": query_id,
                    "query": f"q{query_id}",
                    "product_id": product_id,
                    "product_locale": "jp",
                    "esci_label": label,
                    "small_version": 1,
                    "split": "train" if query_id < 8 else "test",
                }
            )
            products.append(
                {"product_id": product_id, "product_locale": "jp", "product_title": product_id}
            )
            example_id += 1
    examples.extend(
        [
            {
                **examples[0],
                "example_id": example_id,
                "product_locale": "us",
                "product_id": "ignored-us",
            },
            {
                **examples[0],
                "example_id": example_id + 1,
                "small_version": 0,
                "product_id": "ignored-large",
            },
        ]
    )
    pl.DataFrame(examples).write_parquet(examples_path)
    pl.DataFrame(products).write_parquet(products_path)

    preprocess(examples_path, products_path, output_dir, seed=42)

    train_rows = pl.read_ndjson(output_dir / "train.jsonl")
    dev_rows = pl.read_ndjson(output_dir / "dev.jsonl")
    test_rows = pl.read_ndjson(output_dir / "test.jsonl")
    assert train_rows["query_id"].n_unique() == 7
    assert dev_rows["query_id"].n_unique() == 1
    assert (train_rows.height, dev_rows.height, test_rows.height) == (28, 4, 8)
    assert set(train_rows["query_id"]).isdisjoint(set(dev_rows["query_id"]))
    assert set(test_rows["query_id"]) == {8, 9}
    assert set(train_rows["label"]) == {0, 1, 2, 3}
    assert train_rows["product_title"].to_list() == train_rows["product_id"].to_list()
    assert {example_id, example_id + 1}.isdisjoint(set(train_rows["example_id"]))


def test_macro_f1_uses_all_four_labels() -> None:
    prediction = EvalPrediction(
        predictions=(np.array([[1, 0, 0, 0]], dtype=np.float32),),
        label_ids=np.array([0]),
    )
    assert compute_macro_f1(prediction) == {"macro_f1": 0.25}


def _write_esci_jsonl(path: Path, size: int) -> None:
    rows = [
        {
            "example_id": index,
            "query_id": index,
            "query": f"q{index % 10}",
            "product_id": f"p{index % 10}",
            "product_title": f"p{index % 10}",
            "esci_label": label,
            "label": label_id,
        }
        for index, (label, label_id) in enumerate(
            [("E", 0), ("S", 1), ("C", 2), ("I", 3)] * (size // 4)
        )
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_one_step_esci_training_and_evaluation(tiny_model: Path, tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    dev_file = tmp_path / "dev.jsonl"
    test_file = tmp_path / "test.jsonl"
    model_dir = tmp_path / "model"
    evaluation_dir = tmp_path / "evaluation"
    _write_esci_jsonl(train_file, 8)
    _write_esci_jsonl(dev_file, 4)
    _write_esci_jsonl(test_file, 4)

    train(
        ModelArguments(model_name_or_path=str(tiny_model)),
        DataArguments(train_file=str(train_file), dev_file=str(dev_file), max_length=32),
        EsciTrainingArguments(
            output_dir=str(model_dir),
            max_steps=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            eval_strategy="steps",
            eval_steps=1,
            save_strategy="steps",
            save_steps=1,
            dataloader_num_workers=0,
            report_to="none",
            use_cpu=True,
            disable_tqdm=True,
        ),
    )
    metrics = evaluate(
        test_file,
        model_dir,
        evaluation_dir,
        max_length=32,
        batch_size=4,
        num_workers=0,
        use_cpu=True,
    )

    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert json.loads((evaluation_dir / "metrics.json").read_text()) == metrics
