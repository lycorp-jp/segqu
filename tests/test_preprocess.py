from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from datasets import load_from_disk

from segqu.preprocess import METADATA_FILE, preprocess, read_queries, split_rows


def _arguments(
    query_jsonl: Path,
    output_dir: Path,
    tiny_model: Path,
    objective: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        input=query_jsonl,
        output_dir=output_dir,
        model_name_or_path=str(tiny_model),
        objective=objective,
        max_length=32,
        validation_size=2,
        seed=42,
    )


@pytest.mark.parametrize(
    "objective",
    ["segqu", "contrastive", "mlm"],
)
def test_preprocess_objectives(
    tiny_model: Path,
    query_jsonl: Path,
    tmp_path: Path,
    objective: str,
) -> None:
    output_dir = tmp_path / f"processed-{objective}"
    preprocess(_arguments(query_jsonl, output_dir, tiny_model, objective))
    dataset = load_from_disk(str(output_dir))

    assert len(dataset["train"]) == 8
    assert len(dataset["validation"]) == 2
    assert json.loads((output_dir / METADATA_FILE).read_text())["objective"] == objective

    if objective == "segqu":
        assert set(dataset["train"]["ssqp_label"]) == {0, 1}
        for row in dataset["train"]:
            query, paired_query = _split_pair(row["input_ids"])
            assert (paired_query[0] == query[0] + 1) == bool(row["ssqp_label"])
    elif objective == "contrastive":
        for row in dataset["train"]:
            anchor, positive = _split_pair(row["input_ids"])
            assert row["anchor_input_ids"] == anchor
            assert row["positive_input_ids"] == positive
    else:
        second_query_token_ids = set(range(6, 25, 2))
        observed = {token for row in dataset["train"]["input_ids"] for token in row}
        assert observed.intersection(second_query_token_ids)

    assert all(
        3 in input_ids
        for split in (dataset["train"], dataset["validation"])
        for input_ids in split["input_ids"]
    )


def _split_pair(input_ids: list[int]) -> tuple[list[int], list[int]]:
    separator = input_ids.index(3)
    return input_ids[:separator], input_ids[separator + 1 :]


def test_segqu_preprocessing_is_deterministic(
    tiny_model: Path, query_jsonl: Path, tmp_path: Path
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    preprocess(_arguments(query_jsonl, first, tiny_model, "segqu"))
    preprocess(_arguments(query_jsonl, second, tiny_model, "segqu"))

    first_dataset = load_from_disk(str(first))
    second_dataset = load_from_disk(str(second))
    assert first_dataset["train"][:] == second_dataset["train"][:]
    assert first_dataset["validation"][:] == second_dataset["validation"][:]


def test_input_validation_reports_line_number(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        '{"queries":["valid","valid"]}\n{"queries":["valid",""]}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Line 2"):
        read_queries(invalid, "segqu")


@pytest.mark.parametrize("validation_size", [0, 1])
def test_invalid_validation_size_is_rejected(validation_size: int) -> None:
    with pytest.raises(ValueError):
        split_rows([["q", "p"]], validation_size=validation_size, seed=42)
