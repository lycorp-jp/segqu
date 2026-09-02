from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from transformers import ModernBertConfig, ModernBertForMaskedLM, PreTrainedTokenizerFast


@pytest.fixture()
def tiny_model(tmp_path: Path) -> Path:
    model_dir = tmp_path / "tiny-modernbert"
    model_dir.mkdir()

    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocabulary += [token for index in range(10) for token in (f"q{index}", f"p{index}")]
    backend = Tokenizer(WordPiece({token: index for index, token in enumerate(vocabulary)}))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=64,
    )
    tokenizer.save_pretrained(model_dir)

    config = ModernBertConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=64,
        local_attention=32,
        global_attn_every_n_layers=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    ModernBertForMaskedLM(config).save_pretrained(model_dir)
    return model_dir


@pytest.fixture()
def query_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "queries.jsonl"
    rows = [{"queries": [f"q{index} q{index}", f"p{index} p{index}"]} for index in range(10)]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path
