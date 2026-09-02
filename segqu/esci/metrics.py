from __future__ import annotations

from sklearn.metrics import f1_score
from transformers import EvalPrediction

from segqu.esci import LABEL_TO_ID


def compute_macro_f1(prediction: EvalPrediction) -> dict[str, float]:
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted_labels = logits.argmax(axis=-1)
    return {
        "macro_f1": float(
            f1_score(
                prediction.label_ids,
                predicted_labels,
                labels=list(LABEL_TO_ID.values()),
                average="macro",
                zero_division=0,
            )
        )
    }
