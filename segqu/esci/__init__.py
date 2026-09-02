"""Japanese ESCI relevance evaluation."""

LABEL_TO_ID = {"E": 0, "S": 1, "C": 2, "I": 3}
ID_TO_LABEL = {label_id: label for label, label_id in LABEL_TO_ID.items()}
