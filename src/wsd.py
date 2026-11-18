
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_DIR = "models/wsd"

# Initial demonstration dictionary.
# We will expand this using annotated data.
CLARIFICATIONS = {
    "bank": {
        "financial": "financial institution",
        "river": "riverbank"
    }
}


class WordSenseDisambiguator:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR
        ).to(self.device)

        self.model.eval()

    def predict_sense(self, sentence, target_word):

        marked_sentence, count = re.subn(
            rf"\b{re.escape(target_word)}\b",
            f"[TGT] {target_word} [/TGT]",
            sentence,
            count=1,
            flags=re.IGNORECASE
        )

        if count == 0:
            raise ValueError(
                f"Target word '{target_word}' not found."
            )

        inputs = self.tokenizer(
            marked_sentence,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = probabilities.max(dim=1)

        sense = self.model.config.id2label[
            prediction.item()
        ]

        return sense, confidence.item()

    def clarify_sentence(self, sentence, target_word, sense):

        if target_word.lower() != "bank":
            raise ValueError(
                f"No clarification rules available for: {target_word}"
            )

        if sense == "financial":

            # Handle "bank account" as a complete phrase.
            sentence = re.sub(
                r"\ba\s+bank\s+account\b",
                "an account at a financial institution",
                sentence,
                flags=re.IGNORECASE
            )

            sentence = re.sub(
                r"\bbank\s+account\b",
                "account at a financial institution",
                sentence,
                flags=re.IGNORECASE
            )

            # Handle remaining occurrences of bank.
            sentence = re.sub(
                r"\bbank\b",
                "financial institution",
                sentence,
                flags=re.IGNORECASE
            )

        elif sense == "river":

            # Handle common riverbank expressions.
            sentence = re.sub(
                r"\bbank\s+of\s+the\s+river\b",
                "shore of the river",
                sentence,
                flags=re.IGNORECASE
            )

            sentence = re.sub(
                r"\briver\s+bank\b",
                "river shore",
                sentence,
                flags=re.IGNORECASE
            )

            sentence = re.sub(
                r"\bbank\b",
                "shore",
                sentence,
                flags=re.IGNORECASE
            )

        else:
            raise ValueError(f"Unknown sense: {sense}")

        return sentence

