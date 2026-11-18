from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import MODEL_NAME


class Translator:
    def __init__(self):
        print("Loading model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME
        )

        print("Model loaded.")

    def translate(self, text, source_language, target_language):

        self.tokenizer.src_lang = source_language

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )

        target_token_id = self.tokenizer.convert_tokens_to_ids(
            target_language
        )

        output = self.model.generate(
            **inputs,
            forced_bos_token_id=target_token_id,
            max_length=256
        )

        result = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return result