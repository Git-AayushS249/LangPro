from translator import Translator
from config import ENGLISH, HINDI, JAPANESE
from wsd import WordSenseDisambiguator


class TranslationPipelines:

    def __init__(self):
        self.translator = Translator()
        self.wsd = None

    # --------------------------------
    # English → Japanese
    # --------------------------------

    def english_to_japanese(self, text):

        return self.translator.translate(
            text,
            ENGLISH,
            JAPANESE
        )

    # --------------------------------
    # English → Hindi → Japanese
    # --------------------------------

    def english_to_hindi_to_japanese(self, text):

        hindi = self.translator.translate(
            text,
            ENGLISH,
            HINDI
        )

        japanese = self.translator.translate(
            hindi,
            HINDI,
            JAPANESE
        )

        return hindi, japanese

    # --------------------------------
    # Japanese → English
    # --------------------------------

    def japanese_to_english(self, text):

        return self.translator.translate(
            text,
            JAPANESE,
            ENGLISH
        )

    # --------------------------------
    # Japanese → Hindi → English
    # --------------------------------

    def japanese_to_hindi_to_english(self, text):

        hindi = self.translator.translate(
            text,
            JAPANESE,
            HINDI
        )

        english = self.translator.translate(
            hindi,
            HINDI,
            ENGLISH
        )

        return hindi, english

    def context_aware_english_to_japanese(self, text, target_word):
        if self.wsd is None:
            self.wsd = WordSenseDisambiguator()

        sense, confidence = self.wsd.predict_sense(
            text,
            target_word
        )

        clarified_text = self.wsd.clarify_sentence(
            text,
            target_word,
            sense
        )

        japanese = self.translator.translate(
            clarified_text,
            ENGLISH,
            JAPANESE
        )

        return sense, confidence, clarified_text, japanese
