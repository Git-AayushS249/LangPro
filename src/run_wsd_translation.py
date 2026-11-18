from pipelines import TranslationPipelines

pipeline = TranslationPipelines()

test_sentences = [
    ("I deposited money at the bank.", "bank"),
    ("The fisherman sat near the bank of the river.", "bank"),
    ("She opened a bank account yesterday.", "bank"),
]

for sentence, target_word in test_sentences:

    print("\n" + "=" * 60)
    print("ENGLISH:", sentence)

    # 1. Direct translation
    direct = pipeline.english_to_japanese(sentence)
    print("\nDIRECT JAPANESE:")
    print(direct)

    # 2. Hindi pivot translation
    hindi, pivot = pipeline.english_to_hindi_to_japanese(sentence)
    print("\nHINDI TRANSLATION:")
    print(hindi)
    print("\nPIVOT JAPANESE:")
    print(pivot)

    # 3. WSD-assisted translation
    sense, confidence, clarified, contextual = (
        pipeline.context_aware_english_to_japanese(
            sentence,
            target_word
        )
    )

    print("\nWSD PREDICTED SENSE:", sense)
    print("WSD CONFIDENCE:", round(confidence, 4))
    print("CLARIFIED INPUT:", clarified)
    print("\nWSD-ASSISTED JAPANESE:")
    print(contextual)
