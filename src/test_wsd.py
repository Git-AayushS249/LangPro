from wsd import WordSenseDisambiguator

wsd = WordSenseDisambiguator()

test_sentences = [
    ("The bank is offering a lower interest rate.", "bank"),
    ("The river overflowed its bank after heavy rain.", "bank"),
    ("He went to the bank to withdraw cash.", "bank"),
    ("They built a house on the bank overlooking the water.", "bank"),
    ("The company borrowed money from the bank.", "bank"),
    ("We walked along the muddy bank.", "bank"),
    ("The bank was closed on Sunday.", "bank"),
    ("The boat was pulled onto the bank.", "bank"),
    ("She works for an international bank.", "bank"),
    ("The trees along the bank were damaged.", "bank"),
]

for sentence, target in test_sentences:
    sense, confidence = wsd.predict_sense(sentence, target)

    print("\nSentence:", sentence)
    print("Predicted meaning:", sense)
    print("Confidence:", round(confidence, 4))