import pandas as pd
from pipelines import TranslationPipelines

pipeline = TranslationPipelines()

df = pd.read_csv("data/test/final_test.csv")

results = []

for _, row in df.iterrows():

    sentence = row["english"]

    print("\n" + "=" * 60)
    print("INPUT:", sentence)

    # Direct translation
    direct = pipeline.english_to_japanese(sentence)

    # Hindi pivot
    hindi, pivot = pipeline.english_to_hindi_to_japanese(sentence)

    # WSD-assisted translation
    try:
        sense, confidence, clarified, contextual = (
            pipeline.context_aware_english_to_japanese(
                sentence, "bank"
            )
        )
    except Exception as error:
        sense = "ERROR"
        confidence = 0
        clarified = ""
        contextual = str(error)

    print("DIRECT:", direct)
    print("HINDI:", hindi)
    print("PIVOT:", pivot)
    print("SENSE:", sense)
    print("CLARIFIED:", clarified)
    print("WSD:", contextual)

    results.append({
        "english": sentence,
        "direct_japanese": direct,
        "hindi": hindi,
        "pivot_japanese": pivot,
        "predicted_sense": sense,
        "confidence": confidence,
        "clarified_input": clarified,
        "wsd_japanese": contextual
    })

output = pd.DataFrame(results)

output.to_csv(
    "results/final_experiment_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nExperiment completed.")
print("Results saved to results/final_experiment_results.csv")
