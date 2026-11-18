import pandas as pd

from pipelines import TranslationPipelines


INPUT_FILE = "data/test/reverse_test.csv"
OUTPUT_FILE = "results/reverse_results.csv"


def main():

    df = pd.read_csv(INPUT_FILE)

    system = TranslationPipelines()

    results = []

    for _, row in df.iterrows():

        japanese = row["japanese"]
        reference = row["english"]

        direct = system.japanese_to_english(
            japanese
        )

        hindi, pivot = (
            system.japanese_to_hindi_to_english(
                japanese
            )
        )

        results.append({
            "id": row["id"],
            "japanese": japanese,
            "hindi": hindi,
            "direct_english": direct,
            "pivot_english": pivot,
            "reference_english": reference
        })

    pd.DataFrame(results).to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()