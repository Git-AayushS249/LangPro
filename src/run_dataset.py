import pandas as pd

from pipelines import TranslationPipelines


INPUT_FILE = "data/test/test.csv"
OUTPUT_FILE = "results/initial_results.csv"


def main():

    df = pd.read_csv(INPUT_FILE)

    system = TranslationPipelines()

    results = []

    for _, row in df.iterrows():

        sentence_id = row["id"]
        english = row["english"]
        reference = row["japanese"]

        print(f"Processing sentence {sentence_id}")

        # Direct
        direct = system.english_to_japanese(
            english
        )

        # Pivot
        hindi, pivot = (
            system.english_to_hindi_to_japanese(
                english
            )
        )

        results.append({
            "id": sentence_id,
            "english": english,
            "hindi": hindi,
            "direct_japanese": direct,
            "pivot_japanese": pivot,
            "reference_japanese": reference
        })

    output_df = pd.DataFrame(results)

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nResults saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()