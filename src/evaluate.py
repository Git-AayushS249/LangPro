import pandas as pd
import sacrebleu


INPUT_FILE = "results/initial_results.csv"


def main():

    df = pd.read_csv(INPUT_FILE)

    references = [
        str(x)
        for x in df["reference_japanese"]
    ]

    direct_predictions = [
        str(x)
        for x in df["direct_japanese"]
    ]

    pivot_predictions = [
        str(x)
        for x in df["pivot_japanese"]
    ]

    direct_bleu = sacrebleu.corpus_bleu(
        direct_predictions,
        [references]
    )

    pivot_bleu = sacrebleu.corpus_bleu(
        pivot_predictions,
        [references]
    )

    direct_chrf = sacrebleu.corpus_chrf(
        direct_predictions,
        [references]
    )

    pivot_chrf = sacrebleu.corpus_chrf(
        pivot_predictions,
        [references]
    )

    print("\n========== RESULTS ==========")

    print("\nDirect EN → JA")
    print("BLEU :", direct_bleu.score)
    print("chrF :", direct_chrf.score)

    print("\nPivot EN → HI → JA")
    print("BLEU :", pivot_bleu.score)
    print("chrF :", pivot_chrf.score)


if __name__ == "__main__":
    main()