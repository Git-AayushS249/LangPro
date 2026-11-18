from pipelines import TranslationPipelines


def main():

    system = TranslationPipelines()

    english = "The boy is eating an apple."

    print("\n==============================")
    print("ENGLISH → JAPANESE")
    print("==============================")

    direct_japanese = system.english_to_japanese(
        english
    )

    print("English:")
    print(english)

    print("\nDirect Japanese:")
    print(direct_japanese)

    print("\n==============================")
    print("ENGLISH → HINDI → JAPANESE")
    print("==============================")

    hindi, pivot_japanese = (
        system.english_to_hindi_to_japanese(
            english
        )
    )

    print("Hindi:")
    print(hindi)

    print("\nPivot Japanese:")
    print(pivot_japanese)


if __name__ == "__main__":
    main()