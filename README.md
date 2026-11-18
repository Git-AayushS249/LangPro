
# Context-Aware English–Japanese Neural Machine Translation

A research-based B.Tech major project investigating context-aware English–Japanese translation through Word Sense Disambiguation (WSD) and Hindi-based linguistic pivoting.

## 1. Project Overview

Machine translation systems can produce incorrect or less accurate translations when a source word has multiple meanings depending on its context.

This project explores a hybrid translation framework that combines:

- Neural Machine Translation using NLLB-200.
- Contextual Word Sense Disambiguation using DistilBERT.
- Hindi-based linguistic pivoting for English–Japanese translation.
- Sense-aware sentence clarification before translation.

The primary research focus is improving contextual interpretation and preserving semantic meaning during English–Japanese translation.

A secondary objective is to investigate whether Hindi can serve as a useful linguistic bridge between English and Japanese.

## 2. Problem Statement

Words can have multiple meanings depending on their surrounding context.

For example, the English word "bank" can refer to:

- A financial institution.
- The land alongside a river.

A translation system that fails to identify the intended meaning may produce an incorrect translation.

Japanese also presents additional lexical and Kanji-related interpretation challenges, where readings and meanings depend on context.

This project develops an initial context-aware translation prototype to investigate these challenges.

## 3. Research Objectives

1. Implement English–Japanese neural machine translation.
2. Develop a contextual Word Sense Disambiguation component.
3. Integrate sense-aware sentence clarification into the translation pipeline.
4. Compare direct English–Japanese translation with English–Hindi–Japanese pivot translation.
5. Investigate semantic preservation and translation errors across different approaches.
6. Identify limitations and future directions for Japanese Kanji disambiguation.

## 4. System Architecture

The prototype implements three translation approaches.

### A. Direct Translation

English Input → NLLB-200 → Japanese Output

### B. Hindi Pivot Translation

English Input → Hindi Translation → Japanese Translation

### C. WSD-Assisted Translation

English Input → DistilBERT WSD → Sense Prediction → Sentence Clarification → NLLB-200 → Japanese Output

The WSD component identifies the intended meaning of the target word and replaces ambiguous expressions with contextually clearer alternatives before translation.

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Pretrained model integration |
| NLLB-200 | Neural machine translation |
| DistilBERT | Contextual word sense classification |
| Pandas | Dataset and experiment management |
| Git and GitHub | Version control |

### Models

**NLLB-200 Distilled 600M**
- Model: facebook/nllb-200-distilled-600M
- Used for multilingual neural machine translation.
- Supports English, Hindi and Japanese translation.

**DistilBERT**
- Model: distilbert-base-uncased
- Fine-tuned for English word sense classification.
- Currently trained to distinguish financial and river-related meanings of "bank".

## 6. Project Structure

```text
LangPro/
│
├── data/
│   ├── test/
│   └── wsd/
│
├── models/
│   └── wsd/
│
├── results/
│
├── src/
│   ├── config.py
│   ├── translator.py
│   ├── pipelines.py
│   ├── wsd.py
│   ├── train_wsd.py
│   ├── evaluate.py
│   ├── run_translation.py
│   ├── run_dataset.py
│   ├── run_reverse_dataset.py
│   ├── run_wsd_translation.py
│   └── run_final_experiments.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 7. Installation and Setup

### Prerequisites

- Python 3.11
- Git
- Git LFS
- Internet connection for downloading pretrained models

### Step 1: Clone the repository

```bash
git clone https://github.com/Git-AayushS249/LangPro.git
cd LangPro
```

### Step 2: Download Git LFS files

```bash
git lfs install
git lfs pull
```

### Step 3: Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

## 8. Running the Project

Run commands from the project root directory.

### Direct English–Japanese Translation

```powershell
python src/run_translation.py
```

### English–Hindi–Japanese Pivot Translation

```powershell
python src/run_dataset.py
```

### Reverse Translation Experiment

```powershell
python src/run_reverse_dataset.py
```

### WSD Testing

```powershell
python src/test_wsd.py
```

### WSD-Assisted Translation

```powershell
python src/run_wsd_translation.py
```

### Final Experiments

```powershell
python src/run_final_experiments.py
```

Experiment outputs are saved in the results directory.

## 9. Preliminary Experimental Observations

An initial diagnostic experiment was conducted using 10 manually selected English sentences containing the word "bank".

The dataset included:
- 5 financial-institution examples.
- 5 riverbank examples.

### Observations

- The WSD classifier predicted the intended sense correctly in all 10 selected examples.
- Sense-aware clarification changed Japanese lexical choices in some examples.
- Direct translation handled several examples reasonably but produced errors in some contexts.
- Hindi pivot translation introduced semantic drift in certain examples, including subject changes and incorrect lexical interpretation.
- WSD-assisted translation did not consistently improve the final Japanese output.

These results are preliminary observations from a small diagnostic dataset, not evidence of generalized 100% WSD accuracy or statistically established translation improvement.

## 10. Current Limitations

- The WSD classifier currently supports only the English word "bank".
- The WSD training dataset is small and requires expansion.
- Japanese-side contextual WSD is not yet implemented.
- Dedicated Japanese Kanji reading disambiguation is not yet implemented.
- Conversation-level context is not yet integrated.
- Larger benchmark datasets and human evaluation are required.
- Translation quality needs more comprehensive quantitative evaluation.

## 11. Future Scope

1. Expand contextual WSD to additional ambiguous English words.
2. Develop Japanese-side lexical and Kanji disambiguation.
3. Incorporate sentence-level and conversation-level context.
4. Investigate improved sense-aware translation strategies.
5. Evaluate direct and Hindi-pivot translation using larger datasets.
6. Perform human evaluation with Japanese-proficient participants.
7. Explore bidirectional context-aware English–Japanese translation.

## 12. Research Status

This repository contains the current working research prototype, trained English WSD classifier, datasets, source code and preliminary experimental results.

Japanese Kanji reading disambiguation remains a planned extension rather than a completed component.

## 13. Author

**Aayush Sinha**

B.Tech – Information Technology

KIIT

GitHub: https://github.com/Git-AayushS249

---

*This project is developed for academic and research purposes.*
