
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.model_selection import train_test_split

MODEL_NAME = "distilbert-base-uncased"
DATA_FILE = "data/wsd/bank_dataset.csv"
SAVE_DIR = "models/wsd"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

df = pd.read_csv(DATA_FILE)

# Remove leading and trailing spaces
df["sense"] = df["sense"].str.strip().str.lower()

print("Dataset distribution:")
print(df["sense"].value_counts())


labels = sorted(df["sense"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df["label"] = df["sense"].map(label2id)

if df["label"].value_counts().min() < 2:
    raise ValueError(
        "Every class needs at least 2 examples. "
        "Add more examples to your dataset."
    )

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class WSDDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = tokenizer(
            self.texts[index],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return item


train_loader = DataLoader(
    WSDDataset(train_df),
    batch_size=2,
    shuffle=True
)

test_loader = DataLoader(
    WSDDataset(test_df),
    batch_size=2
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS}, "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=1)

        correct += (
            predictions == batch["labels"]
        ).sum().item()

        total += batch["labels"].size(0)

print("Test accuracy:", correct / total)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("WSD model saved successfully.")

tokenizer.add_special_tokens({
    "additional_special_tokens": ["[TGT]", "[/TGT]"]
})

model.resize_token_embeddings(len(tokenizer))