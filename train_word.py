import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.transformer import TransformerLanguageModel
from utils import clean_text, create_sentence_prediction_sequences, word_tokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

MIN_CONTEXT_LEN = 5
MAX_CONTEXT_LEN = 10
TARGET_LEN = 5
STRIDE = 1
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

text = open("data/sherlock.txt").read()

text = clean_text(text)

with open("data/sherlock_cleaned.txt", "w") as f:
    f.write(text)

_, stoi, _ = word_tokenizer(text)

X, y = create_sentence_prediction_sequences(
    text,
    stoi,
    MIN_CONTEXT_LEN,
    MAX_CONTEXT_LEN,
    TARGET_LEN,
    stride=STRIDE
)

if len(X) == 0:
    raise ValueError("Not enough tokens to build word prediction pairs.")

print("Total context-target pairs:", len(X))
print("Context length range:", f"{MIN_CONTEXT_LEN}-{MAX_CONTEXT_LEN} words")

perm = torch.randperm(len(X))
X = X[perm]
y = y[perm]

model = TransformerLanguageModel(
    len(stoi),
    max_len=MAX_CONTEXT_LEN + TARGET_LEN - 1,
    pad_token_id=stoi["<pad>"]
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

loss_fn = nn.CrossEntropyLoss()

train_loss_history = []

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    num_batches = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for i in tqdm(range(0, len(X), BATCH_SIZE), desc="Training"):

        xb = X[i:i+BATCH_SIZE].to(device)
        yb = y[i:i+BATCH_SIZE].to(device)

        teacher_input = torch.cat([xb, yb[:, :-1]], dim=1)
        logits = model(teacher_input)
        target_logits = logits[:, MAX_CONTEXT_LEN - 1:, :]

        loss = loss_fn(
            target_logits.reshape(-1, target_logits.size(-1)),
            yb.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_train_loss = total_loss / max(1, num_batches)
    scheduler.step()

    print("Train Loss:", avg_train_loss)

    train_loss_history.append(avg_train_loss)
    torch.save(model.state_dict(), "word_transformer.pt")

plt.plot(train_loss_history, label="train")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Word Transformer Training Loss")
plt.legend()
plt.savefig("training_loss_word.png")
plt.show()
