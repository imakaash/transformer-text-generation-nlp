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
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42
MODEL_PATH = "word_transformer.pt"

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-8:
    raise ValueError("TRAIN_RATIO, VAL_RATIO, and TEST_RATIO must sum to 1.0.")

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

split_generator = torch.Generator().manual_seed(SPLIT_SEED)
perm = torch.randperm(len(X), generator=split_generator)
X = X[perm]
y = y[perm]

num_pairs = len(X)
train_end = int(num_pairs * TRAIN_RATIO)
val_end = train_end + int(num_pairs * VAL_RATIO)

if train_end == 0 or val_end == train_end or val_end >= num_pairs:
    raise ValueError("Dataset is too small to create train/validation/test splits.")

X_train = X[:train_end]
y_train = y[:train_end]
X_val = X[train_end:val_end]
y_val = y[train_end:val_end]
X_test = X[val_end:]
y_test = y[val_end:]

print("Train pairs:", len(X_train))
print("Validation pairs:", len(X_val))
print("Test pairs:", len(X_test))

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
val_loss_history = []
best_val_loss = float("inf")


def compute_loss(split_x, split_y):

    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, len(split_x), BATCH_SIZE):

            xb = split_x[i:i+BATCH_SIZE].to(device)
            yb = split_y[i:i+BATCH_SIZE].to(device)

            teacher_input = torch.cat([xb, yb[:, :-1]], dim=1)
            logits = model(teacher_input)
            target_logits = logits[:, MAX_CONTEXT_LEN - 1:, :]

            loss = loss_fn(
                target_logits.reshape(-1, target_logits.size(-1)),
                yb.reshape(-1)
            )

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    num_batches = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for i in tqdm(range(0, len(X_train), BATCH_SIZE), desc="Training"):

        xb = X_train[i:i+BATCH_SIZE].to(device)
        yb = y_train[i:i+BATCH_SIZE].to(device)

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
    avg_val_loss = compute_loss(X_val, y_val)
    scheduler.step()

    print("Train Loss:", avg_train_loss)
    print("Validation Loss:", avg_val_loss)

    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print("Saved best model checkpoint.")

plt.plot(train_loss_history, label="train")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Word Transformer Training Loss")
plt.legend()
plt.savefig("training_loss_word.png")
plt.show()

print("\nBest Validation Loss:", best_val_loss)
print("Held-out Test Pairs:", len(X_test))
