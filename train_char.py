import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.transformer import TransformerLanguageModel
from utils import clean_text, char_tokenizer, create_future_prediction_sequences


device = "cuda" if torch.cuda.is_available() else "cpu"

CONTEXT_LEN = 120
TARGET_LEN = 20
STRIDE = 3
EPOCHS = 10
BATCH_SIZE = 64

text = open("data/sherlock.txt").read()

text = clean_text(text)

with open("data/sherlock_cleaned.txt", "w") as f:
    f.write(text)

encoded, stoi, itos = char_tokenizer(text)

X, y = create_future_prediction_sequences(
    encoded,
    CONTEXT_LEN,
    TARGET_LEN,
    stride=STRIDE
)

if len(X) == 0:
    raise ValueError("Not enough characters to build prediction pairs.")

print("Total context-target pairs:", len(X))

perm = torch.randperm(len(X))
X = X[perm]
y = y[perm]

model = TransformerLanguageModel(
    len(stoi),
    max_len=CONTEXT_LEN + TARGET_LEN - 1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

loss_history = []

for epoch in range(EPOCHS):

    total_loss = 0
    num_batches = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for i in tqdm(range(0, len(X), BATCH_SIZE), desc="Training"):

        xb = X[i:i+BATCH_SIZE].to(device)
        yb = y[i:i+BATCH_SIZE].to(device)

        teacher_input = torch.cat([xb, yb[:, :-1]], dim=1)
        logits = model(teacher_input)
        target_logits = logits[:, CONTEXT_LEN - 1:, :]

        loss = loss_fn(
            target_logits.reshape(-1, target_logits.size(-1)),
            yb.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)

    print("Average Loss:", avg_loss)

    loss_history.append(avg_loss)

torch.save(model.state_dict(), "char_transformer.pt")

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Character Transformer Training Loss")
plt.savefig("training_loss_char.png")
plt.show()
