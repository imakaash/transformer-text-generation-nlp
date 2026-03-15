import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.transformer import TransformerLanguageModel
from utils import char_tokenizer, create_sequences


device = "cuda" if torch.cuda.is_available() else "cpu"

text = open("data/sherlock.txt").read()

encoded, stoi, itos = char_tokenizer(text)

seq_len = 100

X, y = create_sequences(encoded, seq_len)

model = TransformerLanguageModel(len(stoi)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

loss_fn = nn.CrossEntropyLoss()

epochs = 10
batch_size = 64

loss_history = []


for epoch in range(epochs):

    total_loss = 0

    print(f"\nEpoch {epoch+1}/{epochs}")

    for i in tqdm(range(0, len(X), batch_size), desc="Training Batches"):

        xb = X[i:i+batch_size].to(device)
        yb = y[i:i+batch_size].to(device)

        logits = model(xb)

        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            yb.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(X)//batch_size)

    print("Average Loss:", avg_loss)

    loss_history.append(avg_loss)


torch.save(model.state_dict(), "char_transformer.pt")


# Plot training loss
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Character Transformer Training Loss")
plt.savefig("training_loss_char.png")
plt.show()