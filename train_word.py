import torch
import torch.nn as nn
from models.transformer import TransformerLanguageModel
from utils import word_tokenizer, create_sequences
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

text = open("data/sherlock.txt").read()

encoded, stoi, itos = word_tokenizer(text)

seq_len = 20

X, y = create_sequences(encoded, seq_len)

model = TransformerLanguageModel(len(stoi)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

loss_fn = nn.CrossEntropyLoss()

epochs = 10
batch_size = 64
loss_history = []

for epoch in range(epochs):

    total_loss = 0

    for i in range(0, len(X), batch_size):

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

    print("Epoch", epoch, "Loss", avg_loss)

    loss_history.append(avg_loss)

torch.save(model.state_dict(), "word_transformer.pt")

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("training_loss_word.png")
plt.show()