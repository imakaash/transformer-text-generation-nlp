import torch
import torch.nn as nn
from models.transformer import TransformerLanguageModel
from utils import char_tokenizer, create_sequences


device = "cuda" if torch.cuda.is_available() else "cpu"

text = open("data/sherlock.txt").read()

encoded, stoi, itos = char_tokenizer(text)

seq_len = 100

X, y = create_sequences(encoded, seq_len)

model = TransformerLanguageModel(len(stoi)).to(device)

model.load_state_dict(torch.load("char_transformer.pt"))

model.eval()

loss_fn = nn.CrossEntropyLoss()

correct = 0
total = 0
total_loss = 0

with torch.no_grad():

    for i in range(len(X)):

        xb = X[i].unsqueeze(0).to(device)
        yb = y[i].unsqueeze(0).to(device)

        logits = model(xb)

        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            yb.view(-1)
        )

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)

        correct += (preds == yb).sum().item()
        total += yb.numel()


accuracy = correct / total

perplexity = torch.exp(torch.tensor(total_loss / len(X)))

print("Accuracy:", accuracy)
print("Perplexity:", perplexity.item())