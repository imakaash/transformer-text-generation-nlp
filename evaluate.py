import torch
import torch.nn as nn

from models.transformer import TransformerLanguageModel
from utils import clean_text, char_tokenizer, word_tokenizer, create_sequences


device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Configuration
# -------------------------------------------------
TOKEN_LEVEL = "word"   # choose: "char" or "word"
TEXT_PATH = "data/sherlock.txt"
SEQ_LEN = 20


# -------------------------------------------------
# Load and clean text
# -------------------------------------------------
text = open(TEXT_PATH).read()
text = clean_text(text)


# -------------------------------------------------
# Tokenization
# -------------------------------------------------
if TOKEN_LEVEL == "char":
    encoded, stoi, itos = char_tokenizer(text)
    model_path = "char_transformer.pt"

elif TOKEN_LEVEL == "word":
    encoded, stoi, itos = word_tokenizer(text)
    model_path = "word_transformer.pt"

else:
    raise ValueError("TOKEN_LEVEL must be 'char' or 'word'")


# -------------------------------------------------
# Create sequences
# -------------------------------------------------
X, y = create_sequences(encoded, SEQ_LEN)


# -------------------------------------------------
# Load model
# -------------------------------------------------
model = TransformerLanguageModel(len(stoi)).to(device)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.eval()

loss_fn = nn.CrossEntropyLoss()

correct = 0
total = 0
total_loss = 0


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
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


# -------------------------------------------------
# Metrics
# -------------------------------------------------
accuracy = correct / total

perplexity = torch.exp(torch.tensor(total_loss / len(X)))

print(f"\nTokenization Level: {TOKEN_LEVEL}")
print("Accuracy:", accuracy)
print("Perplexity:", perplexity.item())