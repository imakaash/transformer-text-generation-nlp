import torch
from models.transformer import TransformerLanguageModel
from utils import char_tokenizer, word_tokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Configuration
# -------------------------------------------------
TOKEN_LEVEL = "char"      # choose: "char" or "word"
TEXT_PATH = "data/sherlock.txt"
GENERATE_TOKENS = 200
CONTEXT = "to sherlock holmes"

# -------------------------------------------------
# Load text
# -------------------------------------------------
text = open(TEXT_PATH).read()

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
# Load model
# -------------------------------------------------
model = TransformerLanguageModel(len(stoi)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -------------------------------------------------
# Prepare context
# -------------------------------------------------
if TOKEN_LEVEL == "char":
    context_tokens = [stoi[c] for c in CONTEXT]

elif TOKEN_LEVEL == "word":
    context_tokens = [stoi[w] for w in CONTEXT.split() if w in stoi]

context = torch.tensor([context_tokens])

# -------------------------------------------------
# Text Generation
# -------------------------------------------------
for _ in range(GENERATE_TOKENS):

    logits = model(context.to(device))

    probs = torch.softmax(logits[0, -1], dim=0)

    next_token = torch.multinomial(probs, 1).item()

    context = torch.cat([context, torch.tensor([[next_token]])], dim=1)

# -------------------------------------------------
# Decode output
# -------------------------------------------------
generated_tokens = [itos[i] for i in context[0].tolist()]

if TOKEN_LEVEL == "char":
    generated = "".join(generated_tokens)
else:
    generated = " ".join(generated_tokens)

print(f"\nTokenization Level: {TOKEN_LEVEL}")
print("\nGenerated Text:\n")
print(generated)