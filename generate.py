import torch

from models.transformer import TransformerLanguageModel
from utils import clean_text, char_tokenizer, word_tokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Configuration
# -------------------------------------------------
TOKEN_LEVEL = "word"   # choose: "char" or "word"
TEXT_PATH = "data/sherlock.txt"
MODEL_DIR = "."
PROMPT = "to sherlock holmes she is always"
GENERATE_LENGTH = 20


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
    model_path = f"{MODEL_DIR}/char_transformer.pt"

elif TOKEN_LEVEL == "word":
    encoded, stoi, itos = word_tokenizer(text)
    model_path = f"{MODEL_DIR}/word_transformer.pt"

else:
    raise ValueError("TOKEN_LEVEL must be 'char' or 'word'")


# -------------------------------------------------
# Load model
# -------------------------------------------------
model = TransformerLanguageModel(len(stoi)).to(device)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.eval()


# -------------------------------------------------
# Text generation function
# -------------------------------------------------
def generate_text(prompt, length=20):

    if TOKEN_LEVEL == "char":
        tokens = [stoi[c] for c in prompt]

    else:
        words = prompt.lower().split()
        tokens = [stoi[w] for w in words if w in stoi]

    context = torch.tensor([tokens]).to(device)

    for _ in range(length):

        logits = model(context)

        probs = torch.softmax(logits[0, -1], dim=0)

        next_token = torch.multinomial(probs, 1).item()

        context = torch.cat(
            [context, torch.tensor([[next_token]]).to(device)],
            dim=1
        )

    generated_tokens = [itos[i] for i in context[0].tolist()]

    if TOKEN_LEVEL == "char":
        return "".join(generated_tokens)

    return " ".join(generated_tokens)


# -------------------------------------------------
# Run generation
# -------------------------------------------------
output = generate_text(PROMPT, GENERATE_LENGTH)

print(f"\nTokenization Level: {TOKEN_LEVEL}")
print("\nGenerated Text:\n")
print(output)