import re

import torch
from models.transformer import TransformerLanguageModel
from utils import char_tokenizer, word_tokenizer, split_word_tokens

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Configuration
# -------------------------------------------------
TOKEN_LEVEL = "word"   # choose: "char" or "word"
TEXT_PATH = "data/sherlock_cleaned.txt"
MODEL_DIR = "."
PROMPT = "the drowsiness of the drug"
GENERATE_LENGTH = 30


# -------------------------------------------------
# Load text
# -------------------------------------------------
text = open(TEXT_PATH).read()


# -------------------------------------------------
# Tokenization
# -------------------------------------------------
if TOKEN_LEVEL == "char":
    encoded, stoi, itos = char_tokenizer(text)
    model_path = f"{MODEL_DIR}/char_transformer.pt"
    CONTEXT_LEN = 120
    TARGET_LEN = 20

elif TOKEN_LEVEL == "word":
    encoded, stoi, itos = word_tokenizer(text)
    model_path = f"{MODEL_DIR}/word_transformer.pt"
    CONTEXT_LEN = 60
    TARGET_LEN = 5

else:
    raise ValueError("TOKEN_LEVEL must be 'char' or 'word'")


# -------------------------------------------------
# Load model
# -------------------------------------------------
model = TransformerLanguageModel(
    len(stoi),
    max_len=CONTEXT_LEN + TARGET_LEN - 1
).to(device)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.eval()


# -------------------------------------------------
# Text generation function
# -------------------------------------------------
def generate_text(prompt, length=30):

    if TOKEN_LEVEL == "char":
        tokens = [stoi[c] for c in prompt if c in stoi]

    else:
        words = split_word_tokens(prompt.lower())
        tokens = [stoi[w] for w in words if w in stoi]

    if not tokens:
        raise ValueError("Prompt does not contain any known tokens.")

    generated = list(tokens)

    for _ in range(length):

        context_tokens = generated[-CONTEXT_LEN:]
        context = torch.tensor([context_tokens]).to(device)

        logits = model(context)

        probs = torch.softmax(logits[0, -1], dim=0)

        next_token = torch.multinomial(probs, 1).item()

        generated.append(next_token)

    generated_tokens = [itos[i] for i in generated]

    if TOKEN_LEVEL == "char":
        return "".join(generated_tokens)

    output = " ".join(generated_tokens)
    output = output.replace("<para>", "\n\n")

    return re.sub(r"\s+([.,!?])", r"\1", output)


# -------------------------------------------------
# Run generation
# -------------------------------------------------
output = generate_text(PROMPT, GENERATE_LENGTH)

print(f"\nTokenization Level: {TOKEN_LEVEL}")
print("\nGenerated Text:\n")
print(output)
