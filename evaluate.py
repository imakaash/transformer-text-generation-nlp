from collections import Counter
import random
import torch
from tqdm import tqdm

from models.transformer import TransformerLanguageModel
from utils import clean_text, split_word_tokens, word_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Config
# -------------------------------------------------
TEXT_PATH = "data/sherlock.txt"
MODEL_PATH = "word_transformer.pt"

MIN_CONTEXT_LEN = 5
MAX_CONTEXT_LEN = 10
TARGET_LEN = 5
NUM_SAMPLES = 20000  # number of evaluation examples

IGNORED_TOKENS = {"<pad>", "<para>"}

# -------------------------------------------------
# Load + Tokenize
# -------------------------------------------------
text = clean_text(open(TEXT_PATH).read())
tokens = split_word_tokens(text)

_, stoi, itos = word_tokenizer(text)

# Convert full corpus to token ids
token_ids = [stoi.get(tok, stoi["<unk>"]) for tok in tokens]

# -------------------------------------------------
# Load Model
# -------------------------------------------------
model = TransformerLanguageModel(
    len(stoi),
    max_len=MAX_CONTEXT_LEN + TARGET_LEN - 1,
    pad_token_id=stoi["<pad>"]
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def decode(token_ids):
    return " ".join(
        itos[t] for t in token_ids if itos[t] not in IGNORED_TOKENS
    )


def generate_step_by_step(context_tokens):
    """Same generation logic as working script"""
    generated = list(context_tokens)

    for _ in range(TARGET_LEN):
        context = generated[-MAX_CONTEXT_LEN:]
        padded = [stoi["<pad>"]] * (MAX_CONTEXT_LEN - len(context)) + context

        x = torch.tensor([padded]).to(device)

        with torch.no_grad():
            logits = model(x)

        probs = torch.softmax(logits[0, -1], dim=0)
        next_token = torch.multinomial(probs, 1).item()

        generated.append(next_token)

    return generated[len(context_tokens):]  # only new tokens


def unordered_match(pred, target):
    """Bag-of-words matching"""
    pred_counts = Counter(pred)
    target_counts = Counter(target)

    matches = pred_counts & target_counts
    return sum(matches.values()), len(target)


# -------------------------------------------------
# Evaluation Loop
# -------------------------------------------------
correct = 0
total = 0

examples = []

valid_starts = list(range(len(token_ids) - MAX_CONTEXT_LEN - TARGET_LEN))
sample_indices = random.sample(valid_starts, min(NUM_SAMPLES, len(valid_starts)))

for idx in tqdm(sample_indices, desc="Evaluating"):

    context = token_ids[idx: idx + MAX_CONTEXT_LEN]
    target = token_ids[idx + MAX_CONTEXT_LEN: idx + MAX_CONTEXT_LEN + TARGET_LEN]

    # Skip bad contexts
    if len(context) < MIN_CONTEXT_LEN:
        continue

    pred = generate_step_by_step(context)

    # Filter ignored tokens
    pred_filtered = [t for t in pred if itos[t] not in IGNORED_TOKENS]
    target_filtered = [t for t in target if itos[t] not in IGNORED_TOKENS]

    matched, count = unordered_match(pred_filtered, target_filtered)

    correct += matched
    total += count

    if len(examples) < 10:
        examples.append((
            decode(context),
            decode(target_filtered),
            decode(pred_filtered),
            matched,
            count
        ))

# -------------------------------------------------
# Results
# -------------------------------------------------
accuracy = correct / total if total else 0.0

print("\nTokenization Level: word")
print("Context Length:", f"{MIN_CONTEXT_LEN}-{MAX_CONTEXT_LEN}")
print("Target Length:", TARGET_LEN)
print("Total Matches:", f"{correct}/{total}")
print("Order-Independent Accuracy:", accuracy)

# -------------------------------------------------
# Examples
# -------------------------------------------------
print("\nSample Predictions:\n")

for i, (ctx, tgt, pred, m, t) in enumerate(examples, 1):
    print(f"Example {i}")
    print("Context :", ctx)
    print(f"Target  : {tgt} ({m}/{t} matched)")
    print("Predicted:", pred)
    print()