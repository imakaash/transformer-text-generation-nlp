from collections import Counter
import random
import torch
import torch.nn as nn
from tqdm import tqdm

from models.transformer import TransformerLanguageModel
from utils import clean_text, create_sentence_prediction_sequences, word_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Config
# -------------------------------------------------
TEXT_PATH = "data/sherlock.txt"
MODEL_PATH = "word_transformer.pt"

MIN_CONTEXT_LEN = 5
MAX_CONTEXT_LEN = 10
TARGET_LEN = 5
STRIDE = 1
NUM_SAMPLES = 99999999  # number of evaluation examples
EVAL_BATCH_SIZE = 256
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-8:
    raise ValueError("TRAIN_RATIO, VAL_RATIO, and TEST_RATIO must sum to 1.0.")

torch.manual_seed(SPLIT_SEED)

IGNORED_TOKENS = {"<pad>", "<para>"}

# -------------------------------------------------
# Load + Tokenize
# -------------------------------------------------
text = clean_text(open(TEXT_PATH).read())
_, stoi, itos = word_tokenizer(text)

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

loss_fn = nn.CrossEntropyLoss()

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def decode(token_ids):
    return " ".join(
        itos[t] for t in token_ids if itos[t] not in IGNORED_TOKENS
    )


def generate_step_by_step(context_tokens):
    """Deterministic rollout from one held-out test context."""
    generated = [token for token in context_tokens if token != stoi["<pad>"]]
    initial_context_len = len(generated)

    for _ in range(TARGET_LEN):
        context = generated[-MAX_CONTEXT_LEN:]
        padded = [stoi["<pad>"]] * (MAX_CONTEXT_LEN - len(context)) + context

        x = torch.tensor([padded]).to(device)

        with torch.no_grad():
            logits = model(x)

        next_token_logits = torch.nan_to_num(logits[0, -1], neginf=float("-inf"))
        next_token = torch.argmax(next_token_logits).item()

        generated.append(next_token)

    return generated[initial_context_len:]  # only new tokens


def unordered_match(pred, target):
    """Bag-of-words matching"""
    pred_counts = Counter(pred)
    target_counts = Counter(target)

    matches = pred_counts & target_counts
    return sum(matches.values()), len(target)


def compute_test_loss():

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, len(X_test), EVAL_BATCH_SIZE):
            xb = X_test[i:i+EVAL_BATCH_SIZE].to(device)
            yb = y_test[i:i+EVAL_BATCH_SIZE].to(device)

            teacher_input = torch.cat([xb, yb[:, :-1]], dim=1)
            logits = model(teacher_input)
            target_logits = logits[:, MAX_CONTEXT_LEN - 1:, :]

            loss = loss_fn(
                target_logits.reshape(-1, target_logits.size(-1)),
                yb.reshape(-1)
            )

            total_loss += loss.item()
            num_batches += 1

    avg_test_loss = total_loss / max(1, num_batches)
    perplexity = torch.exp(torch.tensor(avg_test_loss)).item()

    return avg_test_loss, perplexity


# -------------------------------------------------
# Evaluation Loop
# -------------------------------------------------
correct = 0
total = 0

examples = []

test_indices = list(range(len(X_test)))
sample_rng = random.Random(SPLIT_SEED)
sample_indices = sample_rng.sample(test_indices, min(NUM_SAMPLES, len(test_indices)))

for idx in tqdm(sample_indices, desc="Evaluating"):

    context = X_test[idx].tolist()
    target = y_test[idx].tolist()

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
test_loss, perplexity = compute_test_loss()

print("\nTokenization Level: word")
print("Dataset Split:", f"train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")
print("Evaluated Test Samples:", len(sample_indices))
print("Context Length:", f"{MIN_CONTEXT_LEN}-{MAX_CONTEXT_LEN}")
print("Target Length:", TARGET_LEN)
print("Test Loss:", test_loss)
print("Test Perplexity:", perplexity)
print("Total Matches:", f"{correct}/{total}")
print("Order-Independent Test Accuracy:", accuracy)

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
