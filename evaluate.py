import re

import torch
import torch.nn as nn
from tqdm import tqdm

from models.transformer import TransformerLanguageModel
from utils import (
    char_tokenizer,
    word_tokenizer,
    create_future_prediction_sequences,
)


device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Configuration
# -------------------------------------------------
TOKEN_LEVEL = "word"   # choose: "char" or "word"
TEXT_PATH = "data/sherlock_cleaned.txt"
BATCH_SIZE = 64
EVAL_SPLIT = 0.1
SHOW_EXAMPLES = 3
MIN_CORRECT_TOKENS = 3


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
    CONTEXT_LEN = 120
    TARGET_LEN = 20

elif TOKEN_LEVEL == "word":
    encoded, stoi, itos = word_tokenizer(text)
    model_path = "word_transformer.pt"
    CONTEXT_LEN = 60
    TARGET_LEN = 5

else:
    raise ValueError("TOKEN_LEVEL must be 'char' or 'word'")


def decode_tokens(token_ids):

    tokens = [itos[idx] for idx in token_ids]

    if TOKEN_LEVEL == "char":
        return "".join(tokens)

    text = " ".join(tokens).replace("<para>", "\n\n")

    return re.sub(r"\s+([.,!?])", r"\1", text)


# -------------------------------------------------
# Create sequences
# -------------------------------------------------
X, y = create_future_prediction_sequences(
    encoded,
    CONTEXT_LEN,
    TARGET_LEN
)

if len(X) == 0:
    raise ValueError("Not enough tokens to build evaluation pairs.")

num_eval = max(1, int(len(X) * EVAL_SPLIT))
X = X[-num_eval:]
y = y[-num_eval:]

print("Evaluation pairs:", len(X))


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


loss_fn = nn.CrossEntropyLoss()

correct = 0
total_tokens = 0
threshold_matches = 0
exact_matches = 0
total_loss = 0
examples = []


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
with torch.no_grad():

    for i in tqdm(range(0, len(X), BATCH_SIZE), desc="Evaluating"):

        xb = X[i:i+BATCH_SIZE].to(device)
        yb = y[i:i+BATCH_SIZE].to(device)

        teacher_input = torch.cat([xb, yb[:, :-1]], dim=1)
        logits = model(teacher_input)
        target_logits = logits[:, CONTEXT_LEN - 1:, :]

        loss = loss_fn(
            target_logits.reshape(-1, target_logits.size(-1)),
            yb.reshape(-1)
        )

        batch_tokens = yb.numel()
        total_loss += loss.item() * batch_tokens

        rollout_context = xb.clone()
        rollout_preds = []

        for _ in range(TARGET_LEN):
            context_window = rollout_context[:, -CONTEXT_LEN:]
            rollout_logits = model(context_window)
            next_token = torch.argmax(
                rollout_logits[:, -1, :],
                dim=-1,
                keepdim=True
            )
            rollout_preds.append(next_token)
            rollout_context = torch.cat([rollout_context, next_token], dim=1)

        preds = torch.cat(rollout_preds, dim=1)
        per_example_correct = (preds == yb).sum(dim=1)

        correct += (preds == yb).sum().item()
        total_tokens += batch_tokens
        threshold_matches += (
            per_example_correct >= min(MIN_CORRECT_TOKENS, TARGET_LEN)
        ).sum().item()
        exact_matches += (preds == yb).all(dim=1).sum().item()

        if len(examples) < SHOW_EXAMPLES:
            batch_examples = min(SHOW_EXAMPLES - len(examples), xb.size(0))

            for j in range(batch_examples):
                examples.append(
                    (
                        decode_tokens(xb[j].tolist()),
                        decode_tokens(yb[j].tolist()),
                        decode_tokens(preds[j].tolist()),
                        per_example_correct[j].item(),
                    )
                )


# -------------------------------------------------
# Metrics
# -------------------------------------------------
accuracy = threshold_matches / len(X)

token_accuracy = correct / total_tokens
exact_match_accuracy = exact_matches / len(X)

perplexity = torch.exp(torch.tensor(total_loss / total_tokens))

unit_name = "words" if TOKEN_LEVEL == "word" else "characters"
threshold = min(MIN_CORRECT_TOKENS, TARGET_LEN)

print(f"\nTokenization Level: {TOKEN_LEVEL}")
print("Context Length:", CONTEXT_LEN)
print("Target Length:", TARGET_LEN)
print(f"Accuracy (>= {threshold} correct {unit_name}):", accuracy)
print("Token Accuracy:", token_accuracy)
print("Exact Match Accuracy:", exact_match_accuracy)
print("Perplexity:", perplexity.item())

if examples:
    print("\nSample Predictions:\n")

    for idx, (context, target, prediction, num_correct) in enumerate(examples, start=1):
        print(f"Example {idx}")
        print("Context:", context)
        print(f"Target : {target} ({num_correct}/{TARGET_LEN} correct)")
        print("Predicted:", prediction)
        print()
