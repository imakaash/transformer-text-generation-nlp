import torch

from models.transformer import TransformerLanguageModel
from utils import clean_text, decode_word_tokens, split_word_tokens, word_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_PATH = "data/sherlock.txt"
PROMPTS = [
    "i say, watson,",
    "i had seen little of holmes lately. my",
    "<para> one night--it was on the twentieth",
    "i could not help laughing at the ease with which",
    "indeed, i should have thought a little more.",
    "i could not help laughing at the ease with which",
    "they are coiners on a large scale, and"
]
GENERATE_LENGTH = 5
MIN_CONTEXT_LEN = 5
MAX_CONTEXT_LEN = 10
TARGET_LEN = 5
MODEL_PATH = "word_transformer.pt"

text = clean_text(open(TEXT_PATH).read())
_, stoi, itos = word_tokenizer(text)

model = TransformerLanguageModel(
    len(stoi),
    max_len=MAX_CONTEXT_LEN + TARGET_LEN - 1,
    pad_token_id=stoi["<pad>"]
).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()


def decode_tokens(token_ids, keep_special_tokens=False):

    if keep_special_tokens:
        return " ".join(itos[token_id] for token_id in token_ids)

    return decode_word_tokens(token_ids, itos)


def generate_text(prompt, length=5, show_steps=True):

    words = [word for word in split_word_tokens(prompt.lower()) if word != "<para>"]
    tokens = [stoi.get(word, stoi["<unk>"]) for word in words]

    if not tokens:
        raise ValueError("Prompt does not contain any known tokens.")

    if len(tokens) < MIN_CONTEXT_LEN:
        raise ValueError(
            f"Prompt should contain at least {MIN_CONTEXT_LEN} words for this model."
        )

    generated = list(tokens)
    step_outputs = []

    for step_idx in range(length):

        context_tokens = generated[-MAX_CONTEXT_LEN:]
        padded_context = [stoi["<pad>"]] * (MAX_CONTEXT_LEN - len(context_tokens)) + context_tokens
        context = torch.tensor([padded_context]).to(device)

        logits = model(context)

        probs = torch.softmax(logits[0, -1], dim=0)

        next_token = torch.multinomial(probs, 1).item()

        step_outputs.append(
            {
                "step": step_idx + 1,
                "model_input": padded_context,
                "predicted_token": next_token,
            }
        )
        generated.append(next_token)

    if show_steps:
        for step in step_outputs:
            print(f"Step {step['step']}")
            print("Input to model:", decode_tokens(step["model_input"], keep_special_tokens=True))
            print("Predicted output:", itos[step["predicted_token"]])
            print()

    return decode_word_tokens(generated, itos)


print("\nTokenization Level: word")

for example_idx, prompt in enumerate(PROMPTS, start=1):
    print(f"\nExample {example_idx}")
    print("Prompt:", prompt)
    print()

    output = generate_text(prompt, GENERATE_LENGTH, show_steps=True)

    print("Final Output:")
    print(output)
