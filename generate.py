import torch
from models.transformer import TransformerLanguageModel
from utils import char_tokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"

text = open("data/sherlock.txt").read()

encoded, stoi, itos = char_tokenizer(text)

model = TransformerLanguageModel(len(stoi)).to(device)

model.load_state_dict(torch.load("char_transformer.pt"))
model.eval()

context = "to sherlock holmes"

context = torch.tensor([[stoi[c] for c in context]])

for _ in range(200):

    logits = model(context.to(device))

    probs = torch.softmax(logits[0,-1], dim=0)

    next_token = torch.multinomial(probs,1).item()

    context = torch.cat([context, torch.tensor([[next_token]])], dim=1)

generated = "".join([itos[i] for i in context[0].tolist()])

print(generated)