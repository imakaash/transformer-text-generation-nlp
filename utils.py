import torch


def char_tokenizer(text):

    chars = sorted(list(set(text)))

    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}

    encoded = [stoi[c] for c in text]

    return encoded, stoi, itos


def word_tokenizer(text):

    words = text.split()

    vocab = sorted(set(words))

    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {i:w for w,i in stoi.items()}

    encoded = [stoi[w] for w in words]

    return encoded, stoi, itos


def create_sequences(data, seq_len):

    xs = []
    ys = []

    for i in range(len(data)-seq_len):

        xs.append(data[i:i+seq_len])
        ys.append(data[i+1:i+seq_len+1])

    return torch.tensor(xs), torch.tensor(ys)