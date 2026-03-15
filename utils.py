import re
import torch
from collections import Counter


def remove_gutenberg_header_footer(text):

    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"

    start = text.find(start_marker)
    end = text.find(end_marker)

    if start != -1 and end != -1:
        text = text[start:end]

    return text


def clean_text(text):

    text = remove_gutenberg_header_footer(text)

    # remove table of contents lines
    text = re.sub(r'\n\s*[IVX]+\.\s+[^\n]+', '', text)

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r'http\S+', '', text)

    # remove digits
    text = re.sub(r'\d+', '', text)

    # remove punctuation except sentence markers
    text = re.sub(r'[^\w\s\.\,\?\!]', '', text)

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def remove_rare_words(words, min_freq=3):

    freq = Counter(words)

    words = [w for w in words if freq[w] >= min_freq]

    return words


def word_tokenizer(text):

    words = re.findall(r"\b\w+\b", text)

    words = remove_rare_words(words)

    vocab = sorted(set(words))

    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}

    encoded = [stoi[w] for w in words]

    return encoded, stoi, itos


def char_tokenizer(text):

    chars = sorted(list(set(text)))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    encoded = [stoi[c] for c in text]

    return encoded, stoi, itos


def create_sequences(data, seq_len):

    xs = []
    ys = []

    for i in range(len(data) - seq_len):

        xs.append(data[i:i + seq_len])
        ys.append(data[i + 1:i + seq_len + 1])

    return torch.tensor(xs), torch.tensor(ys)