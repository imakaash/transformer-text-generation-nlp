import re
import torch


WORD_PATTERN = r"<pad>|<unk>|<para>|[a-zA-Z'-]+|[.,!?]"
SPECIAL_TOKENS = ["<pad>", "<unk>"]


def remove_gutenberg_header_footer(text):
    """
    Remove Project Gutenberg metadata.
    """
    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"

    start = text.find(start_marker)
    end = text.find(end_marker)

    if start != -1 and end != -1:
        text = text[start + len(start_marker):end]

    return text


def remove_front_matter(text):
    """
    Remove remaining book front matter like titles, author lines,
    and 'Produced by' lines before the first real sentence.
    """

    # anchor on the first narrative sentence
    first_sentence = "To Sherlock Holmes she is always"

    idx = text.find(first_sentence)

    if idx != -1:
        text = text[idx:]

    return text


def remove_chapter_titles(text):
    """
    Remove roman numeral headings and adventure titles.
    """

    text = re.sub(r'\n\s*[IVX]+\.\s+[^\n]+', '\n', text)

    text = re.sub(r'ADVENTURE\s+[IVX]+\.\s+[A-Z\s]+', '', text)

    text = re.sub(r'\n\s*[IVX]+\.\s*\n', '\n', text)

    return text


def clean_text(text):

    text = remove_gutenberg_header_footer(text)

    text = remove_front_matter(text)

    text = remove_chapter_titles(text)

    text = text.replace("\r\n", "\n")

    # paragraph token
    text = re.sub(r'\n\s*\n+', ' <para> ', text)

    # remove single newlines
    text = re.sub(r'\n+', ' ', text)

    text = text.lower()

    # remove urls
    text = re.sub(r'http\S+', '', text)

    # keep punctuation + apostrophes
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\?\!\'<>\-]", "", text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def word_tokenizer(text):

    words = split_word_tokens(text)

    vocab = SPECIAL_TOKENS + sorted(set(words) - set(SPECIAL_TOKENS))

    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}

    encoded = encode_word_tokens(words, stoi)

    return encoded, stoi, itos


def split_word_tokens(text):

    normalized_text = text.replace("<PARA>", "<para>")

    return re.findall(WORD_PATTERN, normalized_text)


def encode_word_tokens(words, stoi):

    unk_id = stoi["<unk>"]

    return [stoi.get(word, unk_id) for word in words]


def split_into_sentences(text):

    normalized_text = re.sub(r"\s*<para>\s*", " ", text.replace("<PARA>", "<para>"))
    sentence_candidates = re.split(r"(?<=[.!?])\s+", normalized_text)
    sentences = []

    for candidate in sentence_candidates:
        tokens = [token for token in split_word_tokens(candidate) if token != "<para>"]

        if tokens:
            sentences.append(tokens)

    return sentences


def char_tokenizer(text):

    chars = sorted(list(set(text)))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    encoded = [stoi[c] for c in text]

    return encoded, stoi, itos


def create_sequences(data, seq_len, stride=1):

    xs = []
    ys = []

    for i in range(0, len(data) - seq_len, stride):

        xs.append(data[i:i + seq_len])
        ys.append(data[i + 1:i + seq_len + 1])

    return torch.tensor(xs), torch.tensor(ys)


def create_future_prediction_sequences(data, input_len, target_len, stride=1):

    xs = []
    ys = []

    max_start = len(data) - input_len - target_len + 1

    for i in range(0, max_start, stride):

        xs.append(data[i:i + input_len])
        ys.append(data[i + input_len:i + input_len + target_len])

    return torch.tensor(xs), torch.tensor(ys)


def create_sentence_prediction_sequences(
    text,
    stoi,
    min_context_len,
    max_context_len,
    target_len,
    stride=1
):

    pad_id = stoi["<pad>"]
    xs = []
    ys = []

    for sentence in split_into_sentences(text):
        encoded_sentence = encode_word_tokens(sentence, stoi)

        if len(encoded_sentence) < min_context_len + target_len:
            continue

        max_end = len(encoded_sentence) - target_len

        for context_len in range(min_context_len, max_context_len + 1):
            for end_idx in range(context_len, max_end + 1, stride):
                context = encoded_sentence[end_idx - context_len:end_idx]
                target = encoded_sentence[end_idx:end_idx + target_len]
                padded_context = [pad_id] * (max_context_len - context_len) + context

                xs.append(padded_context)
                ys.append(target)

    return torch.tensor(xs), torch.tensor(ys)


def decode_word_tokens(token_ids, itos):

    words = [itos[idx] for idx in token_ids]
    words = [word for word in words if word not in {"<pad>", "<unk>"}]

    text = " ".join(words).replace("<para>", "\n\n")

    return re.sub(r"\s+([.,!?])", r"\1", text).strip()


def filter_special_tokens(token_ids, itos, ignored_tokens=None):

    ignored = set(ignored_tokens or [])

    return [
        token_id
        for token_id in token_ids
        if itos[token_id] not in ignored
    ]
