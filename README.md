# Transformer Text Generation

This repository contains a custom PyTorch Transformer language model trained on *The Adventures of Sherlock Holmes*. The project is now intentionally optimized for one setup only:

- word-level next-word prediction
- short sentence-like prompts of `5-10` words
- evaluation on random chunks from the cleaned Sherlock text after full-data training

The implementation uses a causal Transformer encoder in PyTorch rather than a pretrained GPT model.

## Current Task Setup

The training pipeline creates explicit `(X, y)` pairs from sentence-local windows in the Sherlock corpus:

- `X`: a left-padded short context window
- `y`: the next `5` words to predict

Current defaults:

- minimum context length: `5` words
- maximum context length: `10` words
- target length: `5` words
- stride: `1`

During training, the target chunk is learned with teacher forcing across the full cleaned dataset. During generation and evaluation, the model rolls predictions out autoregressively.

## Folder Structure

```text
transformer-text-generation-nlp/
├── data/                          # local dataset folder, ignored by git
│   ├── sherlock.txt
│   └── sherlock_cleaned.txt
├── models/
│   └── transformer.py
├── notebooks/
│   └── text_generation_demo.ipynb
├── train_word.py
├── evaluate.py
├── generate.py
├── utils.py
├── requirements.txt
├── README.md
├── LICENSE
├── word_transformer.pt            # generated after training
└── training_loss_word.png         # generated after training
```

## Dataset

Dataset used:

- *The Adventures of Sherlock Holmes*
- approximately 105,000 words
- literary prose with recurring style and phrase patterns

Expected raw input:

```text
data/sherlock.txt
```

The scripts clean and normalize the text, then write:

```text
data/sherlock_cleaned.txt
```

The cleaning pipeline in `utils.py`:

- removes Gutenberg header/footer
- trims front matter
- removes chapter headings
- lowercases text
- preserves sentence punctuation
- inserts `<para>` markers for paragraph breaks

## Installation

```bash
git clone https://github.com/imakaash/transformer-text-generation.git
cd transformer-text-generation
pip install -r requirements.txt
```

Core dependencies used by the custom pipeline:

- `torch`
- `tqdm`
- `numpy`
- `matplotlib`

## Model

The model is defined in `models/transformer.py`.

Architecture:

- token embedding with padding support
- positional embedding
- stacked `nn.TransformerEncoderLayer` blocks
- causal attention mask
- key-padding mask for short left-padded prompts
- linear output layer with weight tying

Default hyperparameters:

- embedding size: `384`
- attention heads: `8`
- transformer layers: `6`
- feedforward size: `1024`

## Training

Run:

```bash
python train_word.py
```

Current training behavior:

- reads `data/sherlock.txt`
- cleans and writes `data/sherlock_cleaned.txt`
- tokenizes at the word level only
- builds sentence-local `(context, target)` pairs
- learns from short prompts between `5` and `10` words
- left-pads shorter contexts to a fixed width of `10`
- trains on all available pairs from the cleaned Sherlock text
- trains for `25` epochs with batch size `64`
- optimizes with `AdamW`, gradient clipping, and cosine learning-rate decay
- saves the latest checkpoint each epoch
- records only training loss because there is no validation split

Outputs:

```text
word_transformer.pt
training_loss_word.png
```

## Evaluation

Run:

```bash
python evaluate.py
```

Current evaluation behavior:

- rebuilds the same Sherlock cleaning pipeline from `data/sherlock.txt`
- samples a random contiguous chunk from the cleaned Sherlock text each run
- reconstructs sentence-local short-context pairs only from that chunk
- rolls out predictions for the 5-word target chunk autoregressively
- uses multinomial sampling during rollout, matching `generate.py`
- blocks special tokens such as `<pad>`, `<para>`, and `<unk>` during prediction
- computes order-independent token overlap instead of strict position-by-position accuracy
- ignores `<para>` and `<pad>` when scoring matches
- prints `20` sample predictions by default
- shows raw predicted tokens, matched tokens, wrong predicted tokens, and missing target tokens
- reports token-level accuracy and perplexity

Reported metrics:

- `Total Correct Tokens (unordered)`
- `Order-Independent Token Accuracy`
- `Perplexity`

## Generation

Run:

```bash
python generate.py
```

Generation behavior:

- runs multiple built-in prompt examples
- tokenizes each prompt using the same word tokenizer as training
- expects each prompt to contain at least `5` words
- left-pads prompts shorter than the internal max width
- predicts one word at a time autoregressively
- uses multinomial sampling from the model distribution
- prints the exact input passed to the model at every generation step
- prints the predicted output token at every step
- prints the final generated continuation for each example

Current example prompts:

```text
to sherlock holmes she is always the woman. i
i had seen little of holmes lately. my
<para> one night--it was on the twentieth
i could not help laughing at the ease with which
indeed, i should have thought a little more.
they are coiners on a large scale, and
```
