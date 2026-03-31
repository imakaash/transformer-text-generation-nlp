# Transformer Text Generation

This repository contains a custom PyTorch Transformer language model trained on *The Adventures of Sherlock Holmes*. The project is now intentionally optimized for one setup only:

- word-level next-word prediction
- short sentence-like prompts of `5-10` words
- training with explicit train/validation/test splits from the cleaned Sherlock text

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

Dataset split defaults:

- train: `80%`
- validation: `10%`
- test: `10%`

## Folder Structure

```text
transformer-text-generation-nlp/
├── data/                          # local dataset folder, ignored by git
│   ├── sherlock.txt
│   └── sherlock_cleaned.txt
├── models/
│   └── transformer.py
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
- shuffles and splits the pairs into train, validation, and test sets
- learns from short prompts between `5` and `10` words
- left-pads shorter contexts to a fixed width of `10`
- trains only on the training split
- trains for `25` epochs with batch size `64`
- optimizes with `AdamW`, gradient clipping, and cosine learning-rate decay
- computes validation loss after every epoch
- saves the best checkpoint based on validation loss
- keeps the training-loss plot output in `training_loss_word.png`

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
- reconstructs the same sentence-local prediction pairs used in training
- rebuilds the same deterministic `80/10/10` split and evaluates only on the held-out test set
- rolls out predictions for the 5-word target chunk autoregressively from test contexts
- uses deterministic next-token selection during evaluation
- computes order-independent token overlap instead of strict position-by-position accuracy
- ignores `<para>` and `<pad>` when scoring matches
- prints sample predictions from the test set
- reports test loss, test perplexity, and order-independent test accuracy

Reported metrics:

- `Training Loss`
- `Total Matches`
- `Test Accuracy`

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
    "putty , and he glared at the envelope which he",
    "' you may imagine , mr . holmes ,",
    "which will always secure me from any steps which he",
    "no difficulty in engaging a bedroom and sitting-room at the",
    "upon four before the door opened , and a drunken-looking",
    "basket-chair . this is my friend and colleague , dr",
    ", so it was a close thing , but we",
    "frock-coat , unbuttoned in the front , and a drab",
    "come , and the billet was such a good one",
    ". it must be done at once . you must"
```
