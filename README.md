# Transformer Text Generation

This repository contains a custom PyTorch Transformer language model trained on *The Adventures of Sherlock Holmes*. The current project focus is:

- word-level prediction: given a context window, predict the next few words
- character-level prediction: given a context window, predict the next few characters
- text generation in the learned Sherlock Holmes style

The implementation uses a causal Transformer encoder in PyTorch rather than a pretrained GPT model.

## Current Task Setup

The repository no longer trains on a simple full-sequence next-token objective only. The active training setup creates explicit `(X, y)` pairs:

- `X`: the context window
- `y`: the next few tokens to predict

Current defaults:

- word model: `60` context tokens -> predict the next `5` words
- character model: `120` context characters -> predict the next `20` characters

During training, the target chunk is learned with teacher forcing. During generation and evaluation, the model rolls predictions out autoregressively.

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
├── train_char.py
├── evaluate.py
├── generate.py
├── utils.py
├── requirements.txt
├── README.md
├── LICENSE
├── word_transformer.pt            # generated after training
├── char_transformer.pt            # generated after training
├── training_loss_word.png         # generated after training
└── training_loss_char.png         # generated after training
```

The `data/` folder is used locally for the raw and cleaned corpus and is ignored by git.

## Dataset

Dataset used:

- *The Adventures of Sherlock Holmes*
- approximately 105,000 words
- rich literary prose suited to language modeling and style learning

Expected raw input:

```text
data/sherlock.txt
```

The training scripts clean and normalize the text, then write:

```text
data/sherlock_cleaned.txt
```

The cleaning pipeline in `utils.py` currently:

- removes Gutenberg header/footer
- trims front matter
- removes chapter headings
- lowercases text
- preserves sentence punctuation
- inserts `<para>` markers for paragraph breaks

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/imakaash/transformer-text-generation.git
cd transformer-text-generation
pip install -r requirements.txt
```

Current `requirements.txt` includes:

- `torch`
- `transformers`
- `datasets`
- `accelerate>=1.1.0`
- `tqdm`
- `numpy`
- `sentencepiece`
- `matplotlib`

The core custom Transformer pipeline only depends on `torch`, `tqdm`, `numpy`, and `matplotlib`. The Hugging Face packages are present for experimental work in this environment.

## Model

The model is defined in `models/transformer.py`.

Architecture:

- token embedding
- positional embedding
- stacked `nn.TransformerEncoderLayer` blocks
- causal attention mask
- linear output layer with weight tying

Default model hyperparameters:

- embedding size: `256`
- attention heads: `8`
- transformer layers: `4`
- feedforward size: `512`

Although the implementation uses `TransformerEncoder`, it behaves autoregressively because of the causal mask.

## Training

### Word-Level Model

Run:

```bash
python train_word.py
```

Current word-level training behavior:

- reads `data/sherlock.txt`
- cleans and writes `data/sherlock_cleaned.txt`
- tokenizes at the word level
- builds `(context, target)` pairs with:
  - `CONTEXT_LEN = 60`
  - `TARGET_LEN = 5`
  - `STRIDE = 2`
- trains for `10` epochs with batch size `64`

Outputs:

```text
word_transformer.pt
training_loss_word.png
```

### Character-Level Model

Run:

```bash
python train_char.py
```

Current character-level training behavior:

- reads `data/sherlock.txt`
- cleans and writes `data/sherlock_cleaned.txt`
- tokenizes at the character level
- builds `(context, target)` pairs with:
  - `CONTEXT_LEN = 120`
  - `TARGET_LEN = 20`
  - `STRIDE = 3`
- trains for `10` epochs with batch size `64`

Outputs:

```text
char_transformer.pt
training_loss_char.png
```

## Evaluation

Run:

```bash
python evaluate.py
```

Set `TOKEN_LEVEL` inside `evaluate.py` to choose:

- `"word"`
- `"char"`

Current evaluation behavior:

- rebuilds context-target pairs from `data/sherlock_cleaned.txt`
- evaluates on the last `10%` of those pairs
- uses rollout prediction for the target chunk
- reports perplexity and multiple accuracy views

Reported metrics:

- `Accuracy (>= 3 correct words)` for word-level comparison
- `Token Accuracy`
- `Exact Match Accuracy`
- `Perplexity`

For the comparison-oriented word-level metric, a prediction is counted as correct if at least `3` out of `5` target words are predicted in the correct positions.

Example:

```text
Input: "to sherlock holmes she is always the woman . i have"
Target: "seldom heard him mention her"
Prediction: "seldom heard him mention her" -> 5/5 correct
```

Another valid case under the same metric:

```text
Prediction: "seldom heard him knew her" -> 3/5 correct
```

This matches the comparison strategy where `3` or more correct predicted words counts as a correct sequence.

Important note:

- `evaluate.py` uses an evaluation slice from the cleaned corpus
- `train_word.py` and `train_char.py` currently train on all generated pairs

So this evaluation is useful for comparison and monitoring, but it is not yet a strict fully held-out train/test split.

## Generation

Run:

```bash
python generate.py
```

Set `TOKEN_LEVEL` in `generate.py` to choose word-level or character-level generation.

Generation behavior:

- tokenizes the prompt using the same tokenizer as training
- keeps a sliding context window
- predicts one token at a time
- appends each new prediction autoregressively

Default example prompt:

```text
the drowsiness of the drug
```

For word generation, `<para>` tokens are converted back into paragraph breaks in the final output.

## Utility Functions

`utils.py` currently provides:

- text cleaning helpers
- word tokenizer
- character tokenizer
- word token splitting for prompt handling
- `create_future_prediction_sequences(...)` for building `(X, y)` pairs

This function is the core of the current training task:

```python
X = context window
y = next few tokens to predict
```

## Current Status

The repository is currently aligned around the custom Transformer pipeline:

- training scripts for word and character prediction are active
- evaluation matches the current multi-token prediction task
- generation matches the same context-window setup
- README now reflects the new workflow

Artifacts such as model weights, plots, and local data are generated during runs and may not all be tracked in git.

## Suggested Workflow

1. Place the raw corpus in `data/sherlock.txt`
2. Train a model:

```bash
python train_word.py
```

or

```bash
python train_char.py
```

3. Evaluate:

```bash
python evaluate.py
```

4. Generate text:

```bash
python generate.py
```

## Future Improvements

Possible next steps:

- add a true train/validation/test split
- save token vocab metadata explicitly with checkpoints
- compare word and character models on the same held-out examples
- add top-k or top-p sampling for generation
- log evaluation metrics after each epoch during training

## Author

Akash Kumar Yadav
MSc Data Science and Natural Language Processing
Universitat Trier

Rehman Rasheed
MSc Natural Language Processing
Universitat Trier

## License

This project includes the `LICENSE` file in the repository root.
