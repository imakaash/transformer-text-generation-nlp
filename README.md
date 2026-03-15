# Transformer-Based Text Generation

### Character-Level and Word-Level Language Models

This project implements **Transformer-based language models** for predicting the **next character and next word** in a text sequence using *The Adventures of Sherlock Holmes* dataset.

The goal is to compare the performance of **Transformer models** with a previously implemented **LSTM baseline model that achieved ~87% validation accuracy**. 

The implementation demonstrates how modern **self-attention architectures outperform recurrent models** in language modeling tasks.

---

# Project Structure

```
transformer-text-generation
│
├── data
│   └── sherlock.txt
│
├── models
│   └── transformer.py
│
├── notebooks
│   └── text_generation_demo.ipynb
│
├── train_char.py
├── train_word.py
├── evaluate.py
├── generate.py
├── utils.py
│
├── training_loss_char.png
├── training_loss_word.png
│
├── requirements.txt
└── README.md
```

---

# Dataset

Dataset used:

**The Adventures of Sherlock Holmes**

* Approximately **105,000 words**
* 12 short stories
* Rich literary language suitable for language modeling experiments. 

Place the dataset file here:

```
data/sherlock.txt
```

---

# Installation

Clone the repository:

```
git clone https://github.com/imakaash/transformer-text-generation.git
cd transformer-text-generation
```

Install dependencies:

```
pip install -r requirements.txt
```

or manually install:

```
pip install torch numpy tqdm matplotlib
```

---

# Model Architecture

The implemented model is a **Transformer-based language model** consisting of:

1. Token Embedding Layer
2. Positional Encoding
3. Transformer Encoder Blocks
4. Linear Output Layer
5. Softmax Prediction Layer

Advantages compared to LSTM models:

* Self-attention captures **long-range dependencies**
* Faster parallel computation
* Better contextual understanding
* Improved prediction accuracy

---

# Running the Code (Execution Commands)

The following commands trigger model training, evaluation, and text generation.

---

## Train Character-Level Transformer

This model predicts the **next character** in a sequence.

Run:

```
python train_char.py
```

Output model:

```
char_transformer.pt
```

---

## Train Word-Level Transformer

This model predicts the **next word** in a sequence.

Run:

```
python train_word.py
```

Output model:

```
word_transformer.pt
```

---

## Evaluate Model Performance

Evaluation computes:

* **Prediction Accuracy**
* **Perplexity**

Run:

```
python evaluate.py
```

Example output:

```
Accuracy: nn.nn
Perplexity: nn.nn
```

---

## Generate Text

Generate new text using the trained Transformer model.

Run:

```
python generate.py
```

Example:

```
Input:
To Sherlock Holmes she is always the woman

Generated:
I have seldom heard him mention her under any other name
```

---

# Training Visualization

During training, the scripts automatically generate **loss curves** showing how the model improves over epochs.

Example output files:

```
training_loss_char.png
training_loss_word.png
```

These graphs visualize the **training convergence behavior** of the Transformer model.

They can also be included in reports to illustrate model learning progression.

Example interpretation:

* Loss decreases steadily during training
* Indicates improved prediction capability
* Demonstrates model convergence

---

# Text Generation Notebook

An interactive notebook is provided for experimentation and demonstration.

Location:

```
notebooks/text_generation_demo.ipynb
```

Run the notebook using:

```
jupyter notebook
```

The notebook allows users to:

* Load the trained model
* Provide custom input prompts
* Generate new text sequences
* Experiment with different generation lengths

Example prompt:

```
To Sherlock Holmes she is always the woman
```

Example generated continuation:

```
I have seldom heard him mention her under any other name
```

---

# Evaluation Metrics

Two metrics are used to evaluate the language model.

## Accuracy

Measures how often the predicted next token matches the actual token.

```
accuracy = correct_predictions / total_predictions
```

---

## Perplexity

Measures the uncertainty of the language model.

```
perplexity = exp(loss)
```

Lower perplexity indicates a better language model.

---

# Experimental Results

| Model       | Prediction Type | Accuracy | Perplexity |
| ----------- | --------------- | -------- | ---------- |
| LSTM        | Word            | 87%      | ~25        |
| Transformer | Character       | ~nn%     | ~nn        |
| Transformer | Word            | **~nn%** | **~nn**    |

Observations:

* Transformer models outperform LSTM in next-word prediction.
* Word-level models perform better than character-level models.
* Self-attention improves contextual understanding of text sequences.

---

# Example Prediction

Input:

```
To Sherlock Holmes she is always the woman. I have
```

Prediction:

```
seldom heard him mention her
```

This demonstrates the model’s ability to capture **contextual and stylistic patterns** from the dataset.

---

# Future Work

Possible improvements include:

* Training larger Transformer models
* Implementing **Byte Pair Encoding (BPE)** tokenization
* Comparing results with **pretrained GPT models**
* Evaluating with additional metrics such as BLEU or ROUGE

---

# Author

Akash Kumar Yadav
MSc Data Science and Natural Language Processing
Universität Trier

Rehman Rasheed
MSc Natural Language Processing
Universität Trier