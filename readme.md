# GRIP: Generative non coding RNA Interaction Predictor

A full transformer with multiple encoder decoder modules able to generatively predict a target non-coding RNA (ncRNA) interactor having as input only the sequence of a source ncRNA interactor. It has been trained from scratch using the ncRNA-ncRNA interactions available form the RNA-KG repository.

---

## 1. Setup

### Create conda environment
conda create -n rna_full_transformer python=3.9 -y
conda activate rna_full_transformer

### Install dependencies
pip install -r requirements.txt

## 2. Data Preparation

We assume your data for both training and validation is stored in single TSV files (`train.tsv`, `valid.tsv`), where each line contains a source and target sequence separated by a tab (`	`), for example:

```
AUGCUA...	UAGCGA...
GCGUA...	CGCAU...
```

Each sequence is a contiguous string of `A,U,C,G`; the `STOP` token is appended implicitly during tokenization.

## 3. Tutorial (TO DO)
This tutorial shows how to train a transformer encoder–decoder from scratch on pairs of nucleotide sequences (interacting RNA couples). We use 8 tokens: `A`, `U`, `C`, `G`, `PAD`, `EOS`, `UNK`, `BOS`. The model’s number of heads, layers and hidden size are configurable. We add standard sinusoidal positional encoding. We train and validate via the Hugging Face `Trainer` API.

```
rna_transformer/
├── data.py          # loading & preparing Dataset objects
├── tokenizer.py     # vocab definition & Fast tokenizer
├── pos_encoding.py  # sinusoidal positional encoding
├── model.py         # NucConfig + NucTransformer classes
├── train.py         # main() entrypoint: Trainer setup & run
└── utils.py         # seed setting
```
