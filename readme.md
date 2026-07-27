# GRIP-Transformer

GRIP-Transformer is an encoder-decoder Transformer designed to generate a candidate interacting non-coding RNA partner.

Given a source ncRNA sequence (`RNA1`), the model generates a candidate partner sequence (`RNA2`).

Because the task is one-to-many, the same `RNA1` may have several valid partners. In the fully unguided generation setting, each generated sequence is therefore evaluated against the complete set of known partners, denoted by `T(x)`. In the guided settings, where 30% or 50% of the target sequence is provided as a prefix, the generated sequence is evaluated against the corresponding paired target.

## Repository contents

The main model components are:

| File | Description |
|---|---|
| `train.py` | Model training and validation. |
| `data.py` | Dataset loading and preprocessing. |
| `model.py` | Transformer architecture and configuration. |
| `tokenizer.py` | RNA tokenization. |
| `pos_encoding.py` | Positional encoding. |
| `log_callback.py` | Generation monitoring during evaluation. |
| `utils.py` | Reproducibility utilities. |

The repository also includes scripts for:

- generation with 30% and 50% target prefixes;
- fixed-length generation with `len(z) = lmax`;
- top-k generation with reproducible settings;
- Smith-Waterman and BLAST6 evaluation;
- comparison with a length-matched random baseline;
- paired statistical testing;
- interaction-category heatmap generation.

## Requirements

- Python 3.9 or later
- PyTorch
- Hugging Face Transformers
- pandas
- NumPy
- matplotlib
- safetensors
- NCBI BLAST+

The BLAST+ installation must include `blastn` and `makeblastdb`.

## Installation

```bash
git clone https://github.com/AnacletoLAB/GRIP-Transformer.git
cd GRIP-Transformer

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA and PyTorch versions may need to be adapted to the local GPU and driver configuration.

## Fixed-length generation

For each source sequence `x`, let `T(x)` be the set of all known valid RNA2 partners.

The maximum partner length is defined as:

```text
lmax = max(len(x') for x' in T(x))
```

The model generates one sequence `z` for each distinct `RNA1`, with:

```text
len(z) = lmax
```

During fixed-length generation, the EOS/STOP probability is recorded but does not terminate or truncate the sequence.

Top-k generation is performed with:

```text
top_k = 2
temperature = 1.0
seed = 42
```

Run:

```bash
python generate_B4_only_lmax_topk2.py
```

## Evaluation

Each generated sequence is aligned against every valid partner in `T(x)`.

The reported scores are:

```text
R_SW     = exact Smith-Waterman matches / len(x')
R_BLAST6 = BLAST6 nident / len(x')
```

The denominator is always the complete length of the valid partner `x'`.

The best partner is selected independently for Smith-Waterman and BLAST6.

Run:

```bash
python evaluate_B4_only_lmax_R_smith_waterman_blast6.py
```

## Random baseline

A reproducible random baseline can be generated using RNA sequences composed of `A`, `C`, `G`, and `U`, matched to the same `lmax` used for the model-generated sequences.

```bash
python generate_random_lmax_control.py
```

The random sequences are evaluated with the same Smith-Waterman and BLAST6 pipeline.

The paired model-versus-random comparison is performed with:

```bash
python compare_B4_model_random_R_ttest.py
```

## Data and model files

The experiments require the corresponding datasets, partner maps, and trained model checkpoints.

Large datasets and model checkpoints may not be included directly in the public repository. Paths can be changed through the command-line arguments supported by the scripts.

## Citation

If you use this code, please cite the associated paper.

The complete citation and DOI will be added after publication.
