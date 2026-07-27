# GRIP-Transformer: ncRNA Partner Generation

Given a source RNA sequence, `RNA1`, the model generates a candidate interacting
partner, `RNA2`. The task is one-to-many: the same `RNA1` may have several valid
partners. Consequently, generated sequences are evaluated against the complete
valid partner set, denoted by `T(x)`, rather than against only one paired target.

## Scientific objective

The objective of this work is to generate an interacting ncRNA partner conditioned on a given target ncRNA sequence.

Given an input ncRNA (`RNA1`), the model generates a candidate interacting ncRNA partner (`RNA2`).


## Main scripts

| Script | Description |
|---|---|
| `final_generation_30.py` | Generates the remaining RNA2 sequence after providing the first 30% of the true target as an oracle prefix. |
| `final_generation_50.py` | Generates the remaining RNA2 sequence after providing the first 50% of the true target as an oracle prefix. |
| `evaluate_B4_only_generated_against_t2.py` | Evaluates historical non-`lmax` B4 generations against all valid partners for the corresponding RNA1 using Smith-Waterman, BLAST4, and BLAST6. |
| `generate_B4_only_lmax.py` | Performs greedy B4 1-mer generation while enforcing the exact maximum valid-partner length, `len(z) = lmax`. |
| `generate_B4_only_lmax_topk2.py` | Performs reproducible fixed-length generation with `top_k=2`, temperature 1.0, and seed 42. |
| `evaluate_B4_only_lmax_R_smith_waterman_blast6.py` | Evaluates each fixed-length generation against every valid partner using Smith-Waterman and BLAST6. |
| `generate_random_lmax_control.py` | Generates reproducible uniformly random A/C/G/U sequences matched to the model-generated `lmax` for every RNA1. |
| `compare_B4_model_random_R_ttest.py` | Performs paired model-versus-random statistical comparisons for Smith-Waterman and BLAST6 scores. |
| `heatmap_interaction_types_nonaug_pkl.py` | Produces the RNA1-by-RNA2 interaction-category heatmap and the underlying count tables. |

## GRIP-Transformer runtime

The generation scripts require the original GRIP-Transformer runtime:

| File | Role |
|---|---|
| `model.py` | Historical `NucTransformer` architecture. |
| `tokenizer.py` | 1-mer tokenizer with the vocabulary `PAD`, `EOS`, `UNK`, `A`, `U`, `C`, and `G`. |
| `pos_encoding.py` | Positional encoding used by the Transformer. |
| `requirements.txt` | Python environment used for the original experiments. |


## Requirements

- Python 3.9 or later
- PyTorch
- Hugging Face Transformers
- pandas
- NumPy
- matplotlib
- safetensors
- NCBI BLAST+, including `blastn` and `makeblastdb`

The original server environment is recorded in `requirements.txt`. CUDA and
PyTorch packages may need to be adapted to the local GPU and driver version.

Example installation:

```bash
git clone https://github.com/AnacletoLAB/GRIP-Transformer.git
cd GRIP-Transformer
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Required input files

The principal experiments expect the following files:

| File | Description |
|---|---|
| `data/dry_run/test_filtered_no_aug_pident95_qcov90.tsv` | Historical B4 test set after the sequence-similarity leakage filter. It contains 697 distinct RNA1 sequences. |
| `rna1_to_t2_targets_train+test.tsv` | Grouped map from each RNA1 to the complete set of known valid RNA2 partners used to construct `T(x)` and calculate `lmax`. |
| `train_df_no_augmentation_all.pkl` | Non-augmented training pairs used for the interaction-category heatmap. |
| `test_df_no_augmentation_all.pkl` | Non-augmented test pairs used for the interaction-category heatmap. |
| `ckpt/dry_run/final_model/` | Historical B4 checkpoint directory containing the model configuration and weights. |

Large datasets and model checkpoints are not necessarily distributed through
the Git repository. Their paths can be changed through the command-line
arguments shown below.

## Fixed-length generation protocol

For each test `RNA1`, the scripts recover all valid partners:

```text
T(x) = {x'_1, x'_2, ..., x'_n}
```

The maximum valid-partner length is:

```text
lmax = max(len(x') for x' in T(x))
```

The model then generates exactly one sequence `z` per distinct `RNA1`, with:

```text
len(z) = lmax
```

EOS/STOP is monitored but does not terminate fixed-length generation.

### Top-k generation

```bash
python generate_B4_only_lmax_topk2.py 
```

This script uses the fixed experimental settings:

```text
top_k = 2
temperature = 1.0
seed = 42
```

Only the nucleotide tokens A, U, C, and G can be sampled. PAD, EOS, and UNK
cannot become generated nucleotides.

## Evaluation against the complete partner set

Each generated sequence is aligned against every valid `x'` in `T(x)`.

The reported ratios are:

```text
R_SW     = exact Smith-Waterman matches / len(x')
R_BLAST6 = BLAST6 nident / len(x')
```

The denominator is always the full length of the selected valid partner, not
the local alignment length. The best target is selected independently for
Smith-Waterman and BLAST6.

### Evaluate top-k fixed-length generations

```bash
python evaluate_B4_only_lmax_R_smith_waterman_blast6.py
```

## Length-matched random control

Generate one uniformly random RNA sequence with the same `lmax` as each
model-generated sequence:

```bash
python generate_random_lmax_control.py
```

Evaluate the random sequences with exactly the same alignment pipeline:

```bash
python evaluate_B4_only_lmax_R_smith_waterman_blast6.py
```

Run the paired statistical comparison:

```bash
python compare_B4_model_random_R_ttest.py
```

The script verifies the required columns, missing values, the consistency of
`Category_Couple`, and the total number of train and test pairs before producing
the figure.

## Main results

In the paired model-versus-random comparison:

- Smith-Waterman produced a mean paired difference of `-0.018864`
  (`p = 2.234e-09`), with the model scoring below the random control.
- BLAST6, with no-hit cases encoded as `R = 0`, produced a mean paired
  difference of `+0.007044` (`p = 0.150130`), which was not statistically
  significant.

These results show that top-k sampling largely removes exact sequence collapse,
but the alignment metrics do not establish robust RNA1-specific generation.
The strong Smith-Waterman score of the random control also demonstrates that
local alignment followed by best-target selection can produce optimistic
similarity values.

## Interpretation cautions

- A high number of distinct sequences is not sufficient evidence of
  conditioning on `RNA1`.
- Smith-Waterman always returns a local optimum and can assign substantial
  scores to random sequences.
- BLAST6 is more selective and may return no hit.
- Local alignment identity must not be interpreted as full-sequence identity.
- The 30% and 50% generation conditions contain true target prefixes and are
  oracle controls, not de novo generation.
- The reported `R` values quantify the best local exact-match signal normalized
  by the full selected-target length.

## Current code-organization note

The current versions of:

```text
evaluate_B4_only_lmax_R_smith_waterman_blast6.py
generate_random_lmax_control.py
```

import shared helper functions from `evaluate_generated_against_t2.py`. For a
fully standalone B4-only release, these helpers should be moved to a neutral
module such as `b4_alignment_utils.py`, or imported from
`evaluate_B4_only_generated_against_t2.py`.

## Reproducibility

- B4 tokenization: 1-mer
- Vocabulary size: 7
- Top-k sampling seed: 42
- Top-k: 2
- Sampling temperature: 1.0
- BLAST word size: 6 for the fixed-length `R` evaluation
- BLAST maximum HSPs per target: 1
- BLAST no-hit cases: retained in row-level outputs and excluded from
  hit-conditioned aggregate means

## Citation

If you use this code, please cite the associated paper. The complete citation
and DOI will be added after publication.
