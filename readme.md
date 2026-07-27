# GRIP-Transformer: B4 1-mer RNA Partner Generation

This repository contains the generation, evaluation, and control pipelines used
in **GRIP-Transformer** to study the historical **B4 1-mer Transformer** for RNA
partner generation.

Given a source RNA sequence, `RNA1`, the model generates a candidate interacting
partner, `RNA2`. The task is one-to-many: the same `RNA1` may have several valid
partners. Consequently, generated sequences are evaluated against the complete
valid partner set, denoted by `T(x)`, rather than against only one paired target.

This repository is limited to the historical B4 1-mer model and the analyses
used for the associated paper. It does not contain the later B6 or B7 3-mer
training pipelines.

## Scientific objectives

The code is designed to investigate the following questions:

Conditioned generation of an interacting ncRNA partner given a target ncRNA sequence.

Sequence diversity alone is not interpreted as evidence of robust conditioning
on `RNA1`.

## Main scripts

| Script | Description |
|---|---|
| `final_generation_0.py` | Generates complete RNA2 sequences without providing any target prefix. |
| `final_generation_30.py` | Generates the remaining RNA2 sequence after providing the first 30% of the true target as an oracle prefix. |
| `final_generation_50.py` | Generates the remaining RNA2 sequence after providing the first 50% of the true target as an oracle prefix. |
| `evaluate_B4_only_generated_against_t2.py` | Evaluates historical non-`lmax` B4 generations against all valid partners for the corresponding RNA1 using Smith-Waterman, BLAST4, and BLAST6. |
| `generate_B4_only_lmax.py` | Performs greedy B4 1-mer generation while enforcing the exact maximum valid-partner length, `len(z) = lmax`. |
| `generate_B4_only_lmax_topk2.py` | Performs reproducible fixed-length generation with `top_k=2`, temperature 1.0, and seed 42. |
| `evaluate_B4_only_lmax_R_smith_waterman_blast6.py` | Evaluates each fixed-length generation against every valid partner using Smith-Waterman and BLAST6. |
| `generate_random_lmax_control.py` | Generates reproducible uniformly random A/C/G/U sequences matched to the model-generated `lmax` for every RNA1. |
| `compare_B4_model_random_R_ttest.py` | Performs paired model-versus-random statistical comparisons for Smith-Waterman and BLAST6 scores. |
| `heatmap_interaction_types_nonaug_pkl.py` | Produces the RNA1-by-RNA2 interaction-category heatmap and the underlying count tables. |

## B4 1-mer runtime

The generation scripts require the original B4 1-mer runtime:

| File | Role |
|---|---|
| `model.py` | Historical `NucTransformer` architecture. |
| `tokenizer.py` | 1-mer tokenizer with the vocabulary `PAD`, `EOS`, `UNK`, `A`, `U`, `C`, and `G`. |
| `pos_encoding.py` | Positional encoding used by the Transformer. |
| `requirements.txt` | Python environment used for the original experiments. |

Do not replace the B4 tokenizer with the later 3-mer tokenizer. The historical
B4 checkpoint expects a vocabulary size of 7 and does not use a BOS token.

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
| `Margherita_train_df_no_augmentation_all.pkl` | Non-augmented training pairs used for the interaction-category heatmap. |
| `Margherita_test_df_no_augmentation_all.pkl` | Non-augmented test pairs used for the interaction-category heatmap. |
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

### Greedy generation

Run from the repository root:

```bash
python generate_B4_only_lmax.py \
  --model_dir ckpt/dry_run/final_model \
  --legacy_code_dir . \
  --dataset data/dry_run/test_filtered_no_aug_pident95_qcov90.tsv \
  --t2_dataset rna1_to_t2_targets_train+test.tsv \
  --output B4_only_lmax_generations.tsv
```

The script prints each RNA1, its selected reference partner, the complete
generated sequence, per-example diagnostics, progressive summaries, and a final
summary.

### Top-k generation

```bash
python generate_B4_only_lmax_topk2.py \
  --model_dir ckpt/dry_run/final_model \
  --legacy_code_dir . \
  --dataset data/dry_run/test_filtered_no_aug_pident95_qcov90.tsv \
  --t2_dataset rna1_to_t2_targets_train+test.tsv \
  --output B4_only_lmax_topk2_generations.tsv
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

### Evaluate greedy fixed-length generations

```bash
python evaluate_B4_only_lmax_R_smith_waterman_blast6.py \
  --generations B4_only_lmax_generations.tsv \
  --t2 rna1_to_t2_targets_train+test.tsv \
  --output_prefix B4_only_lmax_T2_R \
  --blast_dir tools/ncbi-blast-2.17.0+/bin
```

### Evaluate top-k fixed-length generations

```bash
python evaluate_B4_only_lmax_R_smith_waterman_blast6.py \
  --generations B4_only_lmax_topk2_generations.tsv \
  --t2 rna1_to_t2_targets_train+test.tsv \
  --output_prefix B4_only_lmax_topk2_T2_R \
  --blast_dir tools/ncbi-blast-2.17.0+/bin
```

For each run, the evaluator writes:

```text
<output_prefix>_details.tsv
<output_prefix>_best.tsv
<output_prefix>_summary.tsv
```

The details file contains one row per `(RNA1, x')` comparison. The best file
contains one row per RNA1. BLAST6 no-hit cases remain visible in the best file
but are excluded from the hit-conditioned mean and reported separately.

## Length-matched random control

Generate one uniformly random RNA sequence with the same `lmax` as each
model-generated sequence:

```bash
python generate_random_lmax_control.py \
  --source_generations B4_only_lmax_topk2_generations.tsv \
  --t2 rna1_to_t2_targets_train+test.tsv \
  --output B4_only_lmax_topk2_random_seed42_generations.tsv \
  --seed 42
```

Evaluate the random sequences with exactly the same alignment pipeline:

```bash
python evaluate_B4_only_lmax_R_smith_waterman_blast6.py \
  --generations B4_only_lmax_topk2_random_seed42_generations.tsv \
  --t2 rna1_to_t2_targets_train+test.tsv \
  --output_prefix B4_only_lmax_topk2_random_seed42_T2_R \
  --blast_dir tools/ncbi-blast-2.17.0+/bin
```

Run the paired statistical comparison:

```bash
python compare_B4_model_random_R_ttest.py \
  --model_best B4_only_lmax_topk2_T2_R_best.tsv \
  --random_best B4_only_lmax_topk2_random_seed42_T2_R_best.tsv \
  --output_prefix B4_only_lmax_topk2_model_vs_random_ttest
```

The comparison writes detailed paired results, summary tables, a JSON summary,
and an English text file suitable for the paper results section.

## Interaction-category heatmap

Place the two non-augmented pickle files in the repository root and run:

```bash
python heatmap_interaction_types_nonaug_pkl.py
```

The script writes:

```text
heatmap_interaction_types_nonaug_pkl.png
heatmap_interaction_types_nonaug_pkl.pdf
heatmap_interaction_types_nonaug_pkl_counts.tsv
heatmap_interaction_types_nonaug_pkl_split_counts.tsv
```

The script verifies the required columns, missing values, the consistency of
`Category_Couple`, and the total number of train and test pairs before producing
the figure.

## Main results

The complete fixed-length experiments contain 697 distinct test RNA1 sequences
and 13,574 `(RNA1, x')` comparisons.

| Generation method | Distinct exact generations | Smith-Waterman mean best R | BLAST6 mean best R among hits | BLAST6 hits | BLAST6 no-hits |
|---|---:|---:|---:|---:|---:|
| Greedy | 187/697 (26.83%) | 0.628900 | 0.221262 | 618 | 79 |
| Top-k=2 | 695/697 (99.71%) | 0.619680 | 0.224590 | 622 | 75 |
| Uniform random control | not used as a diversity endpoint | 0.638543 | 0.219163 | 615 | 82 |

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
