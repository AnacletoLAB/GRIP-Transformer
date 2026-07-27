# ================================================================
# final_generation_0.py
# ------------------------------------------------
# Generazione completa di seq2_pred a partire da seq1 con NucTransformer
# (nessuna parte del target fornita)
# Include temperatura (--temperature), ricostruzione da k-mer,
# metriche e statistiche riassuntive.
# ================================================================

import torch
import pandas as pd
import argparse
from model import NucTransformer, NucConfig
from tokenizer import tokenizer, tokenize_batch
from safetensors.torch import load_file

# Argomenti da riga di comando
parser = argparse.ArgumentParser(description="Generazione completa di sequenze RNA con NucTransformer")
parser.add_argument("--dataset", type=str, default="data/dry_run/all_test_noaug_clean_max250.tsv",
                    help="Percorso del file TSV di input (default = all_test_noaug_clean_max250.tsv)")
parser.add_argument("--temperature", type=float, default=1.0, help="Valore della temperatura (default = 1.0)")
parser.add_argument("--top_k", type=int, default=4,
                    help="Numero di token tra cui campionare (default = 4 = sampling tra A/U/C/G; "
                         "metti 1 per il greedy deterministico)")
args = parser.parse_args()

device = torch.device("cuda")

MODEL_PATH = "/var/tmp/margherita/ckpt/dry_run/ep12_conditioned_TVmax032"
print(f"Carico il modello da {MODEL_PATH} ...")

config = NucConfig.from_pretrained(MODEL_PATH)
model = NucTransformer(config)
state_dict = load_file(f"{MODEL_PATH}/model.safetensors", device=torch.cuda.current_device())
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Caricamento parziale del modello: missing={len(missing)}, unexpected={len(unexpected)}")
model.to(device)
model.eval()

# Carico dataset
print(f"Carico il dataset: {args.dataset}")
if args.dataset.endswith(".csv"):
    df = pd.read_csv(args.dataset, sep=";")
else:
    df = pd.read_csv(args.dataset, sep="\t")

# Determino le colonne
if "RNA_sequence_x" in df.columns and "RNA_sequence_y" in df.columns:
    src_col, tgt_col = "RNA_sequence_x", "RNA_sequence_y"
elif "seq1" in df.columns and "seq2" in df.columns:
    src_col, tgt_col = "seq1", "seq2"
else:
    raise ValueError("Colonne RNA_sequence_x / RNA_sequence_y o seq1 / seq2 non trovate nel dataset.")

print("Tokenizzo le sequenze di input...")
tokenized = tokenize_batch({
    "source": df[src_col].tolist(),
    "target": df[tgt_col].tolist()
})

# tokenize_batch ora restituisce LISTE a lunghezza variabile (padding dinamico):
# niente .to(device) qui. Ogni sorgente viene convertita in tensore nel loop.
input_ids = tokenized["input_ids"]
attention_mask = tokenized["attention_mask"]

# ------------------------------------------------
# Funzioni di supporto
# ------------------------------------------------
'''def percent_identity(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return round((matches / min_len) * 100, 2)
'''
def percent_identity(gen, tgt):
    if len(tgt) == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(gen, tgt))
    return round((matches / len(tgt)) * 100, 2)


def levenshtein_distance(seq1, seq2):
    return abs(len(seq1) - len(seq2)) + sum(1 for x, y in zip(seq1, seq2) if x != y)

def similarity_score(seq1, seq2):
    dist = levenshtein_distance(seq1, seq2)
    return max(0.0, 1 - (dist / max(1, (len(seq1) + len(seq2)) / 2)))

# ------------------------------------------------
# Generazione completa (nessuna parte del target fornita)
# ------------------------------------------------
print(f"\nGenero le sequenze complete con temperatura = {args.temperature} ...")
generated_full = []

# ------------------------------------------------
# Ricostruzione da k-mer
# ------------------------------------------------
def reconstruct_from_kmers(seq):
    tokens = seq.strip().split()
    if not tokens:
        return ""
    s = tokens[0]
    for t in tokens[1:]:
        s += t[-1]
    return s

with torch.no_grad():
    for i in range(len(input_ids)):
        src = torch.tensor(input_ids[i], dtype=torch.long, device=device).unsqueeze(0)
        mask = torch.tensor(attention_mask[i], dtype=torch.long, device=device).unsqueeze(0)
        tgt_seq = df[tgt_col].iloc[i]

        # Numero di token da generare: stessa lunghezza del target
        tgt_tokens = tokenizer(
            tgt_seq,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False
        )["input_ids"]
        total_len = tgt_tokens.shape[1]
        remaining_len = total_len - 1
        #remaining_len = max(0, remaining_len - 1)

        print(f"\n[DEBUG {i+1}] Target len (token): {total_len}")
        gen = model.generate(
            input_ids=src,
            attention_mask=mask,
            max_new_tokens=total_len,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
        reconstructed = reconstruct_from_kmers(decoded)
        generated_full.append(decoded)
        pident = percent_identity(tgt_seq, reconstructed)
        levdist = levenshtein_distance(tgt_seq, reconstructed)

        # print(f"[{i+1}/{len(input_ids)}] target: {tgt_seq} → gen: {decoded}")
        print(f"    → Percent identity: {pident:.2f}% | Levenshtein: {levdist}")

df["generated_seq"] = generated_full

df["reconstructed_seq"] = df["generated_seq"].apply(reconstruct_from_kmers)

# ------------------------------------------------
# Metriche di confronto sull'intera seq2
# ------------------------------------------------
df["percent_identity"] = [
    percent_identity(gen, tgt) for gen, tgt in zip(df["reconstructed_seq"], df[tgt_col])
]
df["levenshtein_distance"] = [
    levenshtein_distance(gen, tgt) for gen, tgt in zip(df["reconstructed_seq"], df[tgt_col])
]
df["length_seq2"] = df[tgt_col].str.len()
df["length_generated"] = df["reconstructed_seq"].str.len()
df["length_diff"] = df["length_generated"] - df["length_seq2"]
df["similarity_score"] = [
    similarity_score(gen, tgt) for gen, tgt in zip(df["reconstructed_seq"], df[tgt_col])
]

# ------------------------------------------------
# Statistiche riassuntive
# ------------------------------------------------
print("\nSTATISTICHE SULLE SEQUENZE GENERATE (intere)")
print("Non è stata fornita nessuna parte della sequenza sul target")
print("------------------------------------------")
print(f"Percent identity  → media: {df['percent_identity'].mean():.2f} | std: {df['percent_identity'].std():.2f}")
print(f"Levenshtein dist. → media: {df['levenshtein_distance'].mean():.2f} | std: {df['levenshtein_distance'].std():.2f}")
print(f"Lunghezza diff.   → media: {df['length_diff'].mean():.2f} | std: {df['length_diff'].std():.2f}")
print(f"Similarity score  → media: {df['similarity_score'].mean():.3f} | std: {df['similarity_score'].std():.3f}")
print("------------------------------------------")

# ------------------------------------------------
# Uniformo i nomi delle colonne a seq1 / seq2
# ------------------------------------------------
rename_map = {}

if src_col != "seq1":
    rename_map[src_col] = "seq1"

if tgt_col != "seq2":
    rename_map[tgt_col] = "seq2"

if rename_map:
    df = df.rename(columns=rename_map)

# aggiorno i riferimenti logici
src_col = "seq1"
tgt_col = "seq2"

output_path = f"ffinal_generated_0-topk{args.top_k}-temp{args.temperature}.tsv"
df.to_csv(output_path, sep="\t", index=False)
print(f"\nSalvataggio completato in: {output_path}")

# ----------------------------------------------------
# serve per capire se apprende almeno le regole locali
# ----------------------------------------------------

from collections import Counter
import pandas as pd

def get_base_distribution(series):
    all_bases = "".join(series.tolist())
    counts = Counter(all_bases)
    total = sum(counts.values())
    return {b: v/total for b, v in counts.items()}

# Distribuzioni globali
target_bases = get_base_distribution(df[tgt_col])
gen_bases = get_base_distribution(df["reconstructed_seq"])

comparison = pd.DataFrame({
    "base": sorted(set(target_bases.keys()) | set(gen_bases.keys())),
    "target_freq": [target_bases.get(b, 0) for b in "AUCG"],
    "generated_freq": [gen_bases.get(b, 0) for b in "AUCG"]
})

corr = comparison["target_freq"].corr(comparison["generated_freq"])
print(f"\nCorrelazione tra distribuzioni delle basi (1-mer): {corr:.3f}")
print(comparison)

# --------------------------------------
# Grafico
# --------------------------------------

import matplotlib.pyplot as plt
import numpy as np

bases = comparison["base"].tolist()
target_freq = comparison["target_freq"].tolist()
generated_freq = comparison["generated_freq"].tolist()

x = np.arange(len(bases))
width = 0.35

plt.figure(figsize=(6,4))
plt.bar(x - width/2, target_freq, width, label="Target", alpha=0.8)
plt.bar(x + width/2, generated_freq, width, label="Generato", alpha=0.8)

plt.xticks(x, bases, fontsize=12)
plt.ylabel("Frequenza relativa", fontsize=12)
plt.title("Distribuzione delle basi A/U/G/C", fontsize=13)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("base_distribution_comparison0.png", dpi=300)
print("\nGrafico salvato come: base_distribution_comparison100.png")
