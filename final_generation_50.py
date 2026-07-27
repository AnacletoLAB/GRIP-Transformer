# ================================================================
# final_generation.py
# ------------------------------------------------
# Generazione di seq2_pred a partire da seq1 con NucTransformer
# Include temperatura (--temperature), dataset selezionabile,
# metriche, statistiche riassuntive e monitoraggio lunghezze.
# ================================================================

import torch
import pandas as pd
import argparse
from model import NucTransformer, NucConfig
from tokenizer import tokenizer, tokenize_batch#, detokenize_batch
from safetensors.torch import load_file

# Argomenti di default da riga di comando
parser = argparse.ArgumentParser(description="Generazione di sequenze RNA con NucTransformer")
parser.add_argument("--dataset", type=str, default="data/dry_run/test_filtered_no_aug_pident95_qcov90.tsv",
                    help="Percorso del file TSV di input (default = all_test_noaug.tsv)")
parser.add_argument("--temperature", type=float, default=1.0, help="Valore della temperatura (default = 1.0)")

args = parser.parse_args()
device = torch.device("cuda")

MODEL_PATH = "ckpt/dry_run/ep12_conditioned_TVmax032"
print(f"Carico il modello da {MODEL_PATH} ...")

config = NucConfig.from_pretrained(MODEL_PATH)
model = NucTransformer(config)
state_dict = load_file(f"{MODEL_PATH}/model.safetensors", device=torch.cuda.current_device())
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Caricamento parziale del modello: missing={len(missing)}, unexpected={len(unexpected)}")
model.to(device)
model.eval()

# carico il dataset
print(f"Carico il dataset: {args.dataset}")
# df = pd.read_csv(args.dataset, sep="\t")
# Rileva il separatore in base all'estensione del file
if args.dataset.endswith(".csv"):
    df = pd.read_csv(args.dataset, sep=";")
else:
    df = pd.read_csv(args.dataset, sep="\t")

print(f"Dataset caricato con {len(df)} righe dopo la pulizia (test_noaug_clean).")
print("Tokenizzo le sequenze di input...")

if "RNA_sequence_x" in df.columns and "RNA_sequence_y" in df.columns:
    src_col, tgt_col = "RNA_sequence_x", "RNA_sequence_y"
elif "seq1" in df.columns and "seq2" in df.columns:
    src_col, tgt_col = "seq1", "seq2"
else:
    raise ValueError("Colonne RNA_sequence_x / RNA_sequence_y o seq1 / seq2 non trovate nel dataset.")

tutte = pd.concat([df[src_col], df[tgt_col]])
media = tutte.str.len().mean()

# Tokenizzazione
tokenized = tokenize_batch({
    "source": df[src_col].tolist(),
    "target": df[tgt_col].tolist()
})

input_ids = tokenized["input_ids"]
attention_mask = tokenized["attention_mask"]

# ------------------------------------------------
# Generazione con parte del target fornito e stop alla lunghezza del target
# ------------------------------------------------
# ------------------------------------------------
# Generazione con parte del target fornito (50%) e valutazione SOLO sulla tail
# ------------------------------------------------
print(f"\nGenero le sequenze con temperatura = {args.temperature} (50% target fornito)...")

given_prefix_50 = []
target_tail_50 = []
reconstructed_full_50 = []
reconstructed_tail_50 = []

# funzione che toglie gli spazi (se decodifica a token separati)
def reconstruct_from_kmers(seq):
    tokens = seq.strip().split()
    if not tokens:
        return ""
    s = tokens[0]
    for t in tokens[1:]:
        s += t[-1]
    return s

def percent_identity(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return round((matches / min_len) * 100, 2)

def levenshtein_distance(seq1, seq2):
    return abs(len(seq1) - len(seq2)) + sum(1 for x, y in zip(seq1, seq2) if x != y)

with torch.no_grad():
    for i in range(len(input_ids)):
        src = torch.tensor(input_ids[i], dtype=torch.long, device=device).unsqueeze(0)
        mask = torch.tensor(attention_mask[i], dtype=torch.long, device=device).unsqueeze(0)

        tgt_seq = df[tgt_col].iloc[i]
        given_len = max(1, int(len(tgt_seq) * 0.5))

        given_tgt = tgt_seq[:given_len]      # prompt reale
        tgt_tail = tgt_seq[given_len:]       # ground truth da generare

        # Tokenizza la parte iniziale del target come prompt decoder
        given_tgt_ids = tokenizer(
            given_tgt,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False
        )["input_ids"].to(device)

        # Prompt del decoder = BOS + prefisso reale. ep12 e' addestrato con decoder_input = BOS + nucleotidi:
        # senza BOS il primo nucleotide cade dove il modello si aspetta il BOS -> distribuzione spostata ->
        # in greedy degenera in code ripetitive. Il BOS viene poi rimosso da skip_special_tokens.
        bos_col = torch.full((1, 1), tokenizer.vocab["BOS"], dtype=torch.long, device=device)
        decoder_prompt = torch.cat([bos_col, given_tgt_ids], dim=1)

        # Numero di token da generare (calcolato in token, non in nucleotidi)
        tgt_tokens = tokenizer(
            tgt_seq,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False
        )["input_ids"]
        total_len = tgt_tokens.shape[1]
        remaining_len = total_len - given_tgt_ids.shape[1]
        remaining_len = max(0, remaining_len - 1)

        print(f"\n[DEBUG {i+1}] Target tokens: {total_len}, Given tokens: {given_tgt_ids.shape[1]}, Remaining tokens: {remaining_len}")

        gen = model.generate(
            input_ids=src,
            attention_mask=mask,
            decoder_input_ids=decoder_prompt,
            max_new_tokens=remaining_len,
            temperature=args.temperature,
            top_k=1
        )

        decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
        reconstructed_full = reconstruct_from_kmers(decoded)   # prompt + continuazione

        # estraggo SOLO la tail generata togliendo il prompt (in nucleotidi)
        reconstructed_tail = reconstructed_full[len(given_tgt):]
        reconstructed_tail = reconstructed_tail[:len(tgt_tail)]  # taglio alla lunghezza corretta

        print(f"[{i+1}/{len(input_ids)}] target_tail: {tgt_tail} | gen_tail: {reconstructed_tail}")

        given_prefix_50.append(given_tgt)
        target_tail_50.append(tgt_tail)
        reconstructed_full_50.append(reconstructed_full)
        reconstructed_tail_50.append(reconstructed_tail)

# Scrivo colonne con semantica corretta
df["given_prefix_50"] = given_prefix_50
df["target_tail_50"] = target_tail_50
df["reconstructed_full_50"] = reconstructed_full_50
df["reconstructed_tail_50"] = reconstructed_tail_50

# (opzionale ma consigliato) elimina colonne ereditate da file vecchi
# df = df.drop(columns=["generated_tail", "target_tail", "reconstructed_partial", "reconstructed_50"], errors="ignore")

# ------------------------------------------------
# Metriche di confronto sulla seconda metà di seq2
# ------------------------------------------------

def similarity_score(seq1, seq2):
    dist = levenshtein_distance(seq1, seq2)
    return max(0.0, 1 - (dist / max(1, (len(seq1) + len(seq2)) / 2)))

# Confronta la parte generata (ricostruita dai k-mer) con la parte reale corrispondente
df["percent_identity"] = [
    percent_identity(gen, tgt) for gen, tgt in zip(df["reconstructed_tail_50"], df["target_tail_50"])
]
df["levenshtein_distance"] = [
    levenshtein_distance(gen, tgt) for gen, tgt in zip(df["reconstructed_tail_50"], df["target_tail_50"])
]
df["length_seq2"] = df["target_tail_50"].str.len()

df["length_reconstructed"] = df["reconstructed_tail_50"].str.len()
df["length_diff"] = df["length_reconstructed"] - df["length_seq2"]
df["similarity_score"] = [
    similarity_score(gen, tgt) for gen, tgt in zip(df["reconstructed_tail_50"], df["target_tail_50"])
]
# df = df.drop(columns=["reconstructed_partial"])

# ------------------------------------------------
# Calcolo le Statistiche
# ------------------------------------------------
print("\nSTATISTICHE SULLE SEQUENZE GENERATE (solo parte generata)")
print("Fornito il 50% della sequenza target")
print("------------------------------------------")
print("Lunghezza media complessiva delle sequenze:", media)
print(f"Percent identity  → media: {df['percent_identity'].mean():.2f} | std: {df['percent_identity'].std():.2f}")
print(f"Levenshtein dist. → media: {df['levenshtein_distance'].mean():.2f} | std: {df['levenshtein_distance'].std():.2f}")
print(f"Lunghezza diff.   → media: {df['length_diff'].mean():.2f} | std: {df['length_diff'].std():.2f}")
print(f"Similarity score  → media: {df['similarity_score'].mean():.3f} | std: {df['similarity_score'].std():.3f}")
print("------------------------------------------")

# questa serve soltanto nel caso in cui non si ferma l'addestramento alla stessa lunghezza del target
#df["percent_identity_trimmed"] = [
#    percent_identity(gen[:len(tgt)], tgt) for gen, tgt in zip(df["reconstructed_partial"], df["target_tail"])
#]
#print(f"\nPercent identity (tagliata a stessa lunghezza) → media: {df['percent_identity_trimmed'].mean():.2f}")


# output_path = f"generated_50pct_{df[src_col].name}.tsv"
output_path = f"ffinal_generated_50_ep12.tsv"
df.to_csv(output_path, sep="\t", index=False)
print(f"\nSalvataggio completato in: {output_path}")

df = df.drop(columns=["generated_tail", "target_tail", "reconstructed_partial", "reconstructed_50"], errors="ignore")

# ----------------------------------------------------
# serve per capire se apprende almeno le regole locali
# ----------------------------------------------------
from collections import Counter

def get_base_distribution(series):
    all_bases = "".join(series.tolist())
    counts = Counter(all_bases)
    total = sum(counts.values())
    return {b: v/total for b, v in counts.items()}

# Distribuzioni globali (solo tail 50%)
target_bases = get_base_distribution(df["target_tail_50"])
reconstructed_bases = get_base_distribution(df["reconstructed_tail_50"])

bases_order = ["A", "U", "C", "G"]

comparison = pd.DataFrame({
    "base": bases_order,
    "target_freq": [target_bases.get(b, 0) for b in bases_order],
    "reconstructed_freq": [reconstructed_bases.get(b, 0) for b in bases_order]
})

corr = comparison["target_freq"].corr(comparison["reconstructed_freq"])
print(f"\nCorrelazione tra distribuzioni delle basi (1-mer): {corr:.3f}")
print(comparison)

# --------------------------------------
# Grafico
# --------------------------------------
import matplotlib.pyplot as plt
import numpy as np

bases = comparison["base"].tolist()
target_freq = comparison["target_freq"].tolist()
reconstructed_freq = comparison["reconstructed_freq"].tolist()

x = np.arange(len(bases))
width = 0.35

plt.figure(figsize=(6,4))
plt.bar(x - width/2, target_freq, width, label="Target", alpha=0.8)
plt.bar(x + width/2, reconstructed_freq, width, label="Ricostruito", alpha=0.8)

plt.xticks(x, bases, fontsize=12)
plt.ylabel("Frequenza relativa", fontsize=12)
plt.title("Distribuzione delle basi A/U/G/C", fontsize=13)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("base_distribution_comparison50.png", dpi=300)
print("\nGrafico salvato come: base_distribution_comparison50.png")



