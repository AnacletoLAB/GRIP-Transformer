from __future__ import annotations

"""Genera un controllo casuale uniforme lungo lmax per ogni RNA1."""

import argparse
import csv
import hashlib
import random
import statistics
from collections import Counter
from pathlib import Path

from evaluate_generated_against_t2 import load_generations, load_t2


DEFAULT_SOURCE_GENERATIONS = "B4_only_lmax_topk2_generations.tsv"
DEFAULT_T2 = "rna1_to_t2_targets_train+test.tsv"
DEFAULT_OUTPUT = "B4_only_lmax_topk2_random_seed42_generations.tsv"
DEFAULT_SEED = 42
NUCLEOTIDES = "ACGU"

OUTPUT_FIELDS = [
    "row_index",
    "seq1",
    "max_len_t2",
    "generated_seq",
    "len_generated_seq",
    "random_seed_base",
    "random_seed_rna1",
    "random_distribution",
    "source_generations",
]


def preview_sequence(sequence: str, max_nt: int) -> str:
    if max_nt <= 0 or len(sequence) <= max_nt:
        return sequence
    return sequence[:max_nt] + f"...[{len(sequence)} nt]"


def stable_rna1_seed(base_seed: int, rna1: str) -> int:
    """Deriva un seed stabile per RNA1, indipendente dall'ordine delle righe."""
    payload = f"{base_seed}\0{rna1}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def validate_source_rows(
    source_rows: list[dict[str, str]],
    t2_by_rna1: dict[str, list[str]],
) -> None:
    seen: set[str] = set()
    for row_number, row in enumerate(source_rows, start=2):
        rna1 = row.get("seq1", "").strip()
        if not rna1:
            raise ValueError(f"Riga {row_number}: seq1 mancante")
        if rna1 in seen:
            raise ValueError(f"Riga {row_number}: RNA1 duplicato")
        seen.add(rna1)

        if rna1 not in t2_by_rna1 or not t2_by_rna1[rna1]:
            raise ValueError(f"Riga {row_number}: T(x) non trovato per RNA1")
        saved_lmax_text = row.get("max_len_t2", "").strip()
        if not saved_lmax_text:
            raise ValueError(f"Riga {row_number}: max_len_t2 mancante")
        saved_lmax = int(saved_lmax_text)
        expected_lmax = max(len(target) for target in t2_by_rna1[rna1])
        if saved_lmax != expected_lmax:
            raise ValueError(
                f"Riga {row_number}: lmax salvato={saved_lmax}, "
                f"lmax ricalcolato da T(x)={expected_lmax}"
            )


def write_tsv(path: str | Path, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Per ogni RNA1 genera una sequenza casuale i.i.d. uniforme su A/C/G/U, "
            "lunga esattamente lmax ricalcolato dallo stesso T(x) usato per B4."
        )
    )
    parser.add_argument(
        "--source_generations",
        default=DEFAULT_SOURCE_GENERATIONS,
        help="File B4 che definisce i 697 RNA1 e il loro ordine.",
    )
    parser.add_argument("--t2", default=DEFAULT_T2)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=0, help="0 = tutti gli RNA1")
    parser.add_argument(
        "--preview_nt",
        type=int,
        default=0,
        help="0 = stampa le sequenze complete; N > 0 = mostra solo i primi N nt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = load_generations(args.source_generations)
    if args.limit > 0:
        source_rows = source_rows[: args.limit]
    if not source_rows:
        raise ValueError("Nessuna riga da elaborare")

    t2_by_rna1 = load_t2(args.t2)
    validate_source_rows(source_rows, t2_by_rna1)

    print(f"Generazioni sorgente: {args.source_generations}", flush=True)
    print(f"T(x) completo: {args.t2}", flush=True)
    print(f"RNA1 da elaborare: {len(source_rows)}", flush=True)
    print(f"Seed base: {args.seed}", flush=True)
    print("Controllo casuale: i.i.d. uniforme P(A)=P(C)=P(G)=P(U)=0.25", flush=True)
    print("Lunghezza: len(random_seq)=lmax per ogni RNA1", flush=True)

    output_rows: list[dict[str, object]] = []
    nucleotide_counts: Counter[str] = Counter()
    lengths: list[int] = []

    for row_index, source_row in enumerate(source_rows):
        rna1 = source_row["seq1"].strip()
        lmax = max(len(target) for target in t2_by_rna1[rna1])
        rna1_seed = stable_rna1_seed(args.seed, rna1)
        rng = random.Random(rna1_seed)
        random_sequence = "".join(rng.choice(NUCLEOTIDES) for _ in range(lmax))

        if len(random_sequence) != lmax:
            raise RuntimeError(f"RNA1 {row_index + 1}: len(random_seq) != lmax")
        if set(random_sequence) - set(NUCLEOTIDES):
            raise RuntimeError(f"RNA1 {row_index + 1}: nucleotide casuale non valido")

        nucleotide_counts.update(random_sequence)
        lengths.append(lmax)
        output_rows.append(
            {
                "row_index": row_index,
                "seq1": rna1,
                "max_len_t2": lmax,
                "generated_seq": random_sequence,
                "len_generated_seq": len(random_sequence),
                "random_seed_base": args.seed,
                "random_seed_rna1": rna1_seed,
                "random_distribution": "iid_uniform_ACGU",
                "source_generations": args.source_generations,
            }
        )

        print(
            f"\n--- Controllo casuale RNA1 {row_index + 1}/{len(source_rows)} ---",
            flush=True,
        )
        print(f"RNA1: {preview_sequence(rna1, args.preview_nt)}", flush=True)
        print(
            f"random_seq: {preview_sequence(random_sequence, args.preview_nt)}",
            flush=True,
        )
        print(
            f"len(random_seq)={len(random_sequence)} lmax={lmax} "
            f"n_T2={len(t2_by_rna1[rna1])} seed_RNA1={rna1_seed}",
            flush=True,
        )

    write_tsv(args.output, output_rows)

    total_nt = sum(nucleotide_counts.values())
    distinct_sequences = len({str(row["generated_seq"]) for row in output_rows})
    print("\nRiepilogo finale controllo casuale:", flush=True)
    print(f"- RNA1 elaborati: {len(output_rows)}", flush=True)
    print(f"- sequenze casuali distinte: {distinct_sequences}/{len(output_rows)}", flush=True)
    print(
        "- lmax min/mediana/media/max: "
        f"{min(lengths)}/{statistics.median(lengths):.2f}/"
        f"{statistics.fmean(lengths):.2f}/{max(lengths)}",
        flush=True,
    )
    for nucleotide in NUCLEOTIDES:
        count = nucleotide_counts[nucleotide]
        percentage = 100.0 * count / total_nt if total_nt else 0.0
        print(f"- {nucleotide}: {count}/{total_nt} ({percentage:.4f}%)", flush=True)
    print(f"- Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
