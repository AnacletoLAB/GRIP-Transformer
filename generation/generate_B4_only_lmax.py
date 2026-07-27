from __future__ import annotations

"""Generazione B4 legacy a lunghezza fissa lmax.

Questo script e' intenzionalmente separato dal workflow B6/3-mer. Usa i 697
RNA1 del test storico, recupera T(x) dal file raggruppato completo e genera
una sola sequenza z per RNA1 imponendo len(z) = lmax.

Deve essere eseguito nell'ambiente che contiene il codice legacy 1-mer di B4
(model.py e tokenizer.py compatibili con il checkpoint a vocabolario 7).
"""

import argparse
import csv
import importlib
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_DATASET = "data/dry_run/test_filtered_no_aug_pident95_qcov90.tsv"
DEFAULT_T2_DATASET = "rna1_to_t2_targets_train+test.tsv"
DEFAULT_OUTPUT = "B4_only_lmax_generations.tsv"
DEFAULT_EXPECTED_RNA1 = 697
RNA_ALPHABET = ("A", "U", "C", "G")


def normalize_column_name(name: str) -> str:
    return "".join(char.lower() for char in name.strip() if char.isalnum())


def pick_column(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    normalized = {
        normalize_column_name(key): value
        for key, value in row.items()
        if key is not None
    }
    for name in names:
        value = normalized.get(normalize_column_name(name))
        if value and value.strip():
            return value.strip()
    return None


def split_target_list(value: str) -> list[str]:
    normalized = value.replace(";", ",").replace("|", ",")
    return [target.strip() for target in normalized.split(",") if target.strip()]


def load_groups(path: str | Path) -> tuple[list[tuple[str, list[str]]], int]:
    groups: dict[str, list[str]] = {}
    total_rows = 0
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            seq1 = pick_column(
                row,
                (
                    "RNA1",
                    "seq1",
                    "source",
                    "RNA_sequence_x",
                    "RNA_sequence_1",
                    "RNA1_sequence",
                ),
            )
            seq2_list = pick_column(
                row,
                ("RNA2_list", "T2", "T2_list", "targets", "target_list", "seq2_list"),
            )
            seq2 = pick_column(
                row,
                ("RNA2", "seq2", "target", "RNA_sequence_y", "RNA_sequence_2"),
            )
            if seq2_list:
                targets_for_row = split_target_list(seq2_list)
            elif seq2:
                targets_for_row = [seq2]
            else:
                targets_for_row = []

            if not seq1:
                raise ValueError(f"Riga {row_number} malformata in {path}: RNA1 mancante")
            bucket = groups.setdefault(seq1, [])
            for target in targets_for_row:
                if target not in bucket:
                    bucket.append(target)

    return list(groups.items()), total_rows


def validate_rna(sequence: str, label: str) -> None:
    invalid = sorted(set(sequence) - set(RNA_ALPHABET))
    if invalid:
        raise ValueError(f"{label} contiene caratteri non RNA: {invalid}")


def load_legacy_runtime(
    legacy_code_dir: str | Path,
    model_dir: str | Path,
    device: torch.device,
) -> tuple[Any, Any, dict[str, int]]:
    code_dir = Path(legacy_code_dir).resolve()
    if not (code_dir / "model.py").exists() or not (code_dir / "tokenizer.py").exists():
        raise FileNotFoundError(
            f"In {code_dir} non trovo model.py e tokenizer.py legacy di B4. "
            "Passa la cartella corretta con --legacy_code_dir."
        )

    sys.path.insert(0, str(code_dir))
    tokenizer_module = importlib.import_module("tokenizer")
    model_module = importlib.import_module("model")
    tokenizer = tokenizer_module.tokenizer
    NucConfig = model_module.NucConfig
    NucTransformer = model_module.NucTransformer

    config = NucConfig.from_pretrained(model_dir)
    vocab_size = int(getattr(config, "vocab_size", -1))
    encoder_layers = int(getattr(config, "num_encoder_layers", -1))
    decoder_layers = int(getattr(config, "num_decoder_layers", -1))
    if vocab_size != 7 or encoder_layers != 12 or decoder_layers != 12:
        raise ValueError(
            "Il modello indicato non ha la configurazione storica B4 attesa: "
            f"vocab_size={vocab_size}, encoder={encoder_layers}, decoder={decoder_layers}. "
            "Attesi: vocab_size=7, encoder=12, decoder=12."
        )

    vocab = dict(tokenizer.get_vocab())
    missing_nucleotides = [token for token in RNA_ALPHABET if token not in vocab]
    if missing_nucleotides:
        raise ValueError(f"Tokenizer B4 incompatibile: mancano {missing_nucleotides}")
    if int(getattr(tokenizer, "vocab_size", len(vocab))) != 7:
        raise ValueError("Il tokenizer caricato non e' il tokenizer 1-mer a 7 token di B4")

    model = NucTransformer(config)
    model_path = Path(model_dir)
    safetensors_path = model_path / "model.safetensors"
    pytorch_path = model_path / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path), device=str(device))
    elif pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location=device)
    else:
        raise FileNotFoundError(
            f"Nessun model.safetensors o pytorch_model.bin trovato in {model_path}"
        )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer, {token: int(vocab[token]) for token in RNA_ALPHABET}


def special_token_id(tokenizer: Any) -> int | None:
    vocab = dict(tokenizer.get_vocab())
    for name in ("STOP", "EOS"):
        if name in vocab:
            return int(vocab[name])
    value = getattr(tokenizer, "eos_token_id", None)
    return int(value) if value is not None else None


def source_tensor(
    sequence: str,
    vocab: dict[str, int],
    pad_id: int,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    validate_rna(sequence, "RNA1")
    ids = [vocab[token] for token in sequence[:max_len]]
    ids.extend([pad_id] * (max_len - len(ids)))
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


@torch.no_grad()
def generate_exact_lmax(
    model: Any,
    tokenizer: Any,
    nucleotide_ids: dict[str, int],
    source: str,
    lmax: int,
    start_token: str,
    device: torch.device,
) -> tuple[str, list[float]]:
    if lmax <= 0:
        raise ValueError(f"lmax non valido: {lmax}")
    model_max_len = int(getattr(model.config, "max_len", 1024))
    if lmax > model_max_len:
        raise ValueError(
            f"lmax={lmax} supera max_len={model_max_len} del modello B4"
        )

    pad_id = int(getattr(tokenizer, "pad_token_id", 0))
    input_ids = source_tensor(
        sequence=source,
        vocab=nucleotide_ids,
        pad_id=pad_id,
        max_len=model_max_len,
        device=device,
    )

    # B4 storico non usava la padding mask: il comportamento viene mantenuto.
    src = model.pos_enc(model.embed_src(input_ids))
    memory = model.transformer.encoder(src)

    # Il file storico 0% parte dalla traiettoria con primo nucleotide G. Questo
    # seed non proviene dal target e viene contato nella lunghezza finale di z.
    generated = torch.tensor(
        [[nucleotide_ids[start_token]]], dtype=torch.long, device=device
    )
    stop_id = special_token_id(tokenizer)
    p_stop: list[float] = []
    allowed_ids = torch.tensor(
        [nucleotide_ids[token] for token in RNA_ALPHABET],
        dtype=torch.long,
        device=device,
    )

    while generated.size(1) < lmax:
        tgt = model.pos_enc(model.embed_tgt(generated))
        length = tgt.size(1)
        tgt_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device),
            diagonal=1,
        )
        decoded = model.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        logits = model.out(decoded)[:, -1, :]
        probabilities = F.softmax(logits, dim=-1)
        if stop_id is not None:
            p_stop.append(float(probabilities[0, stop_id].detach().cpu()))

        # STOP/EOS, PAD e UNK non sono selezionabili: B4 deve arrivare a lmax.
        allowed_logits = logits.index_select(-1, allowed_ids)
        next_local_index = int(torch.argmax(allowed_logits, dim=-1).item())
        next_id = int(allowed_ids[next_local_index].item())
        generated = torch.cat(
            [generated, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )

    inverse_vocab = {token_id: token for token, token_id in nucleotide_ids.items()}
    generated_seq = "".join(inverse_vocab[int(token_id)] for token_id in generated[0].tolist())
    if len(generated_seq) != lmax:
        raise RuntimeError(f"len(z)={len(generated_seq)} ma lmax={lmax}")
    return generated_seq, p_stop


OUTPUT_FIELDS = [
    "seq1",
    "seq2",
    "n_valid_seq2",
    "max_len_t2",
    "max_new_tokens_used",
    "generated_seq",
    "len_generated_seq",
    "len_seq2",
    "percent_identity",
    "levenshtein_distance",
    "length_diff",
    "similarity_score",
    "start_token",
    "prompt_target_fraction",
    "stop_was_disabled",
    "p_stop_probabilities",
    "max_p_stop_position",
    "max_p_stop",
    "fixed_length_reached",
]


def write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def preview(sequence: str, max_nt: int) -> str:
    if max_nt <= 0 or len(sequence) <= max_nt:
        return sequence
    return sequence[:max_nt] + f"...[{len(sequence)} nt]"


def legacy_percent_identity(generated: str, target: str) -> float:
    """Percent identity severa usata dal vecchio generatore B4."""
    if not target:
        return 0.0
    matches = sum(a == b for a, b in zip(generated, target))
    return 100.0 * matches / len(target)


def legacy_levenshtein_distance(generated: str, target: str) -> int:
    """Definizione storica del file B4: mismatch posizione-per-posizione + delta lunghezza."""
    return abs(len(generated) - len(target)) + sum(
        a != b for a, b in zip(generated, target)
    )


def legacy_similarity_score(generated: str, target: str) -> float:
    distance = legacy_levenshtein_distance(generated, target)
    denominator = max(1.0, (len(generated) + len(target)) / 2.0)
    return max(0.0, 1.0 - distance / denominator)


def best_legacy_target_metrics(
    generated: str,
    targets: list[str],
) -> tuple[str, float, int, float]:
    scored = [
        (
            target,
            legacy_percent_identity(generated, target),
            legacy_levenshtein_distance(generated, target),
            legacy_similarity_score(generated, target),
        )
        for target in targets
    ]
    return min(
        scored,
        key=lambda item: (item[2], -item[1], -item[3], abs(len(generated) - len(item[0]))),
    )


def generation_counts(rows: list[dict[str, object]]) -> tuple[int, int, str]:
    counts: dict[str, int] = {}
    for row in rows:
        generated = str(row["generated_seq"])
        counts[generated] = counts.get(generated, 0) + 1
    if not counts:
        return 0, 0, ""
    most_common_sequence = max(counts, key=counts.get)
    return len(counts), counts[most_common_sequence], most_common_sequence


def print_metric_summary(prefix: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    distinct, most_common_count, most_common_sequence = generation_counts(rows)
    print(
        f"{prefix} "
        f"pident_mean={statistics.fmean(float(row['percent_identity']) for row in rows):.2f}% "
        f"levenshtein_mean={statistics.fmean(int(row['levenshtein_distance']) for row in rows):.2f} "
        f"length_diff_mean={statistics.fmean(int(row['length_diff']) for row in rows):.2f} "
        f"similarity_mean={statistics.fmean(float(row['similarity_score']) for row in rows):.4f} "
        f"distinct_generated={distinct}/{len(rows)} "
        f"most_common={most_common_count}/{len(rows)} "
        f"most_common_preview={preview(most_common_sequence, 60)}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generazione legacy B4 1-mer, greedy, prompt target 0%, len(z)=lmax."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Checkpoint storico B4 a vocabolario 7.",
    )
    parser.add_argument(
        "--legacy_code_dir",
        default=".",
        help="Cartella che contiene model.py e tokenizer.py storici di B4.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--t2_dataset", default=DEFAULT_T2_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--expected_rna1", type=int, default=DEFAULT_EXPECTED_RNA1)
    parser.add_argument("--start_token", choices=RNA_ALPHABET, default="G")
    parser.add_argument("--limit", type=int, default=0, help="Solo smoke test; 0 = tutti.")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--live_every", type=int, default=10)
    parser.add_argument("--preview_nt", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Verifica test, T(x) e lmax senza caricare il modello e senza generare.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    test_groups, test_rows = load_groups(args.dataset)
    full_groups, full_rows = load_groups(args.t2_dataset)
    t2_by_rna1 = dict(full_groups)
    selected_rna1 = [rna1 for rna1, _ in test_groups]
    if args.limit > 0:
        selected_rna1 = selected_rna1[: args.limit]
    elif args.expected_rna1 > 0 and len(selected_rna1) != args.expected_rna1:
        raise ValueError(
            f"RNA1 distinti nel test={len(selected_rna1)}, attesi={args.expected_rna1}"
        )

    missing = [rna1 for rna1 in selected_rna1 if rna1 not in t2_by_rna1]
    if missing:
        raise ValueError(
            f"{len(missing)} RNA1 del test non presenti nel T(x) completo; primo={missing[0]}"
        )
    lmax_values = [max(len(target) for target in t2_by_rna1[rna1]) for rna1 in selected_rna1]

    print(f"Dataset test storico: {args.dataset}", flush=True)
    print(f"Righe test: {test_rows}", flush=True)
    print(f"RNA1 distinti selezionati: {len(selected_rna1)}", flush=True)
    print(f"T(x) completo: {args.t2_dataset} ({full_rows} RNA1)", flush=True)
    print(
        "lmax: "
        f"min={min(lmax_values)} median={statistics.median(lmax_values):.1f} "
        f"mean={statistics.fmean(lmax_values):.2f} max={max(lmax_values)}",
        flush=True,
    )
    print("Modello: B4 legacy 1-mer; greedy; prompt target=0%", flush=True)
    print(f"Seed iniziale non-target: {args.start_token}", flush=True)
    print("STOP/EOS disabilitato; vincolo esatto len(z)=lmax", flush=True)

    if args.preflight_only:
        print("Preflight completato: nessun modello caricato, nessuna generazione avviata.", flush=True)
        return

    model, tokenizer, nucleotide_ids = load_legacy_runtime(
        legacy_code_dir=args.legacy_code_dir,
        model_dir=args.model_dir,
        device=device,
    )

    rows: list[dict[str, object]] = []
    for index, rna1 in enumerate(selected_rna1, start=1):
        targets = t2_by_rna1[rna1]
        lmax = max(len(target) for target in targets)
        generated_seq, p_stop = generate_exact_lmax(
            model=model,
            tokenizer=tokenizer,
            nucleotide_ids=nucleotide_ids,
            source=rna1,
            lmax=lmax,
            start_token=args.start_token,
            device=device,
        )
        if p_stop:
            max_stop_index = max(range(len(p_stop)), key=p_stop.__getitem__)
            max_stop_position: int | str = max_stop_index + 2
            max_stop_probability: float | str = p_stop[max_stop_index]
        else:
            max_stop_position = ""
            max_stop_probability = ""
        best_target, pident, levenshtein, similarity = best_legacy_target_metrics(
            generated_seq,
            targets,
        )
        row = {
            "seq1": rna1,
            "seq2": best_target,
            "n_valid_seq2": len(targets),
            "max_len_t2": lmax,
            "max_new_tokens_used": lmax - 1,
            "generated_seq": generated_seq,
            "len_generated_seq": len(generated_seq),
            "len_seq2": len(best_target),
            "percent_identity": pident,
            "levenshtein_distance": levenshtein,
            "length_diff": len(generated_seq) - len(best_target),
            "similarity_score": similarity,
            "start_token": args.start_token,
            "prompt_target_fraction": 0.0,
            "stop_was_disabled": True,
            "p_stop_probabilities": " ".join(f"{value:.8g}" for value in p_stop),
            "max_p_stop_position": max_stop_position,
            "max_p_stop": max_stop_probability,
            "fixed_length_reached": len(generated_seq) == lmax,
        }
        rows.append(row)
        print(f"\n--- B4 lmax {index}/{len(selected_rna1)} ---", flush=True)
        print(f"RNA1: {preview(rna1, args.preview_nt)}", flush=True)
        print(f"RNA2 best: {preview(best_target, args.preview_nt)}", flush=True)
        print(f"z: {preview(generated_seq, args.preview_nt)}", flush=True)
        print(
            f"metriche: pident={pident:.2f}% "
            f"levenshtein={levenshtein} "
            f"similarity={similarity:.4f} "
            f"len_target={len(best_target)} "
            f"len(z)={len(generated_seq)} "
            f"length_diff={len(generated_seq) - len(best_target)} "
            f"lmax={lmax} n_T2={len(targets)} "
            f"max_p_stop={max_stop_probability}",
            flush=True,
        )

        if args.live_every > 0 and (
            index % args.live_every == 0 or index == len(selected_rna1)
        ):
            print_metric_summary(f"[live] {index}/{len(selected_rna1)}", rows)

        if args.save_every > 0 and (
            index % args.save_every == 0 or index == len(selected_rna1)
        ):
            write_rows(args.output, rows)
            print(f"Salvataggio progressivo: {args.output} ({len(rows)} RNA1)", flush=True)

    if args.save_every <= 0:
        write_rows(args.output, rows)
    distinct, most_common, _ = generation_counts(rows)
    print("\nRisultato generazione B4 lmax:", flush=True)
    print(f"- RNA1 generati: {len(rows)}", flush=True)
    print(f"- len(z)=lmax: {sum(bool(row['fixed_length_reached']) for row in rows)}/{len(rows)}", flush=True)
    print(f"- sequenze distinte: {distinct}/{len(rows)}", flush=True)
    print(f"- frequenza sequenza piu' comune: {most_common}/{len(rows)}", flush=True)
    print_metric_summary("- metriche finali:", rows)
    print(f"- output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
