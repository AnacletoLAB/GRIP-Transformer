from __future__ import annotations

"""Generazione B4 1-mer con sampling top-k=2 e lunghezza esatta lmax.

Usa i 697 RNA1 distinti del test storico, recupera T(x) dal file raggruppato
completo e genera una sola RNA2 per RNA1 imponendo len(z) = lmax.

TOP_K e' scritto direttamente nel codice. Il decoder parte da PAD; PAD non fa
parte della sequenza generata. Il primo nucleotide e tutti i successivi sono
campionati tra i due nucleotidi A/U/C/G con logit piu' alto. EOS, PAD e UNK non
sono selezionabili, quindi ogni generazione raggiunge esattamente lmax.
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

import generate_B4_only_lmax as base


TOP_K = 2
TEMPERATURE = 1.0
SAMPLING_SEED = 42

DEFAULT_DATASET = "data/dry_run/test_filtered_no_aug_pident95_qcov90.tsv"
DEFAULT_T2_DATASET = "rna1_to_t2_targets_train+test.tsv"
DEFAULT_OUTPUT = "B4_only_lmax_topk2_generations.tsv"
DEFAULT_EXPECTED_RNA1 = 697
DEFAULT_MODEL_DIR = "ckpt/dry_run/final_model"
DEFAULT_LEGACY_CODE_DIR = "."


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
    "top_k",
    "tokenization",
    "vocab_size",
    "temperature",
    "decoder_start_token",
    "first_generated_token",
    "prompt_target_fraction",
    "stop_was_disabled",
    "p_stop_probabilities",
    "max_p_stop_position",
    "max_p_stop",
    "fixed_length_reached",
]


def load_legacy_runtime(
    legacy_code_dir: str | Path,
    model_dir: str | Path,
    device: torch.device,
) -> tuple[Any, Any, dict[str, int]]:
    """Carica il runtime B4 dalla root del progetto sul server."""
    code_dir = Path(legacy_code_dir).resolve()
    sys.path.insert(0, str(code_dir))
    model_module = importlib.import_module("model")
    if not hasattr(model_module.NucTransformer, "all_tied_weights_keys"):
        model_module.NucTransformer.all_tied_weights_keys = {}
    return base.load_legacy_runtime(
        legacy_code_dir=legacy_code_dir,
        model_dir=model_dir,
        device=device,
    )


@torch.no_grad()
def generate_exact_lmax_topk2(
    model: Any,
    tokenizer: Any,
    nucleotide_ids: dict[str, int],
    source: str,
    lmax: int,
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
    input_ids = base.source_tensor(
        sequence=source,
        vocab=nucleotide_ids,
        pad_id=pad_id,
        max_len=model_max_len,
        device=device,
    )

    # Il modello storico non usava la padding mask: comportamento preservato.
    src = model.pos_enc(model.embed_src(input_ids))
    memory = model.transformer.encoder(src)

    # PAD e' solo il prompt tecnico del decoder e non entra in generated_seq.
    decoder_ids = torch.tensor([[pad_id]], dtype=torch.long, device=device)
    generated_nucleotide_ids: list[int] = []

    stop_id = base.special_token_id(tokenizer)
    p_stop: list[float] = []
    allowed_ids = torch.tensor(
        [nucleotide_ids[token] for token in base.RNA_ALPHABET],
        dtype=torch.long,
        device=device,
    )

    for _ in range(lmax):
        tgt = model.pos_enc(model.embed_tgt(decoder_ids))
        length = tgt.size(1)
        tgt_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device),
            diagonal=1,
        )
        decoded = model.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        logits = model.out(decoded)[:, -1, :]

        original_probabilities = F.softmax(logits, dim=-1)
        if stop_id is not None:
            p_stop.append(
                float(original_probabilities[0, stop_id].detach().cpu())
            )

        # Solo A/U/C/G sono selezionabili. Sampling tra i TOP_K piu' probabili.
        allowed_logits = logits.index_select(-1, allowed_ids) / TEMPERATURE
        k = min(TOP_K, allowed_logits.size(-1))
        topk_values, topk_local_indices = torch.topk(
            allowed_logits,
            k=k,
            dim=-1,
        )
        topk_probabilities = F.softmax(topk_values, dim=-1)
        sampled_rank = torch.multinomial(topk_probabilities, num_samples=1)
        next_local_index = int(
            topk_local_indices.gather(-1, sampled_rank).item()
        )
        next_id = int(allowed_ids[next_local_index].item())

        generated_nucleotide_ids.append(next_id)
        decoder_ids = torch.cat(
            [
                decoder_ids,
                torch.tensor([[next_id]], dtype=torch.long, device=device),
            ],
            dim=1,
        )

    inverse_vocab = {
        token_id: token for token, token_id in nucleotide_ids.items()
    }
    generated_seq = "".join(
        inverse_vocab[token_id] for token_id in generated_nucleotide_ids
    )
    if len(generated_seq) != lmax:
        raise RuntimeError(f"len(z)={len(generated_seq)} ma lmax={lmax}")
    return generated_seq, p_stop


def write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generazione B4 1-mer, top_k=2 scritto nel codice, "
            "prompt target 0%, len(z)=lmax."
        )
    )
    parser.add_argument(
        "--model_dir",
        default=DEFAULT_MODEL_DIR,
        help="Checkpoint storico B4 a vocabolario 7.",
    )
    parser.add_argument(
        "--legacy_code_dir",
        default=DEFAULT_LEGACY_CODE_DIR,
        help="Cartella con model.py e tokenizer.py storici di B4.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--t2_dataset", default=DEFAULT_T2_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--expected_rna1", type=int, default=DEFAULT_EXPECTED_RNA1)
    parser.add_argument("--limit", type=int, default=0, help="Solo smoke test; 0 = tutti.")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--live_every", type=int, default=10)
    parser.add_argument("--preview_nt", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Verifica test, T(x) e lmax senza caricare il modello.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    test_groups, test_rows = base.load_groups(args.dataset)
    full_groups, full_rows = base.load_groups(args.t2_dataset)
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
            f"{len(missing)} RNA1 del test non presenti nel T(x) completo; "
            f"primo={missing[0]}"
        )

    lmax_values = [
        max(len(target) for target in t2_by_rna1[rna1])
        for rna1 in selected_rna1
    ]
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
    print(
        f"Modello: B4 legacy; tokenization=1-mer; vocab_size=7; "
        f"top_k={TOP_K}; temperature={TEMPERATURE}; prompt target=0%",
        flush=True,
    )
    print(
        "Decoder start=PAD; primo nucleotide campionato con top_k=2",
        flush=True,
    )
    print("STOP/EOS disabilitato; vincolo esatto len(z)=lmax", flush=True)

    if args.preflight_only:
        print(
            "Preflight completato: nessun modello caricato, "
            "nessuna generazione avviata.",
            flush=True,
        )
        return

    torch.manual_seed(SAMPLING_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SAMPLING_SEED)

    model, tokenizer, nucleotide_ids = load_legacy_runtime(
        legacy_code_dir=args.legacy_code_dir,
        model_dir=args.model_dir,
        device=device,
    )

    rows: list[dict[str, object]] = []
    for index, rna1 in enumerate(selected_rna1, start=1):
        targets = t2_by_rna1[rna1]
        lmax = max(len(target) for target in targets)
        generated_seq, p_stop = generate_exact_lmax_topk2(
            model=model,
            tokenizer=tokenizer,
            nucleotide_ids=nucleotide_ids,
            source=rna1,
            lmax=lmax,
            device=device,
        )

        if p_stop:
            max_stop_index = max(range(len(p_stop)), key=p_stop.__getitem__)
            max_stop_position: int | str = max_stop_index + 1
            max_stop_probability: float | str = p_stop[max_stop_index]
        else:
            max_stop_position = ""
            max_stop_probability = ""

        best_target, pident, levenshtein, similarity = (
            base.best_legacy_target_metrics(generated_seq, targets)
        )
        row = {
            "seq1": rna1,
            "seq2": best_target,
            "n_valid_seq2": len(targets),
            "max_len_t2": lmax,
            "max_new_tokens_used": lmax,
            "generated_seq": generated_seq,
            "len_generated_seq": len(generated_seq),
            "len_seq2": len(best_target),
            "percent_identity": pident,
            "levenshtein_distance": levenshtein,
            "length_diff": len(generated_seq) - len(best_target),
            "similarity_score": similarity,
            "top_k": TOP_K,
            "tokenization": "1-mer",
            "vocab_size": 7,
            "temperature": TEMPERATURE,
            "decoder_start_token": "PAD",
            "first_generated_token": generated_seq[0],
            "prompt_target_fraction": 0.0,
            "stop_was_disabled": True,
            "p_stop_probabilities": " ".join(
                f"{value:.8g}" for value in p_stop
            ),
            "max_p_stop_position": max_stop_position,
            "max_p_stop": max_stop_probability,
            "fixed_length_reached": len(generated_seq) == lmax,
        }
        rows.append(row)

        # Contratto console preservato: sequenze intere per ogni RNA1.
        print(f"\n--- B4 top-k=2 lmax {index}/{len(selected_rna1)} ---", flush=True)
        print(f"RNA1: {base.preview(rna1, args.preview_nt)}", flush=True)
        print(f"RNA2 best: {base.preview(best_target, args.preview_nt)}", flush=True)
        print(f"z: {base.preview(generated_seq, args.preview_nt)}", flush=True)
        print(
            f"metriche: pident={pident:.2f}% "
            f"levenshtein={levenshtein} "
            f"similarity={similarity:.4f} "
            f"len_target={len(best_target)} "
            f"len(z)={len(generated_seq)} "
            f"length_diff={len(generated_seq) - len(best_target)} "
            f"lmax={lmax} n_T2={len(targets)} "
            f"top_k={TOP_K} max_p_stop={max_stop_probability}",
            flush=True,
        )

        if args.live_every > 0 and (
            index % args.live_every == 0 or index == len(selected_rna1)
        ):
            base.print_metric_summary(f"[live] {index}/{len(selected_rna1)}", rows)

        if args.save_every > 0 and (
            index % args.save_every == 0 or index == len(selected_rna1)
        ):
            write_rows(args.output, rows)
            print(
                f"Salvataggio progressivo: {args.output} ({len(rows)} RNA1)",
                flush=True,
            )

    if args.save_every <= 0:
        write_rows(args.output, rows)

    distinct, most_common, _ = base.generation_counts(rows)
    print("\nRisultato generazione B4 top-k=2 lmax:", flush=True)
    print(f"- RNA1 generati: {len(rows)}", flush=True)
    print(
        f"- len(z)=lmax: "
        f"{sum(bool(row['fixed_length_reached']) for row in rows)}/{len(rows)}",
        flush=True,
    )
    print(f"- sequenze distinte: {distinct}/{len(rows)}", flush=True)
    print(
        f"- frequenza sequenza piu' comune: {most_common}/{len(rows)}",
        flush=True,
    )
    base.print_metric_summary("- metriche finali:", rows)
    print(f"- output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
