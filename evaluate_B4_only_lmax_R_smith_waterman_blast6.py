from __future__ import annotations

"""Valutazione lmax riservata al run storico B4: SW e BLAST6 su tutto T(x)."""

import argparse
import csv
import multiprocessing
import statistics
import subprocess
from pathlib import Path

from evaluate_generated_against_t2 import (
    load_generations,
    load_t2,
    resolve_blast_tool,
    smith_waterman,
    write_fasta,
)


DEFAULT_GENERATIONS = "B4_only_lmax_generations.tsv"
DEFAULT_T2 = "rna1_to_t2_targets_train+test.tsv"
DEFAULT_OUTPUT_PREFIX = "B4_only_lmax_T2_R"
DEFAULT_BLAST_DIR = "tools/ncbi-blast-2.17.0+/bin"
DEFAULT_WORK_DIR = "B4_only_lmax_blast6_R_work"
DEFAULT_BLAST_EVALUE = "1000"
DEFAULT_BLAST_TIMEOUT_SECONDS = 600


DETAIL_FIELDS = [
    "row_index",
    "seq1",
    "generated_seq",
    "len_generated_seq",
    "lmax",
    "n_T2",
    "target_index",
    "xprime",
    "len_xprime",
    "sw_matches",
    "sw_aln_len",
    "sw_aln_pident",
    "sw_score",
    "sw_R",
    "blast6_hit",
    "blast6_nident",
    "blast6_aln_len",
    "blast6_aln_pident",
    "blast6_bitscore",
    "blast6_evalue",
    "blast6_R",
]


BEST_FIELDS = [
    "row_index",
    "seq1",
    "generated_seq",
    "len_generated_seq",
    "lmax",
    "n_T2",
    "sw_selected_target_index",
    "sw_selected_xprime",
    "sw_len_xprime",
    "sw_matches",
    "sw_aln_len",
    "sw_aln_pident",
    "sw_score",
    "sw_R",
    "blast6_selected_target_index",
    "blast6_selected_xprime",
    "blast6_len_xprime",
    "blast6_hit",
    "blast6_nident",
    "blast6_aln_len",
    "blast6_aln_pident",
    "blast6_bitscore",
    "blast6_evalue",
    "blast6_R",
]


SUMMARY_FIELDS = [
    "method",
    "n_RNA1_total",
    "n_RNA1_in_mean",
    "excluded_no_hit_count",
    "included_in_mean_pct",
    "mean_best_R",
    "median_best_R",
    "min_best_R",
    "max_best_R",
    "best_R_zero_count_in_mean",
]


def preview_sequence(sequence: str, max_nt: int) -> str:
    if max_nt <= 0 or len(sequence) <= max_nt:
        return sequence
    return sequence[:max_nt] + f"...[{len(sequence)} nt]"


def write_tsv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def validate_inputs(
    generation_rows: list[dict[str, str]],
    t2_by_rna1: dict[str, list[str]],
) -> None:
    seen_rna1: set[str] = set()
    for row_number, row in enumerate(generation_rows, start=2):
        rna1 = row.get("seq1", "").strip()
        generated = row.get("generated_seq", "").strip()
        if not rna1:
            raise ValueError(f"Riga {row_number}: seq1 mancante")
        if rna1 in seen_rna1:
            raise ValueError(f"Riga {row_number}: RNA1 duplicato nel file generazioni")
        seen_rna1.add(rna1)
        if not generated:
            raise ValueError(f"Riga {row_number}: generated_seq vuota")
        if rna1 not in t2_by_rna1 or not t2_by_rna1[rna1]:
            raise ValueError(f"Riga {row_number}: T(x) non trovato per RNA1")

        lmax_text = row.get("max_len_t2", "").strip()
        if not lmax_text:
            raise ValueError(f"Riga {row_number}: max_len_t2/lmax mancante")
        lmax = int(lmax_text)
        expected_lmax = max(len(target) for target in t2_by_rna1[rna1])
        if lmax != expected_lmax:
            raise ValueError(
                f"Riga {row_number}: lmax salvato={lmax}, "
                f"lmax ricalcolato da T(x)={expected_lmax}"
            )
        if len(generated) != lmax:
            raise ValueError(
                f"Riga {row_number}: len(z)={len(generated)} diversa da lmax={lmax}"
            )


def calculate_smith_waterman_details(
    row_index: int,
    rna1: str,
    generated: str,
    targets: list[str],
    lmax: int,
) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    for target_index, target in enumerate(targets):
        sw = smith_waterman(generated, target)
        matches = int(sw["matches"])
        target_length = len(target)
        r_value = matches / target_length if target_length else 0.0
        if not 0.0 <= r_value <= 1.0:
            raise RuntimeError(f"R_SW fuori intervallo per RNA1 riga {row_index + 1}")
        details.append(
            {
                "row_index": row_index,
                "seq1": rna1,
                "generated_seq": generated,
                "len_generated_seq": len(generated),
                "lmax": lmax,
                "n_T2": len(targets),
                "target_index": target_index,
                "xprime": target,
                "len_xprime": target_length,
                "sw_matches": matches,
                "sw_aln_len": int(sw["aln_len"]),
                "sw_aln_pident": float(sw["pident"]),
                "sw_score": int(sw["score"]),
                "sw_R": r_value,
                "blast6_hit": False,
                "blast6_nident": 0,
                "blast6_aln_len": 0,
                "blast6_aln_pident": 0.0,
                "blast6_bitscore": 0.0,
                "blast6_evalue": "",
                "blast6_R": 0.0,
            }
        )
    return details


def calculate_smith_waterman_task(
    task: tuple[int, str, str, list[str], int],
) -> tuple[int, list[dict[str, object]]]:
    row_index, rna1, generated, targets, lmax = task
    return row_index, calculate_smith_waterman_details(
        row_index=row_index,
        rna1=rna1,
        generated=generated,
        targets=targets,
        lmax=lmax,
    )


def calculate_all_smith_waterman(
    generation_rows: list[dict[str, str]],
    t2_by_rna1: dict[str, list[str]],
    workers: int,
    preview_nt: int,
) -> list[list[dict[str, object]]]:
    tasks = [
        (
            row_index,
            row["seq1"].strip(),
            row["generated_seq"].strip(),
            t2_by_rna1[row["seq1"].strip()],
            int(row["max_len_t2"]),
        )
        for row_index, row in enumerate(generation_rows)
    ]
    results: list[list[dict[str, object]] | None] = [None] * len(tasks)
    if workers <= 1:
        iterator = map(calculate_smith_waterman_task, tasks)
        pool = None
    else:
        available_methods = multiprocessing.get_all_start_methods()
        start_method = "fork" if "fork" in available_methods else "spawn"
        context = multiprocessing.get_context(start_method)
        pool = context.Pool(processes=workers)
        iterator = pool.imap_unordered(calculate_smith_waterman_task, tasks, chunksize=1)

    try:
        for completed, (row_index, row_details) in enumerate(iterator, start=1):
            results[row_index] = row_details
            sw_best = select_best(row_details, "sw")
            if sw_best is None:
                raise RuntimeError(
                    f"Nessun risultato Smith-Waterman per RNA1 alla riga {row_index + 1}"
                )
            print(
                f"\n--- Smith-Waterman RNA1 {row_index + 1}/{len(tasks)} ---",
                flush=True,
            )
            print(
                f"RNA1: {preview_sequence(str(sw_best['seq1']), preview_nt)}",
                flush=True,
            )
            print(
                f"z: {preview_sequence(str(sw_best['generated_seq']), preview_nt)}",
                flush=True,
            )
            print(
                f"T2 selezionata: {preview_sequence(str(sw_best['xprime']), preview_nt)}",
                flush=True,
            )
            print(
                "Metriche Smith-Waterman: "
                f"target_index={sw_best['target_index']} "
                f"R={float(sw_best['sw_R']):.6f} "
                f"matches={sw_best['sw_matches']}/{sw_best['len_xprime']} "
                f"aln_len={sw_best['sw_aln_len']} "
                f"aln_pident={float(sw_best['sw_aln_pident']):.2f}% "
                f"score={sw_best['sw_score']}",
                flush=True,
            )
            if completed % 10 == 0 or completed == len(tasks):
                pair_count = sum(len(value) for value in results if value is not None)
                print(
                    f"[Smith-Waterman] RNA1 completati={completed}/{len(tasks)} "
                    f"coppie completate={pair_count}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if any(value is None for value in results):
        raise RuntimeError("Smith-Waterman incompleto: mancano risultati per alcuni RNA1")
    return [value for value in results if value is not None]


def run_blast6_for_rna1(
    row_index: int,
    generated: str,
    targets: list[str],
    details: list[dict[str, object]],
    blastn: str,
    makeblastdb: str,
    work_dir: Path,
    evalue: str,
    timeout_seconds: int,
    num_threads: int,
) -> None:
    row_dir = work_dir / f"row_{row_index:04d}"
    row_dir.mkdir(parents=True, exist_ok=True)
    query_fasta = row_dir / "z.fa"
    target_fasta = row_dir / "T2.fa"
    db_prefix = row_dir / "T2_db"
    output_tsv = row_dir / "blast6.tsv"

    write_fasta(query_fasta, [(f"q{row_index}", generated)])
    target_records = [(f"t{target_index}", target) for target_index, target in enumerate(targets)]
    write_fasta(target_fasta, target_records)

    subprocess.run(
        [makeblastdb, "-in", str(target_fasta), "-dbtype", "nucl", "-out", str(db_prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    outfmt = (
        "6 qseqid sseqid pident length mismatch gapopen "
        "qstart qend sstart send evalue bitscore qlen slen nident"
    )
    try:
        subprocess.run(
            [
                blastn,
                "-query",
                str(query_fasta),
                "-db",
                str(db_prefix),
                "-task",
                "blastn",
                "-word_size",
                "6",
                "-evalue",
                str(evalue),
                "-dust",
                "no",
                "-soft_masking",
                "false",
                "-max_target_seqs",
                str(len(targets)),
                "-max_hsps",
                "1",
                "-num_threads",
                str(num_threads),
                "-outfmt",
                outfmt,
                "-out",
                str(output_tsv),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"BLAST6 timeout per RNA1 alla riga {row_index + 1} dopo {timeout_seconds}s"
        ) from exc

    best_hit_by_target: dict[int, tuple[tuple[float, int, float, int, float], dict[str, object]]] = {}
    if output_tsv.exists():
        with output_tsv.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 15:
                    raise ValueError(
                        f"Output BLAST6 malformato in {output_tsv}, riga {line_number}: "
                        f"attese 15 colonne, trovate {len(parts)}"
                    )
                sid = parts[1]
                if not sid.startswith("t"):
                    raise ValueError(f"Identificatore target BLAST6 inatteso: {sid}")
                target_index = int(sid[1:])
                target_length = len(targets[target_index])
                nident = int(parts[14])
                aln_length = int(parts[3])
                aln_pident = float(parts[2])
                hit_evalue = float(parts[10])
                bitscore = float(parts[11])
                r_value = nident / target_length if target_length else 0.0
                if not 0.0 <= r_value <= 1.0:
                    raise RuntimeError(f"R_BLAST6 fuori intervallo per target {target_index}")
                hit = {
                    "blast6_hit": True,
                    "blast6_nident": nident,
                    "blast6_aln_len": aln_length,
                    "blast6_aln_pident": aln_pident,
                    "blast6_bitscore": bitscore,
                    "blast6_evalue": hit_evalue,
                    "blast6_R": r_value,
                }
                key = (r_value, nident, bitscore, aln_length, -hit_evalue)
                previous = best_hit_by_target.get(target_index)
                if previous is None or key > previous[0]:
                    best_hit_by_target[target_index] = (key, hit)

    for target_index, (_, hit) in best_hit_by_target.items():
        details[target_index].update(hit)


def select_best(details: list[dict[str, object]], method: str) -> dict[str, object] | None:
    if method == "sw":
        return max(
            details,
            key=lambda row: (
                float(row["sw_R"]),
                int(row["sw_matches"]),
                int(row["sw_score"]),
                int(row["sw_aln_len"]),
                -int(row["target_index"]),
            ),
        )
    if method == "blast6":
        hit_details = [row for row in details if bool(row["blast6_hit"])]
        if not hit_details:
            return None
        return max(
            hit_details,
            key=lambda row: (
                float(row["blast6_R"]),
                int(row["blast6_nident"]),
                float(row["blast6_bitscore"]),
                int(row["blast6_aln_len"]),
                -int(row["target_index"]),
            ),
        )
    raise ValueError(f"Metodo non riconosciuto: {method}")


def build_best_row(
    row_index: int,
    rna1: str,
    generated: str,
    lmax: int,
    targets: list[str],
    sw_best: dict[str, object],
    blast6_best: dict[str, object] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "row_index": row_index,
        "seq1": rna1,
        "generated_seq": generated,
        "len_generated_seq": len(generated),
        "lmax": lmax,
        "n_T2": len(targets),
        "sw_selected_target_index": sw_best["target_index"],
        "sw_selected_xprime": sw_best["xprime"],
        "sw_len_xprime": sw_best["len_xprime"],
        "sw_matches": sw_best["sw_matches"],
        "sw_aln_len": sw_best["sw_aln_len"],
        "sw_aln_pident": sw_best["sw_aln_pident"],
        "sw_score": sw_best["sw_score"],
        "sw_R": sw_best["sw_R"],
        "blast6_selected_target_index": "",
        "blast6_selected_xprime": "",
        "blast6_len_xprime": "",
        "blast6_hit": False,
        "blast6_nident": "",
        "blast6_aln_len": "",
        "blast6_aln_pident": "",
        "blast6_bitscore": "",
        "blast6_evalue": "",
        "blast6_R": 0.0,
    }
    if blast6_best is not None:
        row.update(
            {
                "blast6_selected_target_index": blast6_best["target_index"],
                "blast6_selected_xprime": blast6_best["xprime"],
                "blast6_len_xprime": blast6_best["len_xprime"],
                "blast6_hit": blast6_best["blast6_hit"],
                "blast6_nident": blast6_best["blast6_nident"],
                "blast6_aln_len": blast6_best["blast6_aln_len"],
                "blast6_aln_pident": blast6_best["blast6_aln_pident"],
                "blast6_bitscore": blast6_best["blast6_bitscore"],
                "blast6_evalue": blast6_best["blast6_evalue"],
                "blast6_R": blast6_best["blast6_R"],
            }
        )
    return row


def summarize(best_rows: list[dict[str, object]], include_blast6: bool) -> list[dict[str, object]]:
    methods = [("smith_waterman", "sw_R")]
    if include_blast6:
        methods.append(("blast6", "blast6_R"))
    summaries: list[dict[str, object]] = []
    for method, field in methods:
        if method == "blast6":
            rows_in_mean = [row for row in best_rows if bool(row["blast6_hit"])]
        else:
            rows_in_mean = best_rows
        values = [float(row[field]) for row in rows_in_mean]
        total_count = len(best_rows)
        included_count = len(rows_in_mean)
        excluded_no_hit_count = total_count - included_count if method == "blast6" else 0
        summaries.append(
            {
                "method": method,
                "n_RNA1_total": total_count,
                "n_RNA1_in_mean": included_count,
                "excluded_no_hit_count": excluded_no_hit_count,
                "included_in_mean_pct": 100.0 * included_count / total_count if total_count else 0.0,
                "mean_best_R": statistics.fmean(values) if values else "",
                "median_best_R": statistics.median(values) if values else "",
                "min_best_R": min(values) if values else "",
                "max_best_R": max(values) if values else "",
                "best_R_zero_count_in_mean": sum(value == 0.0 for value in values),
            }
        )
    return summaries


def save_outputs(
    output_prefix: str,
    detail_rows: list[dict[str, object]],
    best_rows: list[dict[str, object]],
    include_blast6: bool,
) -> None:
    write_tsv(f"{output_prefix}_details.tsv", detail_rows, DETAIL_FIELDS)
    write_tsv(f"{output_prefix}_best.tsv", best_rows, BEST_FIELDS)
    write_tsv(f"{output_prefix}_summary.tsv", summarize(best_rows, include_blast6), SUMMARY_FIELDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Allinea ogni z a tutti gli x' in T(x), calcola "
            "R=matches/len(x') con Smith-Waterman e BLAST6 e seleziona R massimo."
        )
    )
    parser.add_argument("--generations", default=DEFAULT_GENERATIONS)
    parser.add_argument("--t2", default=DEFAULT_T2)
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--blast_dir", default=DEFAULT_BLAST_DIR)
    parser.add_argument("--work_dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--blast_evalue", default=DEFAULT_BLAST_EVALUE)
    parser.add_argument("--blast_timeout", type=int, default=DEFAULT_BLAST_TIMEOUT_SECONDS)
    parser.add_argument("--blast_threads", type=int, default=1)
    parser.add_argument(
        "--sw_workers",
        type=int,
        default=1,
        help="Processi paralleli per Smith-Waterman; 1 conserva l'esecuzione sequenziale.",
    )
    parser.add_argument("--skip_blast", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 = tutti gli RNA1")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--preview_nt", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_rows = load_generations(args.generations)
    if args.limit > 0:
        generation_rows = generation_rows[: args.limit]
    t2_by_rna1 = load_t2(args.t2)
    validate_inputs(generation_rows, t2_by_rna1)
    if args.sw_workers < 1:
        raise ValueError("--sw_workers deve essere almeno 1")

    blastn = ""
    makeblastdb = ""
    work_dir = Path(args.work_dir)
    if not args.skip_blast:
        blastn = resolve_blast_tool(args.blast_dir, "blastn")
        makeblastdb = resolve_blast_tool(args.blast_dir, "makeblastdb")
        work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generazioni: {args.generations}", flush=True)
    print(f"T(x) completo: {args.t2}", flush=True)
    print(f"RNA1 da valutare: {len(generation_rows)}", flush=True)
    print(f"Smith-Waterman: attivo", flush=True)
    print(f"Processi Smith-Waterman: {args.sw_workers}", flush=True)
    print(f"BLAST6: {'disattivato' if args.skip_blast else 'attivo (word_size=6, max_hsps=1)'}", flush=True)
    if not args.skip_blast:
        print(f"blastn: {blastn}", flush=True)
        print(f"makeblastdb: {makeblastdb}", flush=True)

    print("\nCalcolo Smith-Waterman esatto...", flush=True)
    sw_details_by_row = calculate_all_smith_waterman(
        generation_rows=generation_rows,
        t2_by_rna1=t2_by_rna1,
        workers=args.sw_workers,
        preview_nt=args.preview_nt,
    )

    detail_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    total_pair_alignments = 0

    for row_index, generation_row in enumerate(generation_rows):
        rna1 = generation_row["seq1"].strip()
        generated = generation_row["generated_seq"].strip()
        lmax = int(generation_row["max_len_t2"])
        targets = t2_by_rna1[rna1]
        total_pair_alignments += len(targets)

        row_details = sw_details_by_row[row_index]
        sw_best = select_best(row_details, "sw")
        if sw_best is None:
            raise RuntimeError(f"Nessun risultato Smith-Waterman per RNA1 alla riga {row_index + 1}")

        blast6_best: dict[str, object] | None = None
        if not args.skip_blast:
            print(f"\n--- BLAST6 RNA1 {row_index + 1}/{len(generation_rows)} ---", flush=True)
            print(f"RNA1: {preview_sequence(rna1, args.preview_nt)}", flush=True)
            print(f"z: {preview_sequence(generated, args.preview_nt)}", flush=True)
            print(f"len(z)={len(generated)} lmax={lmax} n_T2={len(targets)}", flush=True)
            run_blast6_for_rna1(
                row_index=row_index,
                generated=generated,
                targets=targets,
                details=row_details,
                blastn=blastn,
                makeblastdb=makeblastdb,
                work_dir=work_dir,
                evalue=args.blast_evalue,
                timeout_seconds=args.blast_timeout,
                num_threads=args.blast_threads,
            )
            blast6_best = select_best(row_details, "blast6")
            if blast6_best is None:
                print(
                    "BLAST6 best: nessun hit in T(x); RNA1 escluso dalla media BLAST6",
                    flush=True,
                )
            else:
                print(
                    "T2 selezionata: "
                    f"{preview_sequence(str(blast6_best['xprime']), args.preview_nt)}",
                    flush=True,
                )
                print(
                    "Metriche BLAST6: "
                    f"target_index={blast6_best['target_index']} "
                    f"hit={blast6_best['blast6_hit']} "
                    f"R={float(blast6_best['blast6_R']):.6f} "
                    f"nident={blast6_best['blast6_nident']}/{blast6_best['len_xprime']} "
                    f"aln_len={blast6_best['blast6_aln_len']} "
                    f"aln_pident={float(blast6_best['blast6_aln_pident']):.2f}% "
                    f"bitscore={float(blast6_best['blast6_bitscore']):.2f}",
                    flush=True,
                )

        detail_rows.extend(row_details)
        best_rows.append(
            build_best_row(row_index, rna1, generated, lmax, targets, sw_best, blast6_best)
        )

        if args.save_every > 0 and (
            (row_index + 1) % args.save_every == 0 or row_index + 1 == len(generation_rows)
        ):
            save_outputs(args.output_prefix, detail_rows, best_rows, not args.skip_blast)
            print(
                f"Salvataggio progressivo: RNA1={len(best_rows)} coppie={len(detail_rows)}",
                flush=True,
            )

    if args.save_every <= 0:
        save_outputs(args.output_prefix, detail_rows, best_rows, not args.skip_blast)

    print("\nRisultato finale:", flush=True)
    print(f"- RNA1 valutati: {len(best_rows)}", flush=True)
    print(f"- Coppie z-x' valutate per metodo: {total_pair_alignments}", flush=True)
    for summary in summarize(best_rows, not args.skip_blast):
        mean_value = summary["mean_best_R"]
        median_value = summary["median_best_R"]
        mean_text = f"{float(mean_value):.6f}" if mean_value != "" else "NA"
        median_text = f"{float(median_value):.6f}" if median_value != "" else "NA"
        print(
            f"- {summary['method']}: mean_best_R={mean_text} "
            f"median_best_R={median_text} "
            f"n_in_mean={summary['n_RNA1_in_mean']}/{summary['n_RNA1_total']} "
            f"excluded_no_hit={summary['excluded_no_hit_count']} "
            f"zero_in_mean={summary['best_R_zero_count_in_mean']}",
            flush=True,
        )
    print(f"- Dettagli: {args.output_prefix}_details.tsv", flush=True)
    print(f"- Migliori: {args.output_prefix}_best.tsv", flush=True)
    print(f"- Summary: {args.output_prefix}_summary.tsv", flush=True)


if __name__ == "__main__":
    main()
