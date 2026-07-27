from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_GENERATIONS = "ffinal_generated_0.tsv"
DEFAULT_T2 = "data/dry_run/test_filtered_no_aug_pident95_qcov90.tsv"
DEFAULT_OUTPUT_PREFIX = "B4_only_greedy_test_filtered_t2_bestmatch"
DEFAULT_BLAST_DIR = "tools/ncbi-blast-2.17.0+/bin"
DEFAULT_WORK_DIR = "B4_only_blast_t2_work"
DEFAULT_BLAST_TIMEOUT_SECONDS = 600
DEFAULT_BLAST_MAX_TARGET_SEQS = 500
DEFAULT_BLAST_MAX_HSPS = 1
DEFAULT_BLAST_EVALUE = "1000"


def preview_sequence(sequence: str, max_nt: int) -> str:
    if max_nt <= 0 or len(sequence) <= max_nt:
        return sequence
    return sequence[:max_nt] + f"...[{len(sequence)} nt]"


def pick_column(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value:
            return value.strip()
    return None


def normalize_rna(value: str) -> str:
    return "".join(value.split()).upper().replace("T", "U")


def load_t2(path: str | Path) -> dict[str, list[str]]:
    t2: dict[str, list[str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=2):
            raw_rna1 = pick_column(row, ("RNA1", "seq1", "source", "RNA_sequence_x"))
            rna2_list = pick_column(row, ("RNA2_list", "target_list", "seq2_list"))
            rna2 = pick_column(row, ("RNA2", "seq2", "target", "RNA_sequence_y"))
            if rna2_list:
                raw_targets = [value for value in rna2_list.split(",") if value.strip()]
            elif rna2:
                raw_targets = [rna2]
            else:
                raw_targets = []
            if not raw_rna1 or not raw_targets:
                raise ValueError(f"Riga {row_number} malformata in {path}: RNA1/RNA2 mancanti")
            rna1 = normalize_rna(raw_rna1)
            targets = [normalize_rna(value) for value in raw_targets]
            bucket = t2.setdefault(rna1, [])
            for target in targets:
                if target and target not in bucket:
                    bucket.append(target)
    return t2


def load_generations(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=2):
            raw_rna1 = pick_column(row, ("seq1", "RNA1", "source", "RNA_sequence_x"))
            raw_generated = pick_column(
                row,
                ("reconstructed_seq", "generated_seq", "SeqG", "sequence_generated"),
            )
            raw_source_target = pick_column(row, ("seq2", "RNA2", "target", "RNA_sequence_y"))
            if not raw_rna1 or not raw_generated:
                raise ValueError(
                    f"Riga {row_number} malformata in {path}: "
                    "RNA1 o generazione B4 mancanti"
                )
            normalized = dict(row)
            normalized["seq1"] = normalize_rna(raw_rna1)
            normalized["generated_seq"] = normalize_rna(raw_generated)
            normalized["source_seq2"] = normalize_rna(raw_source_target) if raw_source_target else ""
            normalized["_source_row_number"] = str(row_number)
            rows.append(normalized)
    return rows


def select_b4_generation_rows(
    rows: list[dict[str, str]],
    t2_by_rna1: dict[str, list[str]],
    keep_duplicate_rna1: bool,
) -> tuple[list[dict[str, str]], dict[str, int], list[str]]:
    eligible: list[dict[str, str]] = []
    ignored_rna1 = 0
    ignored_pair = 0
    for row in rows:
        rna1 = row["seq1"]
        targets = t2_by_rna1.get(rna1)
        if not targets:
            ignored_rna1 += 1
            continue
        source_target = row.get("source_seq2", "")
        if source_target and source_target not in targets:
            ignored_pair += 1
            continue
        eligible.append(row)

    if keep_duplicate_rna1:
        selected = eligible
    else:
        best_by_rna1: dict[str, dict[str, str]] = {}
        for row in eligible:
            rna1 = row["seq1"]
            candidate_key = (
                len(row["generated_seq"]),
                len(row.get("source_seq2", "")),
                -int(row["_source_row_number"]),
            )
            previous = best_by_rna1.get(rna1)
            if previous is None:
                best_by_rna1[rna1] = row
                continue
            previous_key = (
                len(previous["generated_seq"]),
                len(previous.get("source_seq2", "")),
                -int(previous["_source_row_number"]),
            )
            if candidate_key > previous_key:
                best_by_rna1[rna1] = row
        selected = sorted(
            best_by_rna1.values(),
            key=lambda row: int(row["_source_row_number"]),
        )

    covered_rna1 = {row["seq1"] for row in eligible}
    missing_rna1 = sorted(set(t2_by_rna1) - covered_rna1)
    stats = {
        "source_generation_rows": len(rows),
        "target_rna1": len(t2_by_rna1),
        "eligible_generation_rows": len(eligible),
        "selected_generation_rows": len(selected),
        "covered_rna1": len(covered_rna1),
        "missing_rna1": len(missing_rna1),
        "ignored_rows_rna1_not_in_t2": ignored_rna1,
        "ignored_rows_pair_not_in_t2": ignored_pair,
    }
    return selected, stats, missing_rna1


def collapse_diagnostics(rows: list[dict[str, str]]) -> dict[str, object]:
    sequences = sorted(
        {row["generated_seq"] for row in rows},
        key=lambda value: (len(value), value),
    )
    if not sequences:
        return {
            "unique_exact_generations": 0,
            "unique_generated_lengths": 0,
            "maximal_generation_trajectories": 0,
            "all_prefixes_of_one_trajectory": False,
            "longest_generation_nt": 0,
            "longest_generation": "",
        }
    maximal = [
        sequence
        for sequence in sequences
        if not any(
            len(other) > len(sequence) and other.startswith(sequence)
            for other in sequences
        )
    ]
    one_master = next(
        (
            sequence
            for sequence in maximal
            if all(sequence.startswith(candidate) for candidate in sequences)
        ),
        "",
    )
    longest = max(sequences, key=lambda value: (len(value), value))
    return {
        "unique_exact_generations": len(sequences),
        "unique_generated_lengths": len({len(value) for value in sequences}),
        "maximal_generation_trajectories": len(maximal),
        "all_prefixes_of_one_trajectory": bool(one_master),
        "longest_generation_nt": len(longest),
        "longest_generation": longest,
    }


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def pident_positionwise(generated: str, target: str) -> float:
    if not target:
        return 0.0
    matches = sum(a == b for a, b in zip(generated, target))
    return 100.0 * matches / len(target)


def local_alignment_metrics(
    generated: str,
    target: str,
    aln_len: int,
    aln_pident: float,
) -> dict[str, float]:
    query_len = len(generated)
    target_len = len(target)
    query_coverage = min(100.0, 100.0 * aln_len / query_len) if query_len else 0.0
    target_coverage = min(100.0, 100.0 * aln_len / target_len) if target_len else 0.0
    min_len = min(query_len, target_len)
    min_coverage = min(100.0, 100.0 * aln_len / min_len) if min_len else 0.0
    query_fraction = query_coverage / 100.0
    target_fraction = target_coverage / 100.0
    local_quality = aln_pident * math.sqrt(query_fraction * target_fraction)
    return {
        "query_coverage": query_coverage,
        "target_coverage": target_coverage,
        "min_coverage": min_coverage,
        "local_quality": local_quality,
    }


def smith_waterman(
    query: str,
    target: str,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -2,
) -> dict[str, float | int]:
    if not query or not target:
        return {"score": 0, "aln_len": 0, "matches": 0, "pident": 0.0}

    n = len(query)
    m = len(target)
    prev = [0] * (m + 1)
    pointers: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    best_score = 0
    best_pos = (0, 0)

    for i, qc in enumerate(query, start=1):
        curr = [0] * (m + 1)
        for j, tc in enumerate(target, start=1):
            diag = prev[j - 1] + (match_score if qc == tc else mismatch_score)
            up = prev[j] + gap_score
            left = curr[j - 1] + gap_score
            score = max(0, diag, up, left)
            curr[j] = score
            if score == 0:
                pointers[i][j] = 0
            elif score == diag:
                pointers[i][j] = 1
            elif score == up:
                pointers[i][j] = 2
            else:
                pointers[i][j] = 3
            if score > best_score:
                best_score = score
                best_pos = (i, j)
        prev = curr

    i, j = best_pos
    aln_len = 0
    matches = 0
    while i > 0 and j > 0 and pointers[i][j] != 0:
        direction = pointers[i][j]
        if direction == 1:
            aln_len += 1
            matches += int(query[i - 1] == target[j - 1])
            i -= 1
            j -= 1
        elif direction == 2:
            aln_len += 1
            i -= 1
        else:
            aln_len += 1
            j -= 1

    return {
        "score": best_score,
        "aln_len": aln_len,
        "matches": matches,
        "pident": 100.0 * matches / aln_len if aln_len else 0.0,
    }


def best_smith_waterman(generated: str, targets: list[str]) -> dict[str, object]:
    best: dict[str, object] | None = None
    for index, target in enumerate(targets):
        sw = smith_waterman(generated, target)
        local_metrics = local_alignment_metrics(
            generated,
            target,
            int(sw["aln_len"]),
            float(sw["pident"]),
        )
        candidate = {
            "target_index": index,
            "target": target,
            "sw_score": sw["score"],
            "sw_aln_len": sw["aln_len"],
            "sw_pident": sw["pident"],
            "query_coverage": local_metrics["query_coverage"],
            "target_coverage": local_metrics["target_coverage"],
            "min_coverage": local_metrics["min_coverage"],
            "local_quality": local_metrics["local_quality"],
            "pident": pident_positionwise(generated, target),
            "levenshtein": levenshtein(generated, target),
            "length_diff": len(generated) - len(target),
        }
        key = (
            float(candidate["sw_score"]),
            float(candidate["local_quality"]),
            float(candidate["sw_pident"]),
            -int(candidate["levenshtein"]),
            -abs(int(candidate["length_diff"])),
        )
        if best is None or key > best["_key"]:
            candidate["_key"] = key
            best = candidate
    if best is None:
        return {}
    best.pop("_key", None)
    return best


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record_id, sequence in records:
            sequence = sequence.replace("U", "T")
            handle.write(f">{record_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start:start + 80] + "\n")


def resolve_blast_tool(blast_dir: str | Path, tool_name: str) -> str:
    blast_dir = Path(blast_dir)
    candidates = [
        blast_dir / tool_name,
        blast_dir / f"{tool_name}.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    found = shutil.which(tool_name)
    if found:
        return found
    found_exe = shutil.which(f"{tool_name}.exe")
    if found_exe:
        return found_exe

    raise FileNotFoundError(
        f"{tool_name} non trovato. Cercato in {blast_dir} e nel PATH. "
        f"Su server installare NCBI BLAST+ oppure passare --blast_dir alla cartella bin corretta."
    )


def parse_blast_tsv(
    out_tsv: Path,
    query_to_seq: dict[str, str],
    target_to_rna1: dict[str, str],
    target_to_seq: dict[str, str],
) -> dict[str, dict[str, object]]:
    best_by_query: dict[str, dict[str, object]] = {}
    if not out_tsv.exists():
        return best_by_query

    with out_tsv.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 14:
                continue
            qid, sid = parts[0], parts[1]
            if target_to_rna1.get(sid) != qid:
                continue
            pident_aln = float(parts[2])
            aln_len = int(parts[3])
            evalue = float(parts[10])
            bitscore = float(parts[11])
            target = target_to_seq[sid]
            generated = query_to_seq[qid]
            local_metrics = local_alignment_metrics(generated, target, aln_len, pident_aln)
            candidate = {
                "target": target,
                "blast_pident": pident_aln,
                "blast_aln_len": aln_len,
                "blast_evalue": evalue,
                "blast_bitscore": bitscore,
                "query_coverage": local_metrics["query_coverage"],
                "target_coverage": local_metrics["target_coverage"],
                "min_coverage": local_metrics["min_coverage"],
                "local_quality": local_metrics["local_quality"],
                "pident": pident_positionwise(generated, target),
                "levenshtein": levenshtein(generated, target),
                "length_diff": len(generated) - len(target),
            }
            key = (
                float(candidate["local_quality"]),
                bitscore,
                aln_len,
                pident_aln,
                -evalue,
                -int(candidate["levenshtein"]),
            )
            if qid not in best_by_query or key > best_by_query[qid]["_key"]:
                candidate["_key"] = key
                best_by_query[qid] = candidate

    for value in best_by_query.values():
        value.pop("_key", None)
    return best_by_query


def run_blast(
    rows: list[dict[str, str]],
    t2_by_rna1: dict[str, list[str]],
    blast_dir: str | Path,
    word_size: int,
    tmpdir: Path,
    evalue: str,
    max_target_seqs: int,
    max_hsps: int,
    timeout_seconds: int,
    num_threads: int,
) -> dict[str, dict[str, object]]:
    blastn = resolve_blast_tool(blast_dir, "blastn")
    makeblastdb = resolve_blast_tool(blast_dir, "makeblastdb")
    print(f"[BLAST{word_size}] blastn: {blastn}", flush=True)
    print(f"[BLAST{word_size}] makeblastdb: {makeblastdb}", flush=True)

    query_records: list[tuple[str, str]] = []
    query_to_seq: dict[str, str] = {}
    target_records: list[tuple[str, str]] = []
    target_to_rna1: dict[str, str] = {}
    target_to_seq: dict[str, str] = {}

    for row_index, row in enumerate(rows):
        qid = f"q{row_index}"
        generated = row.get("generated_seq", "")
        if generated:
            query_records.append((qid, generated))
            query_to_seq[qid] = generated
        rna1 = row["seq1"]
        for target_index, target in enumerate(t2_by_rna1.get(rna1, [])):
            tid = f"t{row_index}_{target_index}"
            target_records.append((tid, target))
            target_to_rna1[tid] = qid
            target_to_seq[tid] = target

    print(
        f"[BLAST{word_size}] record preparati: "
        f"query_non_vuote={len(query_records)} target={len(target_records)}",
        flush=True,
    )
    effective_max_target_seqs = max(max_target_seqs, len(target_records))
    if effective_max_target_seqs != max_target_seqs:
        print(
            f"[BLAST{word_size}] aumento max_target_seqs da {max_target_seqs} "
            f"a {effective_max_target_seqs}: in modalita global i target vengono "
            "filtrati per T2(RNA1) dopo BLAST, quindi non devono essere tagliati prima.",
            flush=True,
        )
    query_fasta = tmpdir / f"queries_w{word_size}.fa"
    target_fasta = tmpdir / f"targets_w{word_size}.fa"
    db_prefix = tmpdir / f"targets_w{word_size}_db"
    out_tsv = tmpdir / f"blast_w{word_size}.tsv"

    print(f"[BLAST{word_size}] scrivo FASTA: {query_fasta} / {target_fasta}", flush=True)
    write_fasta(query_fasta, query_records)
    write_fasta(target_fasta, target_records)

    print(f"[BLAST{word_size}] makeblastdb avvio", flush=True)
    subprocess.run(
        [makeblastdb, "-in", str(target_fasta), "-dbtype", "nucl", "-out", str(db_prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[BLAST{word_size}] makeblastdb fine", flush=True)
    outfmt = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen"
    print(f"[BLAST{word_size}] blastn avvio", flush=True)
    try:
        subprocess.run(
            [
                blastn,
                "-query", str(query_fasta),
                "-db", str(db_prefix),
                "-task", "blastn",
                "-word_size", str(word_size),
                "-evalue", str(evalue),
                "-dust", "no",
                "-soft_masking", "false",
                "-max_target_seqs", str(effective_max_target_seqs),
                "-max_hsps", str(max_hsps),
                "-num_threads", str(num_threads),
                "-outfmt", outfmt,
                "-out", str(out_tsv),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"BLAST word_size={word_size} ha superato il timeout di "
            f"{timeout_seconds}s in modalita global. Aumentare --blast_timeout "
            "oppure usare parametri piu' restrittivi."
        ) from exc
    print(f"[BLAST{word_size}] blastn fine: {out_tsv}", flush=True)

    print(f"[BLAST{word_size}] parsing risultati avvio", flush=True)
    best_by_query = parse_blast_tsv(out_tsv, query_to_seq, target_to_rna1, target_to_seq)
    print(f"[BLAST{word_size}] parsing risultati fine: hit={len(best_by_query)}/{len(rows)}", flush=True)
    return best_by_query


def run_blast_per_row(
    rows: list[dict[str, str]],
    t2_by_rna1: dict[str, list[str]],
    blast_dir: str | Path,
    word_size: int,
    tmpdir: Path,
    preview_nt: int,
    evalue: str,
    max_target_seqs: int,
    max_hsps: int,
    timeout_seconds: int,
    num_threads: int,
) -> dict[str, dict[str, object]]:
    blastn = resolve_blast_tool(blast_dir, "blastn")
    makeblastdb = resolve_blast_tool(blast_dir, "makeblastdb")
    print(f"[BLAST{word_size}] modalita per_row", flush=True)
    print(f"[BLAST{word_size}] blastn: {blastn}", flush=True)
    print(f"[BLAST{word_size}] makeblastdb: {makeblastdb}", flush=True)

    row_dir = tmpdir / f"blast_w{word_size}_per_row"
    row_dir.mkdir(parents=True, exist_ok=True)
    outfmt = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen"
    best_by_query: dict[str, dict[str, object]] = {}

    for row_index, row in enumerate(rows):
        qid = f"q{row_index}"
        rna1 = row["seq1"]
        generated = row.get("generated_seq", "")
        targets = t2_by_rna1.get(rna1, [])
        query_to_seq = {qid: generated} if generated else {}
        target_to_rna1: dict[str, str] = {}
        target_to_seq: dict[str, str] = {}
        target_records: list[tuple[str, str]] = []
        for target_index, target in enumerate(targets):
            tid = f"t{row_index}_{target_index}"
            target_records.append((tid, target))
            target_to_rna1[tid] = qid
            target_to_seq[tid] = target

        print(
            f"[BLAST{word_size}] riga {row_index + 1}/{len(rows)}: "
            f"len_generated={len(generated)} target_T2={len(target_records)}",
            flush=True,
        )
        if not generated or not target_records:
            print(f"[BLAST{word_size}] riga {row_index + 1}: salto, query o target vuoti", flush=True)
            continue
        if len(generated) < word_size:
            print(
                f"[BLAST{word_size}] riga {row_index + 1}: salto, "
                f"query piu' corta di word_size ({len(generated)} < {word_size})",
                flush=True,
            )
            continue

        query_fasta = row_dir / f"query_{row_index:04d}.fa"
        target_fasta = row_dir / f"targets_{row_index:04d}.fa"
        db_prefix = row_dir / f"targets_{row_index:04d}_db"
        out_tsv = row_dir / f"blast_{row_index:04d}.tsv"
        write_fasta(query_fasta, [(qid, generated)])
        write_fasta(target_fasta, target_records)

        print(f"[BLAST{word_size}] riga {row_index + 1}: makeblastdb avvio", flush=True)
        subprocess.run(
            [makeblastdb, "-in", str(target_fasta), "-dbtype", "nucl", "-out", str(db_prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[BLAST{word_size}] riga {row_index + 1}: makeblastdb fine", flush=True)
        print(f"[BLAST{word_size}] riga {row_index + 1}: blastn avvio", flush=True)
        try:
            subprocess.run(
                [
                    blastn,
                    "-query", str(query_fasta),
                    "-db", str(db_prefix),
                    "-task", "blastn",
                    "-word_size", str(word_size),
                    "-evalue", str(evalue),
                    "-dust", "no",
                    "-soft_masking", "false",
                    "-max_target_seqs", str(min(max_target_seqs, len(target_records))),
                    "-max_hsps", str(max_hsps),
                    "-num_threads", str(num_threads),
                    "-outfmt", outfmt,
                    "-out", str(out_tsv),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            print(
                f"[BLAST{word_size}] riga {row_index + 1}: TIMEOUT dopo "
                f"{timeout_seconds}s, salto la riga",
                flush=True,
            )
            continue
        print(f"[BLAST{word_size}] riga {row_index + 1}: blastn fine, parsing avvio", flush=True)
        hit = parse_blast_tsv(out_tsv, query_to_seq, target_to_rna1, target_to_seq).get(qid)
        if hit:
            best_by_query[qid] = hit
            print(f"[BLAST{word_size}] riga {row_index + 1}: hit trovato", flush=True)
            print_best_match(
                method=f"BLAST word_size={word_size}",
                row_number=row_index + 1,
                total_rows=len(rows),
                generated=generated,
                selected_target=str(hit.get("target", "")),
                pident=float(hit.get("pident", 0.0)),
                levenshtein_distance=int(hit.get("levenshtein", 0)),
                length_diff=int(hit.get("length_diff", 0)),
                preview_nt=preview_nt,
                extra=(
                    f"aln_len={hit.get('blast_aln_len', 0)} "
                    f"aln_pident={float(hit.get('blast_pident', 0.0)):.2f}% "
                    f"query_cov={float(hit.get('query_coverage', 0.0)):.2f}% "
                    f"target_cov={float(hit.get('target_coverage', 0.0)):.2f}% "
                    f"min_cov={float(hit.get('min_coverage', 0.0)):.2f}% "
                    f"local_quality={float(hit.get('local_quality', 0.0)):.2f} "
                    f"bitscore={float(hit.get('blast_bitscore', 0.0)):.2f}"
                ),
            )
        else:
            print(f"[BLAST{word_size}] riga {row_index + 1}: nessun hit", flush=True)

    print(f"[BLAST{word_size}] per_row fine: hit={len(best_by_query)}/{len(rows)}", flush=True)
    return best_by_query


def summarize(rows: list[dict[str, object]], prefix: str, metric_filter_col: str = "len_generated_seq") -> dict[str, object]:
    metric_rows = [row for row in rows if int(row.get(metric_filter_col, 0)) >= int(row.get("min_stop_nt", 20))]
    if prefix.startswith("blast"):
        metric_rows = [row for row in metric_rows if float(row.get(f"{prefix}_aln_len", 0) or 0) > 0]
    summary: dict[str, object] = {
        "method": prefix,
        "n_rows": len(rows),
        "n_metric_rows": len(metric_rows),
    }
    for field in [f"{prefix}_pident", f"{prefix}_levenshtein", f"{prefix}_length_diff"]:
        vals = [float(row[field]) for row in metric_rows if row.get(field) not in ("", None)]
        summary[f"mean_{field}"] = sum(vals) / len(vals) if vals else 0.0
    for field in (
        f"{prefix}_query_coverage",
        f"{prefix}_target_coverage",
        f"{prefix}_min_coverage",
        f"{prefix}_local_quality",
    ):
        vals = [float(row[field]) for row in metric_rows if row.get(field) not in ("", None)]
        summary[f"mean_{field}"] = sum(vals) / len(vals) if vals else 0.0
    if prefix == "sw":
        for field in ("sw_score", "sw_aln_len", "sw_aln_pident"):
            vals = [float(row[field]) for row in metric_rows if row.get(field) not in ("", None)]
            summary[f"mean_{field}"] = sum(vals) / len(vals) if vals else 0.0
    elif prefix.startswith("blast"):
        for field in (f"{prefix}_aln_len", f"{prefix}_aln_pident", f"{prefix}_bitscore"):
            vals = [float(row[field]) for row in metric_rows if row.get(field) not in ("", None)]
            summary[f"mean_{field}"] = sum(vals) / len(vals) if vals else 0.0
    return summary


def print_summary(summary: dict[str, object]) -> None:
    method = str(summary["method"])
    extra = ""
    if method == "sw":
        extra = (
            f" sw_score={summary['mean_sw_score']:.2f}"
            f" sw_aln_len={summary['mean_sw_aln_len']:.2f}"
            f" sw_aln_pident={summary['mean_sw_aln_pident']:.2f}%"
            f" qcov={summary['mean_sw_query_coverage']:.2f}%"
            f" tcov={summary['mean_sw_target_coverage']:.2f}%"
            f" mincov={summary['mean_sw_min_coverage']:.2f}%"
            f" local_quality={summary['mean_sw_local_quality']:.2f}"
        )
    elif method.startswith("blast"):
        extra = (
            f" aln_len={summary[f'mean_{method}_aln_len']:.2f}"
            f" aln_pident={summary[f'mean_{method}_aln_pident']:.2f}%"
            f" qcov={summary[f'mean_{method}_query_coverage']:.2f}%"
            f" tcov={summary[f'mean_{method}_target_coverage']:.2f}%"
            f" mincov={summary[f'mean_{method}_min_coverage']:.2f}%"
            f" local_quality={summary[f'mean_{method}_local_quality']:.2f}"
            f" bitscore={summary[f'mean_{method}_bitscore']:.2f}"
        )
    print(
        f"\nMetriche medie {method}: "
        f"n={summary['n_metric_rows']}/{summary['n_rows']} "
        f"pident={summary[f'mean_{method}_pident']:.2f}% "
        f"levenshtein={summary[f'mean_{method}_levenshtein']:.2f} "
        f"length_diff={summary[f'mean_{method}_length_diff']:.2f}"
        f"{extra}",
        flush=True,
    )


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def print_best_match(
    method: str,
    row_number: int,
    total_rows: int,
    generated: str,
    selected_target: str,
    pident: float,
    levenshtein_distance: int,
    length_diff: int,
    preview_nt: int,
    extra: str = "",
) -> None:
    print(f"\n--- {method} {row_number}/{total_rows} ---", flush=True)
    print(f"generated_seq : {preview_sequence(generated, preview_nt)}", flush=True)
    print(f"selected_T2   : {preview_sequence(selected_target, preview_nt)}", flush=True)
    print(
        "metriche      : "
        f"pident={pident:.2f}% "
        f"levenshtein={levenshtein_distance} "
        f"length_diff={length_diff}"
        + (f" {extra}" if extra else ""),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B4 ONLY: allinea le generazioni greedy di ffinal_generated_0.tsv "
            "a T2(RNA1) con Smith-Waterman e BLAST."
        )
    )
    parser.add_argument("--generations", default=DEFAULT_GENERATIONS)
    parser.add_argument("--t2", default=DEFAULT_T2)
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--blast_dir", default=DEFAULT_BLAST_DIR)
    parser.add_argument("--work_dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--skip_blast", action="store_true")
    parser.add_argument(
        "--keep_duplicate_rna1",
        action="store_true",
        help=(
            "Valuta tutte le coppie B4 compatibili. Senza questa opzione seleziona "
            "una sola generazione per RNA1: la piu' lunga osservata."
        ),
    )
    parser.add_argument(
        "--strict_rna1_coverage",
        action="store_true",
        help="Interrompe il run se il file B4 non copre tutti gli RNA1 presenti in --t2.",
    )
    parser.add_argument(
        "--min_generated_nt",
        type=int,
        default=0,
        help="Lunghezza minima inclusa nelle medie; per B4 il default corretto e' 0.",
    )
    parser.add_argument(
        "--blast_mode",
        choices=("per_row", "global"),
        default="global",
        help="global prepara FASTA/DB una volta ed e' il default per run completi; per_row e' solo debug su pochi esempi.",
    )
    parser.add_argument("--blast_evalue", default=DEFAULT_BLAST_EVALUE)
    parser.add_argument("--blast_max_target_seqs", type=int, default=DEFAULT_BLAST_MAX_TARGET_SEQS)
    parser.add_argument("--blast_max_hsps", type=int, default=DEFAULT_BLAST_MAX_HSPS)
    parser.add_argument("--blast_timeout", type=int, default=DEFAULT_BLAST_TIMEOUT_SECONDS)
    parser.add_argument("--blast_threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 = tutte le righe selezionate.")
    parser.add_argument("--preview_nt", type=int, default=0, help="Nucleotidi mostrati nei log; 0 = sequenza intera.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Script B4 ONLY: {Path(__file__).resolve()}", flush=True)
    print(f"Argomenti: {' '.join(sys.argv[1:]) if len(sys.argv) > 1 else '(nessuno)'}", flush=True)
    print(f"Default generations B4: {DEFAULT_GENERATIONS}", flush=True)
    t2_by_rna1 = load_t2(args.t2)
    source_generation_rows = load_generations(args.generations)
    generation_rows, selection_stats, missing_rna1 = select_b4_generation_rows(
        source_generation_rows,
        t2_by_rna1,
        args.keep_duplicate_rna1,
    )
    if args.strict_rna1_coverage and missing_rna1:
        preview = ", ".join(missing_rna1[:3])
        raise ValueError(
            f"Copertura B4 incompleta: {len(missing_rna1)}/{len(t2_by_rna1)} RNA1 mancanti. "
            f"Primi RNA1 mancanti: {preview}"
        )
    if not generation_rows:
        raise ValueError("Nessuna generazione B4 compatibile con il file T2 selezionato")
    collapse_stats = collapse_diagnostics(generation_rows)
    if args.limit > 0:
        generation_rows = generation_rows[:args.limit]

    print(f"Generazioni B4: {args.generations}", flush=True)
    print(f"T2: {args.t2}", flush=True)
    print(
        "Copertura B4: "
        f"{selection_stats['covered_rna1']}/{selection_stats['target_rna1']} RNA1; "
        f"mancanti={selection_stats['missing_rna1']}; "
        f"righe compatibili={selection_stats['eligible_generation_rows']}; "
        f"righe selezionate={selection_stats['selected_generation_rows']}",
        flush=True,
    )
    if missing_rna1:
        print(
            "ATTENZIONE: il riepilogo riguarda soltanto gli RNA1 coperti dal file B4. "
            "Non usare il totale del dataset T2 come denominatore.",
            flush=True,
        )
        print(f"Primi RNA1 mancanti: {', '.join(missing_rna1[:3])}", flush=True)
    print(
        "Diagnosi collasso B4: "
        f"generazioni_esatte={collapse_stats['unique_exact_generations']} "
        f"lunghezze={collapse_stats['unique_generated_lengths']} "
        f"traiettorie_massimali={collapse_stats['maximal_generation_trajectories']} "
        f"tutti_prefissi_stessa_traiettoria={collapse_stats['all_prefixes_of_one_trajectory']}",
        flush=True,
    )
    print(f"Righe analizzate: {len(generation_rows)}", flush=True)
    print("Log: stampa generated_seq e sequenza T2 scelta per ogni riga e per ogni metodo", flush=True)
    print("Smith-Waterman: avvio", flush=True)

    detail_rows: list[dict[str, object]] = []
    for index, row in enumerate(generation_rows):
        rna1 = row["seq1"]
        generated = row.get("generated_seq", "")
        targets = t2_by_rna1.get(rna1, [])
        if not targets:
            raise ValueError(f"RNA1 non trovato nel T2: riga {index + 1}")
        sw_best = best_smith_waterman(generated, targets)
        detail = {
            "row_index": index,
            "source_row_number": row.get("_source_row_number", ""),
            "seq1": rna1,
            "source_seq2": row.get("source_seq2", ""),
            "generated_seq": generated,
            "len_generated_seq": len(generated),
            "n_valid_seq2": len(targets),
            "min_stop_nt": args.min_generated_nt,
            "sw_best_seq2": sw_best.get("target", ""),
            "sw_score": sw_best.get("sw_score", 0),
            "sw_aln_len": sw_best.get("sw_aln_len", 0),
            "sw_aln_pident": sw_best.get("sw_pident", 0.0),
            "sw_query_coverage": sw_best.get("query_coverage", 0.0),
            "sw_target_coverage": sw_best.get("target_coverage", 0.0),
            "sw_min_coverage": sw_best.get("min_coverage", 0.0),
            "sw_local_quality": sw_best.get("local_quality", 0.0),
            "sw_pident": sw_best.get("pident", 0.0),
            "sw_levenshtein": sw_best.get("levenshtein", 0),
            "sw_length_diff": sw_best.get("length_diff", 0),
        }
        detail_rows.append(detail)
        print_best_match(
            method="Smith-Waterman",
            row_number=index + 1,
            total_rows=len(generation_rows),
            generated=generated,
            selected_target=str(detail["sw_best_seq2"]),
            pident=float(detail["sw_pident"]),
            levenshtein_distance=int(detail["sw_levenshtein"]),
            length_diff=int(detail["sw_length_diff"]),
            preview_nt=args.preview_nt,
            extra=(
                f"sw_score={detail['sw_score']} "
                f"sw_aln_len={detail['sw_aln_len']} "
                f"sw_aln_pident={float(detail['sw_aln_pident']):.2f}% "
                f"query_cov={float(detail['sw_query_coverage']):.2f}% "
                f"target_cov={float(detail['sw_target_coverage']):.2f}% "
                f"min_cov={float(detail['sw_min_coverage']):.2f}% "
                f"local_quality={float(detail['sw_local_quality']):.2f}"
            ),
        )
        if (index + 1) % 25 == 0 or index + 1 == len(generation_rows):
            print(f"[SW] {index + 1}/{len(generation_rows)}", flush=True)

    sw_summary = summarize(detail_rows, "sw")
    print_summary(sw_summary)

    if not args.skip_blast:
        tmpdir = Path(args.work_dir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        for word_size in (4, 6):
            print(f"BLAST word_size={word_size}: avvio", flush=True)
            if args.blast_mode == "per_row":
                best = run_blast_per_row(
                    generation_rows,
                    t2_by_rna1,
                    args.blast_dir,
                    word_size,
                    tmpdir,
                    args.preview_nt,
                    args.blast_evalue,
                    args.blast_max_target_seqs,
                    args.blast_max_hsps,
                    args.blast_timeout,
                    args.blast_threads,
                )
            else:
                best = run_blast(
                    generation_rows,
                    t2_by_rna1,
                    args.blast_dir,
                    word_size,
                    tmpdir,
                    args.blast_evalue,
                    args.blast_max_target_seqs,
                    args.blast_max_hsps,
                    args.blast_timeout,
                    args.blast_threads,
                )
            for row_index, detail in enumerate(detail_rows):
                hit = best.get(f"q{row_index}", {})
                prefix = f"blast{word_size}"
                has_hit = bool(hit)
                detail[f"{prefix}_best_seq2"] = hit.get("target", "")
                detail[f"{prefix}_hit"] = has_hit
                detail[f"{prefix}_aln_pident"] = hit.get("blast_pident", 0.0)
                detail[f"{prefix}_aln_len"] = hit.get("blast_aln_len", 0)
                detail[f"{prefix}_bitscore"] = hit.get("blast_bitscore", 0.0)
                detail[f"{prefix}_evalue"] = hit.get("blast_evalue", "")
                detail[f"{prefix}_query_coverage"] = hit.get("query_coverage", 0.0)
                detail[f"{prefix}_target_coverage"] = hit.get("target_coverage", 0.0)
                detail[f"{prefix}_min_coverage"] = hit.get("min_coverage", 0.0)
                detail[f"{prefix}_local_quality"] = hit.get("local_quality", 0.0)
                detail[f"{prefix}_pident"] = hit.get("pident", 0.0)
                detail[f"{prefix}_levenshtein"] = hit.get("levenshtein", 0)
                detail[f"{prefix}_length_diff"] = hit.get("length_diff", 0)
                if args.blast_mode == "global":
                    if has_hit:
                        print_best_match(
                            method=f"BLAST word_size={word_size}",
                            row_number=row_index + 1,
                            total_rows=len(generation_rows),
                            generated=str(detail["generated_seq"]),
                            selected_target=str(detail[f"{prefix}_best_seq2"]),
                            pident=float(detail[f"{prefix}_pident"]),
                            levenshtein_distance=int(detail[f"{prefix}_levenshtein"]),
                            length_diff=int(detail[f"{prefix}_length_diff"]),
                            preview_nt=args.preview_nt,
                            extra=(
                                f"aln_len={detail[f'{prefix}_aln_len']} "
                                f"aln_pident={float(detail[f'{prefix}_aln_pident']):.2f}% "
                                f"query_cov={float(detail[f'{prefix}_query_coverage']):.2f}% "
                                f"target_cov={float(detail[f'{prefix}_target_coverage']):.2f}% "
                                f"min_cov={float(detail[f'{prefix}_min_coverage']):.2f}% "
                                f"local_quality={float(detail[f'{prefix}_local_quality']):.2f} "
                                f"bitscore={float(detail[f'{prefix}_bitscore']):.2f}"
                            ),
                        )
                    else:
                        print(f"\n--- BLAST word_size={word_size} {row_index + 1}/{len(generation_rows)} ---", flush=True)
                        print(f"generated_seq : {preview_sequence(str(detail['generated_seq']), args.preview_nt)}", flush=True)
                        print("selected_T2   : [NESSUN HIT BLAST DENTRO T2(RNA1)]", flush=True)
                        print(
                            f"fallback_SW_T2: {preview_sequence(str(detail['sw_best_seq2']), args.preview_nt)}",
                            flush=True,
                        )
                        print(
                            "metriche      : "
                            "blast_hit=False "
                            f"sw_pident={float(detail['sw_pident']):.2f}% "
                            f"sw_lev={int(detail['sw_levenshtein'])} "
                            f"sw_score={detail['sw_score']}",
                            flush=True,
                        )
            print(f"BLAST word_size={word_size}: hit per {len(best)}/{len(generation_rows)} righe", flush=True)
            print_summary(summarize(detail_rows, f"blast{word_size}"))

    summaries = [sw_summary]
    if not args.skip_blast:
        summaries.append(summarize(detail_rows, "blast4"))
        summaries.append(summarize(detail_rows, "blast6"))

    run_metadata: dict[str, object] = {
        "b4_only": True,
        "generation_file": str(args.generations),
        "t2_file": str(args.t2),
        "keep_duplicate_rna1": args.keep_duplicate_rna1,
        "min_generated_nt": args.min_generated_nt,
        "analyzed_rows": len(generation_rows),
        **{f"selection_{key}": value for key, value in selection_stats.items()},
        **{f"collapse_{key}": value for key, value in collapse_stats.items()},
    }
    for summary in summaries:
        summary.update(run_metadata)

    detail_path = Path(f"{args.output_prefix}_details.tsv")
    summary_tsv = Path(f"{args.output_prefix}_summary.tsv")
    summary_json = Path(f"{args.output_prefix}_summary.json")
    missing_rna1_path = Path(f"{args.output_prefix}_missing_rna1.tsv")
    write_tsv(detail_path, detail_rows)
    write_tsv(summary_tsv, summaries)
    summary_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if missing_rna1:
        write_tsv(
            missing_rna1_path,
            [
                {
                    "seq1": rna1,
                    "reason": "nessuna coppia RNA1-RNA2 esatta condivisa tra generazioni B4 e T2",
                }
                for rna1 in missing_rna1
            ],
        )

    print(
        f"\nMetriche medie B4 su generazioni con len >= {args.min_generated_nt} nt:",
        flush=True,
    )
    for summary in summaries:
        method = summary["method"]
        print(
            f"- {method}: "
            f"pident={summary[f'mean_{method}_pident']:.2f}% "
            f"lev={summary[f'mean_{method}_levenshtein']:.2f} "
            f"length_diff={summary[f'mean_{method}_length_diff']:.2f} "
            f"qcov={summary[f'mean_{method}_query_coverage']:.2f}% "
            f"tcov={summary[f'mean_{method}_target_coverage']:.2f}% "
            f"mincov={summary[f'mean_{method}_min_coverage']:.2f}% "
            f"local_quality={summary[f'mean_{method}_local_quality']:.2f} "
            f"n={summary['n_metric_rows']}",
            flush=True,
        )
    print("\nFile scritti:", flush=True)
    print(f"- {detail_path}", flush=True)
    print(f"- {summary_tsv}", flush=True)
    print(f"- {summary_json}", flush=True)
    if missing_rna1:
        print(f"- {missing_rna1_path}", flush=True)


if __name__ == "__main__":
    main()
