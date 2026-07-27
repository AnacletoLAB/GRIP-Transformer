from __future__ import annotations

"""Confronto appaiato tra R del modello B4 e R del controllo casuale."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DEFAULT_MODEL_BEST = "B4_only_lmax_topk2_T2_R_best.tsv"
DEFAULT_RANDOM_BEST = "B4_only_lmax_topk2_random_seed42_T2_R_best.tsv"
DEFAULT_OUTPUT_PREFIX = "B4_only_lmax_topk2_model_vs_random_ttest"

DETAIL_FIELDS = [
    "row_index",
    "seq1",
    "lmax",
    "n_T2",
    "model_generated_seq",
    "random_generated_seq",
    "model_sw_R",
    "random_sw_R",
    "difference_sw_R",
    "model_blast6_hit",
    "random_blast6_hit",
    "model_blast6_R_no_hit_zero",
    "random_blast6_R_no_hit_zero",
    "difference_blast6_R_no_hit_zero",
]

SUMMARY_FIELDS = [
    "method",
    "analysis",
    "alternative",
    "no_hit_handling",
    "n_pairs",
    "df",
    "model_mean_R",
    "random_mean_R",
    "mean_paired_difference",
    "sd_paired_difference",
    "se_paired_difference",
    "ci95_difference_low",
    "ci95_difference_high",
    "t_statistic",
    "p_value_two_sided",
    "cohen_dz",
    "model_greater_count",
    "equal_count",
    "random_greater_count",
    "model_hit_count",
    "random_hit_count",
    "both_hit_count",
]


def preview_sequence(sequence: str, max_nt: int) -> str:
    if max_nt <= 0 or len(sequence) <= max_nt:
        return sequence
    return sequence[:max_nt] + f"...[{len(sequence)} nt]"


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"File vuoto: {path}")
    return rows


def write_tsv(
    path: str | Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Frazione continua per la beta incompleta (Numerical Recipes)."""
    max_iterations = 300
    epsilon = 3.0e-14
    fp_min = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fp_min:
        d = fp_min
    d = 1.0 / d
    result = d

    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fp_min:
            d = fp_min
        c = 1.0 + aa / c
        if abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        result *= d * c

        aa = -(a + iteration) * (qab + iteration) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1.0 + aa * d
        if abs(d) < fp_min:
            d = fp_min
        c = 1.0 + aa / c
        if abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result

    raise RuntimeError("La frazione continua della beta incompleta non converge")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError("x deve essere compreso tra 0 e 1")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_factor = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    factor = math.exp(log_factor)
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_sided_p_value(t_statistic: float, df: int) -> float:
    if df < 1:
        raise ValueError("Servono almeno due coppie per il t-test")
    if math.isnan(t_statistic):
        return math.nan
    if math.isinf(t_statistic):
        return 0.0
    absolute_t = abs(t_statistic)
    if absolute_t == 0.0:
        return 1.0
    x = df / (df + absolute_t * absolute_t)
    return min(1.0, max(0.0, regularized_incomplete_beta(df / 2.0, 0.5, x)))


def student_t_critical_two_sided(alpha: float, df: int) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha deve essere compreso tra 0 e 1")
    low = 0.0
    high = 1.0
    while student_t_two_sided_p_value(high, df) > alpha:
        high *= 2.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if student_t_two_sided_p_value(midpoint, df) > alpha:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def paired_t_test(
    model_values: list[float],
    random_values: list[float],
) -> dict[str, float | int]:
    if len(model_values) != len(random_values):
        raise ValueError("Le due serie appaiate hanno lunghezze diverse")
    n_pairs = len(model_values)
    if n_pairs < 2:
        raise ValueError("Servono almeno due coppie per il t-test")

    differences = [
        model_value - random_value
        for model_value, random_value in zip(model_values, random_values)
    ]
    mean_difference = statistics.fmean(differences)
    sd_difference = statistics.stdev(differences)
    se_difference = sd_difference / math.sqrt(n_pairs)
    if se_difference == 0.0:
        if mean_difference == 0.0:
            t_statistic = 0.0
            p_value = 1.0
        else:
            t_statistic = math.copysign(math.inf, mean_difference)
            p_value = 0.0
    else:
        t_statistic = mean_difference / se_difference
        p_value = student_t_two_sided_p_value(t_statistic, n_pairs - 1)

    t_critical = student_t_critical_two_sided(0.05, n_pairs - 1)
    margin = t_critical * se_difference
    cohen_dz = (
        mean_difference / sd_difference
        if sd_difference > 0.0
        else math.copysign(math.inf, mean_difference)
        if mean_difference != 0.0
        else 0.0
    )
    return {
        "n_pairs": n_pairs,
        "df": n_pairs - 1,
        "model_mean_R": statistics.fmean(model_values),
        "random_mean_R": statistics.fmean(random_values),
        "mean_paired_difference": mean_difference,
        "sd_paired_difference": sd_difference,
        "se_paired_difference": se_difference,
        "ci95_difference_low": mean_difference - margin,
        "ci95_difference_high": mean_difference + margin,
        "t_statistic": t_statistic,
        "p_value_two_sided": p_value,
        "cohen_dz": cohen_dz,
        "model_greater_count": sum(value > 0.0 for value in differences),
        "equal_count": sum(value == 0.0 for value in differences),
        "random_greater_count": sum(value < 0.0 for value in differences),
    }


def validate_and_pair(
    model_rows: list[dict[str, str]],
    random_rows: list[dict[str, str]],
) -> list[tuple[dict[str, str], dict[str, str]]]:
    required_fields = {
        "seq1",
        "generated_seq",
        "lmax",
        "n_T2",
        "sw_R",
        "blast6_hit",
        "blast6_R",
    }
    for label, rows in (("modello", model_rows), ("casuale", random_rows)):
        missing = required_fields - set(rows[0])
        if missing:
            raise ValueError(
                f"File {label}: colonne mancanti: {', '.join(sorted(missing))}"
            )
        keys = [row["seq1"].strip() for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"File {label}: RNA1 duplicati")

    if len(model_rows) != len(random_rows):
        raise ValueError(
            f"Numero righe diverso: modello={len(model_rows)}, casuale={len(random_rows)}"
        )
    random_by_rna1 = {row["seq1"].strip(): row for row in random_rows}
    if {row["seq1"].strip() for row in model_rows} != set(random_by_rna1):
        raise ValueError("Gli insiemi di RNA1 del modello e del controllo non coincidono")

    paired_rows: list[tuple[dict[str, str], dict[str, str]]] = []
    for row_number, model_row in enumerate(model_rows, start=2):
        rna1 = model_row["seq1"].strip()
        random_row = random_by_rna1[rna1]
        if int(model_row["lmax"]) != int(random_row["lmax"]):
            raise ValueError(f"Riga {row_number}: lmax diverso per lo stesso RNA1")
        if int(model_row["n_T2"]) != int(random_row["n_T2"]):
            raise ValueError(f"Riga {row_number}: n_T2 diverso per lo stesso RNA1")
        if len(model_row["generated_seq"].strip()) != int(model_row["lmax"]):
            raise ValueError(f"Riga {row_number}: sequenza modello non lunga lmax")
        if len(random_row["generated_seq"].strip()) != int(random_row["lmax"]):
            raise ValueError(f"Riga {row_number}: sequenza casuale non lunga lmax")
        paired_rows.append((model_row, random_row))
    return paired_rows


def build_summary_row(
    method: str,
    analysis: str,
    no_hit_handling: str,
    model_values: list[float],
    random_values: list[float],
    model_hits: list[bool],
    random_hits: list[bool],
) -> dict[str, object]:
    result: dict[str, object] = {
        "method": method,
        "analysis": analysis,
        "alternative": "two-sided",
        "no_hit_handling": no_hit_handling,
    }
    result.update(paired_t_test(model_values, random_values))
    result.update(
        {
            "model_hit_count": sum(model_hits),
            "random_hit_count": sum(random_hits),
            "both_hit_count": sum(
                model_hit and random_hit
                for model_hit, random_hit in zip(model_hits, random_hits)
            ),
        }
    )
    return result


def format_p_value(p_value: float) -> str:
    if p_value == 0.0:
        return "<1e-300"
    if p_value < 0.001:
        return f"{p_value:.3e}"
    return f"{p_value:.6f}"


def paper_sentence(summary: dict[str, object]) -> str:
    method = str(summary["method"])
    no_hit_note = (
        " with no-hit cases encoded as R=0"
        if summary["no_hit_handling"] == "R=0"
        else ""
    )
    return (
        f"For {method}{no_hit_note}, the mean best R was "
        f"{float(summary['model_mean_R']):.6f} for model-generated sequences "
        f"and {float(summary['random_mean_R']):.6f} for length-matched random "
        f"sequences (mean paired difference "
        f"{float(summary['mean_paired_difference']):.6f}, 95% CI "
        f"[{float(summary['ci95_difference_low']):.6f}, "
        f"{float(summary['ci95_difference_high']):.6f}]; paired two-sided "
        f"t-test, t({int(summary['df'])})="
        f"{float(summary['t_statistic']):.3f}, "
        f"p={format_p_value(float(summary['p_value_two_sided']))})."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Confronta, per lo stesso RNA1, il migliore R del modello B4 e del "
            "controllo casuale mediante t-test appaiato a due code."
        )
    )
    parser.add_argument("--model_best", default=DEFAULT_MODEL_BEST)
    parser.add_argument("--random_best", default=DEFAULT_RANDOM_BEST)
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--limit", type=int, default=0, help="0 = tutte le coppie")
    parser.add_argument(
        "--preview_nt",
        type=int,
        default=0,
        help="0 = stampa le sequenze complete; N > 0 = mostra solo i primi N nt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_rows = load_tsv(args.model_best)
    random_rows = load_tsv(args.random_best)
    if args.limit > 0:
        model_rows = model_rows[: args.limit]
        random_rows = random_rows[: args.limit]
    paired_rows = validate_and_pair(model_rows, random_rows)

    print(f"Best R modello: {args.model_best}", flush=True)
    print(f"Best R controllo casuale: {args.random_best}", flush=True)
    print(f"Coppie RNA1: {len(paired_rows)}", flush=True)
    print("Test: t-test appaiato a due code", flush=True)
    print(
        "BLAST6 primario: tutti gli RNA1, con R=0 quando non esiste alcun hit",
        flush=True,
    )

    details: list[dict[str, object]] = []
    model_sw: list[float] = []
    random_sw: list[float] = []
    model_blast: list[float] = []
    random_blast: list[float] = []
    model_hits: list[bool] = []
    random_hits: list[bool] = []

    for row_index, (model_row, random_row) in enumerate(paired_rows):
        model_sw_r = float(model_row["sw_R"])
        random_sw_r = float(random_row["sw_R"])
        model_blast_hit = parse_bool(model_row["blast6_hit"])
        random_blast_hit = parse_bool(random_row["blast6_hit"])
        model_blast_r = float(model_row["blast6_R"]) if model_blast_hit else 0.0
        random_blast_r = float(random_row["blast6_R"]) if random_blast_hit else 0.0

        model_sw.append(model_sw_r)
        random_sw.append(random_sw_r)
        model_blast.append(model_blast_r)
        random_blast.append(random_blast_r)
        model_hits.append(model_blast_hit)
        random_hits.append(random_blast_hit)

        detail = {
            "row_index": row_index,
            "seq1": model_row["seq1"].strip(),
            "lmax": int(model_row["lmax"]),
            "n_T2": int(model_row["n_T2"]),
            "model_generated_seq": model_row["generated_seq"].strip(),
            "random_generated_seq": random_row["generated_seq"].strip(),
            "model_sw_R": model_sw_r,
            "random_sw_R": random_sw_r,
            "difference_sw_R": model_sw_r - random_sw_r,
            "model_blast6_hit": model_blast_hit,
            "random_blast6_hit": random_blast_hit,
            "model_blast6_R_no_hit_zero": model_blast_r,
            "random_blast6_R_no_hit_zero": random_blast_r,
            "difference_blast6_R_no_hit_zero": model_blast_r - random_blast_r,
        }
        details.append(detail)

        print(
            f"\n--- Confronto appaiato RNA1 {row_index + 1}/{len(paired_rows)} ---",
            flush=True,
        )
        print(
            f"RNA1: {preview_sequence(str(detail['seq1']), args.preview_nt)}",
            flush=True,
        )
        print(
            "sequenza_modello: "
            f"{preview_sequence(str(detail['model_generated_seq']), args.preview_nt)}",
            flush=True,
        )
        print(
            "sequenza_casuale: "
            f"{preview_sequence(str(detail['random_generated_seq']), args.preview_nt)}",
            flush=True,
        )
        print(
            f"lmax={detail['lmax']} n_T2={detail['n_T2']} | "
            f"SW: modello={model_sw_r:.6f} casuale={random_sw_r:.6f} "
            f"diff={model_sw_r - random_sw_r:+.6f} | "
            f"BLAST6 (no-hit=0): modello={model_blast_r:.6f} "
            f"casuale={random_blast_r:.6f} "
            f"diff={model_blast_r - random_blast_r:+.6f} "
            f"hit_modello={model_blast_hit} hit_casuale={random_blast_hit}",
            flush=True,
        )

    summaries = [
        build_summary_row(
            method="Smith-Waterman",
            analysis="primary_all_RNA1",
            no_hit_handling="not_applicable",
            model_values=model_sw,
            random_values=random_sw,
            model_hits=[True] * len(model_sw),
            random_hits=[True] * len(random_sw),
        ),
        build_summary_row(
            method="BLAST6",
            analysis="primary_all_RNA1",
            no_hit_handling="R=0",
            model_values=model_blast,
            random_values=random_blast,
            model_hits=model_hits,
            random_hits=random_hits,
        ),
    ]

    both_hit_indices = [
        index
        for index, (model_hit, random_hit) in enumerate(zip(model_hits, random_hits))
        if model_hit and random_hit
    ]
    if len(both_hit_indices) >= 2:
        summaries.append(
            build_summary_row(
                method="BLAST6",
                analysis="sensitivity_both_hit_pairs",
                no_hit_handling="excluded_unless_both_hit",
                model_values=[model_blast[index] for index in both_hit_indices],
                random_values=[random_blast[index] for index in both_hit_indices],
                model_hits=[True] * len(both_hit_indices),
                random_hits=[True] * len(both_hit_indices),
            )
        )

    details_path = f"{args.output_prefix}_details.tsv"
    summary_path = f"{args.output_prefix}_summary.tsv"
    json_path = f"{args.output_prefix}_summary.json"
    paper_text_path = f"{args.output_prefix}_paper_text.txt"
    write_tsv(details_path, details, DETAIL_FIELDS)
    write_tsv(summary_path, summaries, SUMMARY_FIELDS)
    Path(json_path).write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    primary_sentences = "\n\n".join(paper_sentence(row) for row in summaries[:2])
    Path(paper_text_path).write_text(primary_sentences + "\n", encoding="utf-8")

    print("\nRisultati finali t-test appaiato:", flush=True)
    for summary in summaries:
        print(
            f"- {summary['method']} [{summary['analysis']}]: "
            f"n={summary['n_pairs']} "
            f"media_modello={float(summary['model_mean_R']):.6f} "
            f"media_casuale={float(summary['random_mean_R']):.6f} "
            f"diff={float(summary['mean_paired_difference']):+.6f} "
            f"t({summary['df']})={float(summary['t_statistic']):.6f} "
            f"p={format_p_value(float(summary['p_value_two_sided']))} "
            f"dz={float(summary['cohen_dz']):.6f}",
            flush=True,
        )
    print(f"- Dettagli: {details_path}", flush=True)
    print(f"- Summary TSV: {summary_path}", flush=True)
    print(f"- Summary JSON: {json_path}", flush=True)
    print(f"- Testo pronto per il paper: {paper_text_path}", flush=True)


if __name__ == "__main__":
    main()
