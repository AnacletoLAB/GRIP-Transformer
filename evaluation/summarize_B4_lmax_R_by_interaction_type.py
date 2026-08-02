from __future__ import annotations

"""Raggruppa la valutazione B4 lmax top-k=2 nelle classi del test filtrato.

Lo script non carica il modello, non rigenera sequenze e non ricalcola gli
allineamenti. Legge il TSV dettagliato gia' prodotto da
evaluate_B4_only_lmax_R_smith_waterman_blast6.py, associa Category_x e
Category_y tramite la coppia esatta (seq1, xprime), conserva esclusivamente
le coppie presenti nel file di test filtrato, unisce le direzioni reciproche
e produce:

1. un dettaglio arricchito, una riga per ogni confronto z-x';
2. il best match per ogni RNA1 e classe non direzionale;
3. un summary principale delle 14 classi presenti nel test filtrato;
4. un summary di controllo con le 17 direzioni presenti nel test filtrato.

Tutti i controlli vengono completati prima di scrivere gli output.
"""

import argparse
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Lo script e' pubblicato in ``evaluation/``. I default sono ancorati alla
# root del repository, cosi' il comportamento non cambia in base alla cwd.
DEFAULT_DETAILS = PROJECT_ROOT / "B4_only_lmax_topk2_T2_R_details.tsv"
DEFAULT_TRAIN_PICKLE = (
    PROJECT_ROOT
    / "data"
    / "dry_run"
    / "Dataset_originali_pkl"
    / "all_train_df_no_augmentation.pkl"
)
DEFAULT_TEST_PICKLE = (
    PROJECT_ROOT
    / "data"
    / "dry_run"
    / "Dataset_originali_pkl"
    / "all_test_df_no_augmentation.pkl"
)
DEFAULT_TEST_PAIRS = (
    PROJECT_ROOT
    / "data"
    / "dry_run"
    / "test_filtered_no_aug_pident95_qcov90.tsv"
)
DEFAULT_HEATMAP_COUNTS = (
    PROJECT_ROOT / "heatmap_interaction_types_nonaug_pkl_split_counts.tsv"
)
DEFAULT_OUTPUT_PREFIX = (
    PROJECT_ROOT / "B4_only_lmax_topk2_test_pairs_R_by_interaction_type"
)

DEFAULT_EXPECTED_DETAILS_ROWS = 13_574
DEFAULT_EXPECTED_RNA1 = 697
DEFAULT_EXPECTED_SOURCE_ROWS = 23_557
DEFAULT_EXPECTED_TEST_SOURCE_ROWS = 1_544
DEFAULT_EXPECTED_UNIQUE_TEST_PAIRS = 1_533
DEFAULT_EXPECTED_TYPES = 14
DEFAULT_EXPECTED_DIRECTIONAL_TYPES = 17
DEFAULT_EXPECTED_REFERENCE_TYPES = 15
DEFAULT_EXPECTED_REFERENCE_DIRECTIONAL_TYPES = 18

DETAIL_REQUIRED_COLUMNS = [
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

DETAIL_INTEGER_COLUMNS = [
    "row_index",
    "len_generated_seq",
    "lmax",
    "n_T2",
    "target_index",
    "len_xprime",
    "sw_matches",
    "sw_aln_len",
    "sw_score",
    "blast6_nident",
    "blast6_aln_len",
]

DETAIL_FLOAT_COLUMNS = [
    "sw_aln_pident",
    "sw_R",
    "blast6_aln_pident",
    "blast6_bitscore",
    "blast6_R",
]

CATEGORY_REQUIRED_COLUMNS = [
    "RNA_sequence_x",
    "RNA_sequence_y",
    "Category_x",
    "Category_y",
    "Category_Couple",
]

HEATMAP_REQUIRED_COLUMNS = [
    "Category_x",
    "Category_y",
    "train",
    "test",
    "total",
]

SUMMARY_COLUMNS = [
    "type_order",
    "interaction_type",
    "category_mode",
    "heatmap_train_pairs",
    "heatmap_test_pairs",
    "heatmap_total_pairs",
    "n_directional_subtypes",
    "directional_subtypes",
    "n_pair_comparisons",
    "n_RNA1",
    "sw_pair_mean_R",
    "sw_pair_median_R",
    "sw_best_mean_R",
    "sw_best_median_R",
    "sw_best_min_R",
    "sw_best_max_R",
    "blast6_pair_hit_count",
    "blast6_pair_no_hit_count",
    "blast6_pair_hit_pct",
    "blast6_pair_mean_R_hits",
    "blast6_pair_median_R_hits",
    "blast6_pair_mean_R_no_hit_zero",
    "blast6_best_RNA1_hit_count",
    "blast6_best_RNA1_no_hit_count",
    "blast6_best_RNA1_hit_pct",
    "blast6_best_mean_R_hits",
    "blast6_best_median_R_hits",
    "blast6_best_min_R_hits",
    "blast6_best_max_R_hits",
    "blast6_best_mean_R_no_hit_zero",
]


@dataclass(frozen=True)
class OutputPaths:
    details: Path
    best_per_rna1_type: Path
    summary_types: Path
    summary_directional_types: Path


def require_columns(
    frame: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label}: colonne mancanti: {missing}")


def require_expected_count(
    actual: int,
    expected: int,
    label: str,
) -> None:
    if expected > 0 and actual != expected:
        raise ValueError(f"{label}: trovato {actual}, atteso {expected}")


def clean_text_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if result[column].isna().any():
            count = int(result[column].isna().sum())
            raise ValueError(f"{label}: {count} valori nulli in {column}")
        result[column] = result[column].astype(str).str.strip()
        empty = result[column].eq("")
        if empty.any():
            raise ValueError(
                f"{label}: {int(empty.sum())} valori vuoti in {column}"
            )
    return result


def canonical_interaction_key(category_x: str, category_y: str) -> str:
    ordered = sorted(
        (category_x.strip(), category_y.strip()),
        key=lambda value: (value.casefold(), value),
    )
    return "\x1f".join(ordered)


def parse_boolean_series(series: pd.Series, label: str) -> pd.Series:
    true_values = {"true", "1", "yes"}
    false_values = {"false", "0", "no"}

    def parse_one(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in true_values:
            return True
        if normalized in false_values:
            return False
        raise ValueError(f"{label}: valore booleano non riconosciuto: {value!r}")

    return series.map(parse_one).astype(bool)


def convert_numeric_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except Exception as exc:
            raise ValueError(
                f"{label}: impossibile convertire {column} in numerico"
            ) from exc
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{label}: valori non finiti in {column}")
    return result


def load_and_validate_details(
    path: str | Path,
    expected_rows: int,
    expected_rna1: int,
) -> tuple[pd.DataFrame, list[str]]:
    details_path = Path(path)
    if not details_path.is_file():
        raise FileNotFoundError(f"File dettagli non trovato: {details_path}")

    details = pd.read_csv(
        details_path,
        sep="\t",
        keep_default_na=False,
        dtype={"seq1": str, "generated_seq": str, "xprime": str},
    )
    original_columns = details.columns.tolist()
    require_columns(details, DETAIL_REQUIRED_COLUMNS, "Dettagli")
    require_expected_count(len(details), expected_rows, "Righe dettagli")

    details = clean_text_columns(
        details,
        ["seq1", "generated_seq", "xprime"],
        "Dettagli",
    )
    details = convert_numeric_columns(
        details,
        DETAIL_INTEGER_COLUMNS + DETAIL_FLOAT_COLUMNS,
        "Dettagli",
    )
    details["blast6_hit"] = parse_boolean_series(
        details["blast6_hit"],
        "Dettagli blast6_hit",
    )

    for column in DETAIL_INTEGER_COLUMNS:
        values = details[column].to_numpy(dtype=float)
        if not np.allclose(values, np.rint(values), rtol=0.0, atol=0.0):
            raise ValueError(f"Dettagli: valori non interi in {column}")
        details[column] = details[column].astype("int64")

    if details.duplicated(["row_index", "target_index"]).any():
        duplicates = int(
            details.duplicated(["row_index", "target_index"], keep=False).sum()
        )
        raise ValueError(
            f"Dettagli: {duplicates} righe duplicate per row_index/target_index"
        )

    unique_rna1 = int(details["seq1"].nunique())
    unique_row_indices = int(details["row_index"].nunique())
    require_expected_count(unique_rna1, expected_rna1, "RNA1 distinti")
    if unique_row_indices != unique_rna1:
        raise ValueError(
            "Dettagli: row_index distinti e RNA1 distinti non coincidono "
            f"({unique_row_indices} vs {unique_rna1})"
        )

    group_checks = details.groupby("row_index", sort=False).agg(
        n_rows=("target_index", "size"),
        n_T2_min=("n_T2", "min"),
        n_T2_max=("n_T2", "max"),
        n_seq1=("seq1", "nunique"),
        n_generated=("generated_seq", "nunique"),
        n_lmax=("lmax", "nunique"),
    )
    invalid_group = (
        (group_checks["n_T2_min"] != group_checks["n_T2_max"])
        | (group_checks["n_rows"] != group_checks["n_T2_min"])
        | (group_checks["n_seq1"] != 1)
        | (group_checks["n_generated"] != 1)
        | (group_checks["n_lmax"] != 1)
    )
    if invalid_group.any():
        raise ValueError(
            "Dettagli: struttura incoerente in "
            f"{int(invalid_group.sum())} gruppi row_index"
        )

    for row_index, group in details.groupby("row_index", sort=False):
        expected_indices = np.arange(len(group), dtype=np.int64)
        actual_indices = np.sort(group["target_index"].to_numpy(dtype=np.int64))
        if not np.array_equal(actual_indices, expected_indices):
            raise ValueError(
                f"Dettagli: target_index non contigui per row_index={row_index}"
            )

    actual_target_lengths = details["xprime"].str.len().to_numpy(dtype=np.int64)
    if not np.array_equal(
        actual_target_lengths,
        details["len_xprime"].to_numpy(dtype=np.int64),
    ):
        count = int(
            (
                actual_target_lengths
                != details["len_xprime"].to_numpy(dtype=np.int64)
            ).sum()
        )
        raise ValueError(f"Dettagli: {count} len_xprime incoerenti")

    actual_generated_lengths = (
        details["generated_seq"].str.len().to_numpy(dtype=np.int64)
    )
    if not np.array_equal(
        actual_generated_lengths,
        details["len_generated_seq"].to_numpy(dtype=np.int64),
    ):
        count = int(
            (
                actual_generated_lengths
                != details["len_generated_seq"].to_numpy(dtype=np.int64)
            ).sum()
        )
        raise ValueError(f"Dettagli: {count} len_generated_seq incoerenti")

    if not np.array_equal(
        details["len_generated_seq"].to_numpy(dtype=np.int64),
        details["lmax"].to_numpy(dtype=np.int64),
    ):
        count = int(
            (
                details["len_generated_seq"].to_numpy(dtype=np.int64)
                != details["lmax"].to_numpy(dtype=np.int64)
            ).sum()
        )
        raise ValueError(f"Dettagli: {count} righe con len(z) diversa da lmax")

    expected_sw_r = (
        details["sw_matches"].to_numpy(dtype=float)
        / details["len_xprime"].to_numpy(dtype=float)
    )
    actual_sw_r = details["sw_R"].to_numpy(dtype=float)
    if not np.allclose(expected_sw_r, actual_sw_r, rtol=1e-12, atol=1e-12):
        count = int(
            (~np.isclose(expected_sw_r, actual_sw_r, rtol=1e-12, atol=1e-12)).sum()
        )
        raise ValueError(f"Dettagli: {count} formule sw_R incoerenti")

    hit_mask = details["blast6_hit"].to_numpy(dtype=bool)
    expected_blast_r = (
        details["blast6_nident"].to_numpy(dtype=float)
        / details["len_xprime"].to_numpy(dtype=float)
    )
    actual_blast_r = details["blast6_R"].to_numpy(dtype=float)
    if hit_mask.any() and not np.allclose(
        expected_blast_r[hit_mask],
        actual_blast_r[hit_mask],
        rtol=1e-12,
        atol=1e-12,
    ):
        count = int(
            (
                ~np.isclose(
                    expected_blast_r[hit_mask],
                    actual_blast_r[hit_mask],
                    rtol=1e-12,
                    atol=1e-12,
                )
            ).sum()
        )
        raise ValueError(f"Dettagli: {count} formule blast6_R incoerenti")

    no_hit_mask = ~hit_mask
    if (
        (details.loc[no_hit_mask, "blast6_nident"] != 0).any()
        or (details.loc[no_hit_mask, "blast6_R"] != 0.0).any()
    ):
        raise ValueError(
            "Dettagli: righe BLAST6 no-hit con nident o R diversi da zero"
        )

    for column in ["sw_R", "blast6_R"]:
        outside = ~details[column].between(0.0, 1.0, inclusive="both")
        if outside.any():
            raise ValueError(
                f"Dettagli: {int(outside.sum())} valori {column} fuori [0, 1]"
            )

    details["_input_order"] = np.arange(len(details), dtype=np.int64)
    return details, original_columns


def load_and_validate_categories(
    train_pickle: str | Path,
    test_pickle: str | Path,
    expected_source_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    test_categories: pd.DataFrame | None = None
    for split, path in [("train", train_pickle), ("test", test_pickle)]:
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Pickle categorie {split} non trovato: {source_path}"
            )
        frame = pd.read_pickle(source_path)
        require_columns(frame, CATEGORY_REQUIRED_COLUMNS, f"Pickle {split}")
        frame = clean_text_columns(
            frame,
            CATEGORY_REQUIRED_COLUMNS,
            f"Pickle {split}",
        )
        frame = frame[CATEGORY_REQUIRED_COLUMNS].copy()
        frame["Split"] = split
        frames.append(frame)
        if split == "test":
            test_categories = frame[CATEGORY_REQUIRED_COLUMNS].copy()

    categories_all = pd.concat(frames, ignore_index=True)
    require_expected_count(
        len(categories_all),
        expected_source_rows,
        "Righe complessive dei pickle non aumentati",
    )

    expected_couple = (
        categories_all["Category_x"]
        + " - "
        + categories_all["Category_y"]
    )
    mismatch = expected_couple.ne(categories_all["Category_Couple"])
    if mismatch.any():
        raise ValueError(
            "Pickle categorie: "
            f"{int(mismatch.sum())} Category_Couple incoerenti con Category_x/y"
        )

    category_map = categories_all[CATEGORY_REQUIRED_COLUMNS].drop_duplicates()
    ambiguity = (
        category_map.groupby(
            ["RNA_sequence_x", "RNA_sequence_y"],
            sort=False,
        )
        .size()
        .gt(1)
    )
    if ambiguity.any():
        raise ValueError(
            "Pickle categorie: "
            f"{int(ambiguity.sum())} coppie RNA1/RNA2 con categorie ambigue"
        )

    if test_categories is None:
        raise RuntimeError("Pickle categorie: sorgente test non caricata")

    test_category_map = test_categories.drop_duplicates()
    test_ambiguity = (
        test_category_map.groupby(
            ["RNA_sequence_x", "RNA_sequence_y"],
            sort=False,
        )
        .size()
        .gt(1)
    )
    if test_ambiguity.any():
        raise ValueError(
            "Pickle categorie test: "
            f"{int(test_ambiguity.sum())} coppie RNA1/RNA2 con categorie ambigue"
        )

    return categories_all, test_category_map


def load_and_validate_filtered_test_pairs(
    path: str | Path,
    test_category_map: pd.DataFrame,
    expected_source_rows: int,
    expected_unique_pairs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_path = Path(path)
    if not test_path.is_file():
        raise FileNotFoundError(
            f"File coppie del test filtrato non trovato: {test_path}"
        )

    source = pd.read_csv(test_path, sep="\t", keep_default_na=False)
    require_columns(source, CATEGORY_REQUIRED_COLUMNS, "Test filtrato")
    source = clean_text_columns(
        source,
        CATEGORY_REQUIRED_COLUMNS,
        "Test filtrato",
    )
    source = source[CATEGORY_REQUIRED_COLUMNS].copy()
    require_expected_count(
        len(source),
        expected_source_rows,
        "Righe sorgente del test filtrato",
    )

    expected_couple = source["Category_x"] + " - " + source["Category_y"]
    mismatch = expected_couple.ne(source["Category_Couple"])
    if mismatch.any():
        raise ValueError(
            "Test filtrato: "
            f"{int(mismatch.sum())} Category_Couple incoerenti con Category_x/y"
        )

    pair_map = source.drop_duplicates()
    ambiguity = (
        pair_map.groupby(
            ["RNA_sequence_x", "RNA_sequence_y"],
            sort=False,
        )
        .size()
        .gt(1)
    )
    if ambiguity.any():
        raise ValueError(
            "Test filtrato: "
            f"{int(ambiguity.sum())} coppie RNA1/RNA2 con categorie ambigue"
        )
    pair_map = pair_map.drop_duplicates(
        ["RNA_sequence_x", "RNA_sequence_y"]
    )
    require_expected_count(
        len(pair_map),
        expected_unique_pairs,
        "Coppie RNA1/RNA2 uniche del test filtrato",
    )

    provenance_check = pair_map.merge(
        test_category_map,
        on=CATEGORY_REQUIRED_COLUMNS,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing = ~provenance_check["_merge"].eq("both")
    if missing.any():
        examples = (
            provenance_check.loc[
                missing,
                ["RNA_sequence_x", "RNA_sequence_y"],
            ]
            .head(3)
            .to_dict("records")
        )
        raise ValueError(
            "Test filtrato: "
            f"{int(missing.sum())} coppie/categorie non appartengono al "
            f"pickle originale di test; esempi={examples}"
        )

    return source, pair_map


def load_and_validate_heatmap_counts(
    path: str | Path,
    categories_all: pd.DataFrame,
    expected_types_15: int,
    expected_directional_types: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    heatmap_path = Path(path)
    if not heatmap_path.is_file():
        raise FileNotFoundError(
            f"Conteggi heatmap non trovati: {heatmap_path}"
        )

    heatmap = pd.read_csv(heatmap_path, sep="\t", keep_default_na=False)
    require_columns(heatmap, HEATMAP_REQUIRED_COLUMNS, "Conteggi heatmap")
    heatmap = clean_text_columns(
        heatmap,
        ["Category_x", "Category_y"],
        "Conteggi heatmap",
    )
    heatmap = convert_numeric_columns(
        heatmap,
        ["train", "test", "total"],
        "Conteggi heatmap",
    )
    for column in ["train", "test", "total"]:
        values = heatmap[column].to_numpy(dtype=float)
        if not np.allclose(values, np.rint(values), rtol=0.0, atol=0.0):
            raise ValueError(f"Conteggi heatmap: valori non interi in {column}")
        heatmap[column] = heatmap[column].astype("int64")

    if heatmap.duplicated(["Category_x", "Category_y"]).any():
        raise ValueError("Conteggi heatmap: coppie direzionali duplicate")
    if not (heatmap["train"] + heatmap["test"]).equals(heatmap["total"]):
        raise ValueError("Conteggi heatmap: train + test non coincide con total")
    require_expected_count(
        len(heatmap),
        expected_directional_types,
        "Tipi direzionali nella heatmap",
    )

    source_counts = (
        categories_all.groupby(["Category_x", "Category_y"], sort=False)
        .size()
        .rename("source_total")
        .reset_index()
    )
    count_check = heatmap.merge(
        source_counts,
        on=["Category_x", "Category_y"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not count_check["_merge"].eq("both").all():
        raise ValueError(
            "Conteggi heatmap: categorie diverse da quelle dei pickle sorgente"
        )
    if not count_check["total"].equals(count_check["source_total"]):
        mismatch = int((count_check["total"] != count_check["source_total"]).sum())
        raise ValueError(
            f"Conteggi heatmap: {mismatch} totali diversi dai pickle sorgente"
        )

    heatmap = heatmap.copy()
    heatmap["Category_Couple"] = (
        heatmap["Category_x"] + " - " + heatmap["Category_y"]
    )
    heatmap["_interaction_key"] = [
        canonical_interaction_key(category_x, category_y)
        for category_x, category_y in zip(
            heatmap["Category_x"],
            heatmap["Category_y"],
        )
    ]

    reference_rows: list[dict[str, object]] = []
    for interaction_key, group in heatmap.groupby(
        "_interaction_key",
        sort=False,
    ):
        preferred = group.sort_values(
            ["total", "Category_Couple"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        directional_subtypes = sorted(
            group["Category_Couple"].tolist(),
            key=lambda value: (value.casefold(), value),
        )
        reference_rows.append(
            {
                "_interaction_key": interaction_key,
                "interaction_type": preferred["Category_Couple"],
                "heatmap_train_pairs": int(group["train"].sum()),
                "heatmap_test_pairs": int(group["test"].sum()),
                "heatmap_total_pairs": int(group["total"].sum()),
                "n_directional_subtypes": len(group),
                "directional_subtypes": " | ".join(directional_subtypes),
            }
        )

    reference_15 = pd.DataFrame(reference_rows)
    require_expected_count(
        len(reference_15),
        expected_types_15,
        "Tipi non direzionali della heatmap",
    )
    reference_15 = reference_15.sort_values(
        ["heatmap_total_pairs", "interaction_type"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    reference_15.insert(
        0,
        "type_order",
        np.arange(1, len(reference_15) + 1, dtype=np.int64),
    )
    reference_15["category_mode"] = "non_directional_15"

    reference_18 = heatmap[
        ["Category_Couple", "train", "test", "total"]
    ].rename(
        columns={
            "Category_Couple": "interaction_type",
            "train": "heatmap_train_pairs",
            "test": "heatmap_test_pairs",
            "total": "heatmap_total_pairs",
        }
    )
    reference_18 = reference_18.sort_values(
        ["heatmap_total_pairs", "interaction_type"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    reference_18.insert(
        0,
        "type_order",
        np.arange(1, len(reference_18) + 1, dtype=np.int64),
    )
    reference_18["category_mode"] = "directional_18"
    reference_18["n_directional_subtypes"] = 1
    reference_18["directional_subtypes"] = reference_18["interaction_type"]

    return reference_15, reference_18


def filter_and_enrich_details_with_test_categories(
    details: pd.DataFrame,
    original_columns: list[str],
    category_map: pd.DataFrame,
    reference_15: pd.DataFrame,
    reference_18: pd.DataFrame,
    expected_rows: int,
    expected_rna1: int,
    expected_types: int,
    expected_directional_types: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enriched = details.merge(
        category_map,
        left_on=["seq1", "xprime"],
        right_on=["RNA_sequence_x", "RNA_sequence_y"],
        how="inner",
        sort=False,
        validate="many_to_one",
    )
    require_expected_count(
        len(enriched),
        expected_rows,
        "Confronti appartenenti alle coppie del test filtrato",
    )
    require_expected_count(
        int(enriched["seq1"].nunique()),
        expected_rna1,
        "RNA1 distinti dopo il filtro sulle coppie del test",
    )
    matched_pairs = enriched[["seq1", "xprime"]].drop_duplicates()
    if len(matched_pairs) != len(category_map):
        raise ValueError(
            "Filtro test: non tutte le coppie uniche del test filtrato sono "
            f"presenti nei dettagli ({len(matched_pairs)}/{len(category_map)})"
        )

    enriched = enriched.sort_values(
        "_input_order",
        kind="mergesort",
    ).reset_index(drop=True)
    if not enriched["_input_order"].is_monotonic_increasing:
        raise RuntimeError("Filtro test: ordine relativo delle righe non preservato")

    enriched["_interaction_key"] = [
        canonical_interaction_key(category_x, category_y)
        for category_x, category_y in zip(
            enriched["Category_x"],
            enriched["Category_y"],
        )
    ]
    type_lookup = reference_15.set_index("_interaction_key")
    enriched["interaction_type"] = enriched["_interaction_key"].map(
        type_lookup["interaction_type"]
    )
    enriched["interaction_type_order"] = enriched["_interaction_key"].map(
        type_lookup["type_order"]
    )
    if enriched["interaction_type"].isna().any():
        raise ValueError(
            "Filtro test: categorie presenti nei dettagli ma assenti "
            "dalla heatmap"
        )

    require_expected_count(
        int(enriched["interaction_type"].nunique()),
        expected_types,
        "Tipi non direzionali nelle coppie del test filtrato",
    )
    require_expected_count(
        int(enriched["Category_Couple"].nunique()),
        expected_directional_types,
        "Tipi direzionali nelle coppie del test filtrato",
    )

    present_keys = set(enriched["_interaction_key"])
    filtered_reference = reference_15.loc[
        reference_15["_interaction_key"].isin(present_keys)
    ].copy()
    require_expected_count(
        len(filtered_reference),
        expected_types,
        "Tipi non direzionali della reference test-only",
    )
    filtered_reference = filtered_reference.sort_values(
        ["heatmap_test_pairs", "interaction_type"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    filtered_reference["type_order"] = np.arange(
        1,
        len(filtered_reference) + 1,
        dtype=np.int64,
    )
    filtered_reference["category_mode"] = "non_directional_test_only"
    filtered_type_order = filtered_reference.set_index("_interaction_key")[
        "type_order"
    ]
    enriched["interaction_type_order"] = enriched["_interaction_key"].map(
        filtered_type_order
    )

    present_directions = set(enriched["Category_Couple"])
    filtered_reference_directional = reference_18.loc[
        reference_18["interaction_type"].isin(present_directions)
    ].copy()
    require_expected_count(
        len(filtered_reference_directional),
        expected_directional_types,
        "Tipi direzionali della reference test-only",
    )
    filtered_reference_directional = filtered_reference_directional.sort_values(
        ["heatmap_test_pairs", "interaction_type"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    filtered_reference_directional["type_order"] = np.arange(
        1,
        len(filtered_reference_directional) + 1,
        dtype=np.int64,
    )
    filtered_reference_directional["category_mode"] = "directional_test_only"

    output_columns = (
        original_columns
        + [
            "Category_x",
            "Category_y",
            "Category_Couple",
            "interaction_type_order",
            "interaction_type",
        ]
    )
    enriched = enriched.drop(
        columns=[
            "RNA_sequence_x",
            "RNA_sequence_y",
        ]
    )
    internal_columns = [
        "_input_order",
        "_interaction_key",
    ]
    return (
        enriched[output_columns + internal_columns],
        filtered_reference,
        filtered_reference_directional,
    )


def build_best_per_rna1_group(
    details: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    group_columns = ["seq1", group_column]
    target_counts = (
        details.groupby(group_columns, sort=False)
        .size()
        .rename("n_targets_in_type")
        .reset_index()
    )

    sw_best = (
        details.sort_values(
            group_columns
            + [
                "sw_R",
                "sw_matches",
                "sw_score",
                "sw_aln_len",
                "target_index",
            ],
            ascending=[True, True, False, False, False, False, True],
            kind="mergesort",
        )
        .drop_duplicates(group_columns, keep="first")
        .copy()
    )

    common_columns = [
        "row_index",
        "seq1",
        "generated_seq",
        "len_generated_seq",
        "lmax",
        "n_T2",
        group_column,
    ]
    sw_columns = [
        "target_index",
        "xprime",
        "len_xprime",
        "Category_x",
        "Category_y",
        "Category_Couple",
        "sw_matches",
        "sw_aln_len",
        "sw_aln_pident",
        "sw_score",
        "sw_R",
    ]
    sw_columns = [
        column for column in sw_columns if column not in common_columns
    ]
    sw_rename = {
        "target_index": "sw_selected_target_index",
        "xprime": "sw_selected_xprime",
        "len_xprime": "sw_selected_len_xprime",
        "Category_x": "sw_selected_Category_x",
        "Category_y": "sw_selected_Category_y",
        "Category_Couple": "sw_selected_Category_Couple",
    }
    sw_rename.pop(group_column, None)
    best = sw_best[common_columns + sw_columns].rename(
        columns=sw_rename
    )
    best = best.merge(
        target_counts,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )

    blast_hits = details.loc[details["blast6_hit"]].copy()
    if not blast_hits.empty:
        blast_best = (
            blast_hits.sort_values(
                group_columns
                + [
                    "blast6_R",
                    "blast6_nident",
                    "blast6_bitscore",
                    "blast6_aln_len",
                    "target_index",
                ],
                ascending=[True, True, False, False, False, False, True],
                kind="mergesort",
            )
            .drop_duplicates(group_columns, keep="first")
            .copy()
        )
        blast_value_columns = [
            "target_index",
            "xprime",
            "len_xprime",
            "Category_x",
            "Category_y",
            "Category_Couple",
            "blast6_hit",
            "blast6_nident",
            "blast6_aln_len",
            "blast6_aln_pident",
            "blast6_bitscore",
            "blast6_evalue",
            "blast6_R",
        ]
        blast_value_columns = [
            column
            for column in blast_value_columns
            if column not in group_columns
        ]
        blast_columns = group_columns + blast_value_columns
        blast_rename = {
            "target_index": "blast6_selected_target_index",
            "xprime": "blast6_selected_xprime",
            "len_xprime": "blast6_selected_len_xprime",
            "Category_x": "blast6_selected_Category_x",
            "Category_y": "blast6_selected_Category_y",
            "Category_Couple": "blast6_selected_Category_Couple",
        }
        blast_rename.pop(group_column, None)
        blast_best = blast_best[blast_columns].rename(
            columns=blast_rename
        )
        best = best.merge(
            blast_best,
            on=group_columns,
            how="left",
            validate="one_to_one",
        )
    else:
        for column in [
            "blast6_selected_target_index",
            "blast6_selected_xprime",
            "blast6_selected_len_xprime",
            "blast6_selected_Category_x",
            "blast6_selected_Category_y",
            "blast6_selected_Category_Couple",
            "blast6_hit",
            "blast6_nident",
            "blast6_aln_len",
            "blast6_aln_pident",
            "blast6_bitscore",
            "blast6_evalue",
            "blast6_R",
        ]:
            best[column] = np.nan

    best["blast6_hit"] = best["blast6_hit"].fillna(False).astype(bool)
    best["blast6_R"] = best["blast6_R"].fillna(0.0).astype(float)
    if best.duplicated(group_columns).any():
        raise RuntimeError(
            f"Best per RNA1/{group_column}: gruppi duplicati dopo la selezione"
        )

    return best.sort_values(
        [group_column, "seq1"],
        kind="mergesort",
    ).reset_index(drop=True)


def numeric_stat(series: pd.Series, operation: str) -> float | str:
    if series.empty:
        return ""
    if operation == "mean":
        return float(series.mean())
    if operation == "median":
        return float(series.median())
    if operation == "min":
        return float(series.min())
    if operation == "max":
        return float(series.max())
    raise ValueError(f"Operazione statistica non riconosciuta: {operation}")


def build_summary(
    details: pd.DataFrame,
    best: pd.DataFrame,
    reference: pd.DataFrame,
    details_group_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for reference_row in reference.to_dict("records"):
        interaction_type = str(reference_row["interaction_type"])
        detail_group = details.loc[
            details[details_group_column].eq(interaction_type)
        ]
        best_group = best.loc[best[details_group_column].eq(interaction_type)]
        if detail_group.empty:
            raise ValueError(
                f"Summary: nessun confronto per {interaction_type}"
            )
        if best_group.empty:
            raise ValueError(
                f"Summary: nessun best per RNA1 per {interaction_type}"
            )

        pair_hits = detail_group.loc[detail_group["blast6_hit"], "blast6_R"]
        best_hits = best_group.loc[best_group["blast6_hit"], "blast6_R"]
        pair_total = len(detail_group)
        best_total = len(best_group)
        pair_hit_count = len(pair_hits)
        best_hit_count = len(best_hits)

        rows.append(
            {
                "type_order": int(reference_row["type_order"]),
                "interaction_type": interaction_type,
                "category_mode": reference_row["category_mode"],
                "heatmap_train_pairs": int(reference_row["heatmap_train_pairs"]),
                "heatmap_test_pairs": int(reference_row["heatmap_test_pairs"]),
                "heatmap_total_pairs": int(reference_row["heatmap_total_pairs"]),
                "n_directional_subtypes": int(
                    reference_row["n_directional_subtypes"]
                ),
                "directional_subtypes": reference_row["directional_subtypes"],
                "n_pair_comparisons": pair_total,
                "n_RNA1": int(detail_group["seq1"].nunique()),
                "sw_pair_mean_R": float(detail_group["sw_R"].mean()),
                "sw_pair_median_R": float(detail_group["sw_R"].median()),
                "sw_best_mean_R": float(best_group["sw_R"].mean()),
                "sw_best_median_R": float(best_group["sw_R"].median()),
                "sw_best_min_R": float(best_group["sw_R"].min()),
                "sw_best_max_R": float(best_group["sw_R"].max()),
                "blast6_pair_hit_count": pair_hit_count,
                "blast6_pair_no_hit_count": pair_total - pair_hit_count,
                "blast6_pair_hit_pct": 100.0 * pair_hit_count / pair_total,
                "blast6_pair_mean_R_hits": numeric_stat(pair_hits, "mean"),
                "blast6_pair_median_R_hits": numeric_stat(pair_hits, "median"),
                "blast6_pair_mean_R_no_hit_zero": float(
                    detail_group["blast6_R"].mean()
                ),
                "blast6_best_RNA1_hit_count": best_hit_count,
                "blast6_best_RNA1_no_hit_count": best_total - best_hit_count,
                "blast6_best_RNA1_hit_pct": 100.0
                * best_hit_count
                / best_total,
                "blast6_best_mean_R_hits": numeric_stat(best_hits, "mean"),
                "blast6_best_median_R_hits": numeric_stat(best_hits, "median"),
                "blast6_best_min_R_hits": numeric_stat(best_hits, "min"),
                "blast6_best_max_R_hits": numeric_stat(best_hits, "max"),
                "blast6_best_mean_R_no_hit_zero": float(
                    best_group["blast6_R"].mean()
                ),
            }
        )

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if int(summary["n_pair_comparisons"].sum()) != len(details):
        raise RuntimeError(
            "Summary: la somma dei confronti non coincide con i dettagli"
        )
    return summary


def make_output_paths(
    output_prefix: str | Path,
    n_types: int,
    n_directional_types: int,
) -> OutputPaths:
    prefix = Path(output_prefix)
    return OutputPaths(
        details=Path(f"{prefix}_details.tsv"),
        best_per_rna1_type=Path(f"{prefix}_best_per_RNA1_type.tsv"),
        summary_types=Path(f"{prefix}_summary_{n_types}_types.tsv"),
        summary_directional_types=Path(
            f"{prefix}_summary_{n_directional_types}_directional_types.tsv"
        ),
    )


def check_output_targets(paths: OutputPaths, overwrite: bool) -> None:
    outputs = [
        paths.details,
        paths.best_per_rna1_type,
        paths.summary_types,
        paths.summary_directional_types,
    ]
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("Output: path duplicati")
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Gli output esistono gia'. Usa --overwrite per sostituirli:\n"
            f"{formatted}"
        )


def write_tsv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            frame.to_csv(
                handle,
                sep="\t",
                index=False,
                na_rep="",
                lineterminator="\n",
            )
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def verify_written_outputs(
    paths: OutputPaths,
    expected_detail_rows: int,
    expected_best_rows: int,
    expected_types: int,
    expected_directional_types: int,
) -> None:
    details = pd.read_csv(paths.details, sep="\t", keep_default_na=False)
    best = pd.read_csv(
        paths.best_per_rna1_type,
        sep="\t",
        keep_default_na=False,
    )
    summary_types = pd.read_csv(
        paths.summary_types,
        sep="\t",
        keep_default_na=False,
    )
    summary_directional = pd.read_csv(
        paths.summary_directional_types,
        sep="\t",
        keep_default_na=False,
    )

    require_expected_count(len(details), expected_detail_rows, "Output dettagli")
    require_expected_count(len(best), expected_best_rows, "Output best")
    require_expected_count(
        len(summary_types),
        expected_types,
        "Output summary non direzionale",
    )
    require_expected_count(
        len(summary_directional),
        expected_directional_types,
        "Output summary direzionale",
    )
    if int(summary_types["n_pair_comparisons"].sum()) != len(details):
        raise RuntimeError(
            "Output summary non direzionale: totale confronti incoerente"
        )
    if int(summary_directional["n_pair_comparisons"].sum()) != len(details):
        raise RuntimeError(
            "Output summary direzionale: totale confronti incoerente"
        )
    if details["interaction_type"].eq("").any():
        raise RuntimeError("Output dettagli: interaction_type vuoto")
    if best.duplicated(["seq1", "interaction_type"]).any():
        raise RuntimeError("Output best: RNA1/tipo duplicati")


def print_summary(summary: pd.DataFrame) -> None:
    n_types = len(summary)
    print(f"\n=== SUMMARY {n_types} TIPI DI INTERAZIONE DEL TEST ===")
    for row in summary.itertuples(index=False):
        blast_mean = row.blast6_best_mean_R_hits
        blast_mean_text = (
            f"{float(blast_mean):.6f}" if blast_mean != "" else "NA"
        )
        print(
            f"[{int(row.type_order):02d}/{n_types}] "
            f"{row.interaction_type} | "
            f"confronti={int(row.n_pair_comparisons)} | "
            f"RNA1={int(row.n_RNA1)} | "
            f"SW best mean={float(row.sw_best_mean_R):.6f} | "
            f"BLAST6 hit={int(row.blast6_best_RNA1_hit_count)}/"
            f"{int(row.n_RNA1)} "
            f"({float(row.blast6_best_RNA1_hit_pct):.2f}%) | "
            f"BLAST6 best mean hits={blast_mean_text}"
        )


def run_analysis(
    *,
    details_path: str | Path,
    train_pickle: str | Path,
    test_pickle: str | Path,
    test_pairs_path: str | Path,
    heatmap_counts: str | Path,
    output_prefix: str | Path,
    expected_details_rows: int,
    expected_rna1: int,
    expected_source_rows: int,
    expected_test_source_rows: int,
    expected_unique_test_pairs: int,
    expected_types: int,
    expected_directional_types: int,
    expected_reference_types: int,
    expected_reference_directional_types: int,
    overwrite: bool,
) -> OutputPaths:
    outputs = make_output_paths(
        output_prefix,
        n_types=expected_types,
        n_directional_types=expected_directional_types,
    )
    check_output_targets(outputs, overwrite=overwrite)

    print("=== B4 LMAX TOP-K=2: ANALISI SULLE SOLE COPPIE DEL TEST ===")
    print(f"Dettagli allineamenti : {Path(details_path)}")
    print(f"Pickle train          : {Path(train_pickle)}")
    print(f"Pickle test           : {Path(test_pickle)}")
    print(f"Coppie test valutate  : {Path(test_pairs_path)}")
    print(f"Conteggi heatmap      : {Path(heatmap_counts)}")
    print(f"Prefisso output       : {Path(output_prefix)}")

    details, original_columns = load_and_validate_details(
        details_path,
        expected_rows=expected_details_rows,
        expected_rna1=expected_rna1,
    )
    print(
        "Dettagli validati     : "
        f"{len(details)} confronti, {details['seq1'].nunique()} RNA1"
    )

    categories_all, test_category_map = load_and_validate_categories(
        train_pickle,
        test_pickle,
        expected_source_rows=expected_source_rows,
    )
    print(
        "Categorie originali   : "
        f"{len(categories_all)} record sorgente, "
        f"{len(test_category_map)} coppie test originali uniche"
    )

    test_source, filtered_test_map = load_and_validate_filtered_test_pairs(
        test_pairs_path,
        test_category_map,
        expected_source_rows=expected_test_source_rows,
        expected_unique_pairs=expected_unique_test_pairs,
    )
    print(
        "Test filtrato         : "
        f"{len(test_source)} righe sorgente, "
        f"{len(filtered_test_map)} coppie uniche"
    )

    reference_15, reference_18 = load_and_validate_heatmap_counts(
        heatmap_counts,
        categories_all,
        expected_types_15=expected_reference_types,
        expected_directional_types=expected_reference_directional_types,
    )
    print(
        "Heatmap validata      : "
        f"{len(reference_15)} tipi non direzionali, "
        f"{len(reference_18)} tipi direzionali"
    )

    enriched, reference_test, reference_directional_test = (
        filter_and_enrich_details_with_test_categories(
        details,
        original_columns,
        filtered_test_map,
        reference_15,
        reference_18,
        expected_rows=expected_unique_test_pairs,
        expected_rna1=expected_rna1,
        expected_types=expected_types,
        expected_directional_types=expected_directional_types,
        )
    )
    print(
        "Filtro coppie test    : "
        f"{len(enriched)}/{len(details)} confronti conservati; "
        f"{enriched['seq1'].nunique()} RNA1; "
        f"{enriched['interaction_type'].nunique()} tipi"
    )

    best_types = build_best_per_rna1_group(
        enriched,
        group_column="interaction_type",
    )
    best_directional = build_best_per_rna1_group(
        enriched,
        group_column="Category_Couple",
    )
    summary_types = build_summary(
        enriched,
        best_types,
        reference_test,
        details_group_column="interaction_type",
    )
    summary_directional = build_summary(
        enriched,
        best_directional,
        reference_directional_test,
        details_group_column="Category_Couple",
    )
    require_expected_count(
        len(summary_types),
        expected_types,
        "Summary non direzionale",
    )
    require_expected_count(
        len(summary_directional),
        expected_directional_types,
        "Summary direzionale",
    )

    details_output = enriched.drop(
        columns=["_input_order", "_interaction_key"],
    )
    print_summary(summary_types)

    write_tsv_atomic(details_output, outputs.details)
    write_tsv_atomic(best_types, outputs.best_per_rna1_type)
    write_tsv_atomic(summary_types, outputs.summary_types)
    write_tsv_atomic(
        summary_directional,
        outputs.summary_directional_types,
    )

    verify_written_outputs(
        outputs,
        expected_detail_rows=len(details_output),
        expected_best_rows=len(best_types),
        expected_types=expected_types,
        expected_directional_types=expected_directional_types,
    )
    print("\n=== OUTPUT VERIFICATI ===")
    print(f"Dettagli categorie    : {outputs.details}")
    print(f"Best RNA1 x tipo      : {outputs.best_per_rna1_type}")
    print(f"Summary {expected_types} tipi       : {outputs.summary_types}")
    print(
        f"Audit {expected_directional_types} direzioni    : "
        f"{outputs.summary_directional_types}"
    )
    print(
        "Controlli finali      : righe, join, formule R, classi e totali OK"
    )
    return outputs


def run_self_test() -> None:
    token = uuid.uuid4().hex
    root = Path.cwd()
    train_pickle = root / f".b4_interaction_selftest_{token}_train.pkl"
    test_pickle = root / f".b4_interaction_selftest_{token}_test.pkl"
    test_pairs = root / f".b4_interaction_selftest_{token}_test_pairs.tsv"
    heatmap_counts = root / f".b4_interaction_selftest_{token}_heatmap.tsv"
    details_path = root / f".b4_interaction_selftest_{token}_details.tsv"
    output_prefix = root / f".b4_interaction_selftest_{token}_result"
    cleanup_paths = [
        train_pickle,
        test_pickle,
        test_pairs,
        heatmap_counts,
        details_path,
        *make_output_paths(
            output_prefix,
            n_types=2,
            n_directional_types=3,
        ).__dict__.values(),
    ]

    try:
        categories = pd.DataFrame(
            [
                ["r1", "AA", "miRNA", "snoRNA", "miRNA - snoRNA"],
                ["r1", "AU", "miRNA", "snoRNA", "miRNA - snoRNA"],
                ["r1", "AC", "miRNA", "snoRNA", "miRNA - snoRNA"],
                ["r2", "CC", "snoRNA", "miRNA", "snoRNA - miRNA"],
                ["r2", "CG", "snoRNA", "snoRNA", "snoRNA - snoRNA"],
            ],
            columns=CATEGORY_REQUIRED_COLUMNS,
        )
        categories.iloc[:2].to_pickle(train_pickle)
        categories.iloc[2:].to_pickle(test_pickle)
        pd.concat(
            [
                categories.iloc[2:],
                categories.iloc[[2]],
            ],
            ignore_index=True,
        ).to_csv(test_pairs, sep="\t", index=False)

        pd.DataFrame(
            [
                ["miRNA", "snoRNA", 2, 1, 3],
                ["snoRNA", "miRNA", 0, 1, 1],
                ["snoRNA", "snoRNA", 0, 1, 1],
            ],
            columns=HEATMAP_REQUIRED_COLUMNS,
        ).to_csv(heatmap_counts, sep="\t", index=False)

        detail_rows = [
            [0, "r1", "AAAA", 4, 4, 3, 0, "AA", 2, 1, 2, 50.0, 2, 0.5,
             False, 0, 0, 0.0, 0.0, "", 0.0],
            [0, "r1", "AAAA", 4, 4, 3, 1, "AU", 2, 2, 2, 100.0, 4, 1.0,
             True, 1, 1, 100.0, 5.0, "1e-3", 0.5],
            [0, "r1", "AAAA", 4, 4, 3, 2, "AC", 2, 2, 2, 100.0, 4, 1.0,
             True, 2, 2, 100.0, 8.0, "1e-4", 1.0],
            [1, "r2", "CCCC", 4, 4, 2, 0, "CC", 2, 2, 2, 100.0, 4, 1.0,
             True, 2, 2, 100.0, 10.0, "1e-4", 1.0],
            [1, "r2", "CCCC", 4, 4, 2, 1, "CG", 2, 1, 2, 50.0, 2, 0.5,
             False, 0, 0, 0.0, 0.0, "", 0.0],
        ]
        pd.DataFrame(
            detail_rows,
            columns=DETAIL_REQUIRED_COLUMNS,
        ).to_csv(details_path, sep="\t", index=False)

        outputs = run_analysis(
            details_path=details_path,
            train_pickle=train_pickle,
            test_pickle=test_pickle,
            test_pairs_path=test_pairs,
            heatmap_counts=heatmap_counts,
            output_prefix=output_prefix,
            expected_details_rows=5,
            expected_rna1=2,
            expected_source_rows=5,
            expected_test_source_rows=4,
            expected_unique_test_pairs=3,
            expected_types=2,
            expected_directional_types=3,
            expected_reference_types=2,
            expected_reference_directional_types=3,
            overwrite=True,
        )

        best = pd.read_csv(
            outputs.best_per_rna1_type,
            sep="\t",
            keep_default_na=False,
        )
        selected = best.loc[
            best["seq1"].eq("r1")
            & best["interaction_type"].eq("miRNA - snoRNA")
        ]
        if len(selected) != 1:
            raise AssertionError("Self-test: best r1/miRNA-snoRNA mancante")
        if selected.iloc[0]["sw_selected_xprime"] != "AC":
            raise AssertionError("Self-test: tie-break/best Smith-Waterman errato")
        if float(selected.iloc[0]["sw_R"]) != 1.0:
            raise AssertionError("Self-test: sw_R best errato")
    finally:
        for path in cleanup_paths:
            path = Path(path)
            if path.exists():
                path.unlink()

    print("\nSELF-TEST COMPLETATO: OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Associa le categorie ai dettagli B4 lmax top-k=2 gia' calcolati "
            "e valuta esclusivamente le coppie del file di test filtrato."
        )
    )
    parser.add_argument("--details", default=DEFAULT_DETAILS)
    parser.add_argument("--train_pickle", default=DEFAULT_TRAIN_PICKLE)
    parser.add_argument("--test_pickle", default=DEFAULT_TEST_PICKLE)
    parser.add_argument("--test_pairs", default=DEFAULT_TEST_PAIRS)
    parser.add_argument("--heatmap_counts", default=DEFAULT_HEATMAP_COUNTS)
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--expected_details_rows",
        type=int,
        default=DEFAULT_EXPECTED_DETAILS_ROWS,
        help="0 disabilita il controllo sul numero di confronti.",
    )
    parser.add_argument(
        "--expected_rna1",
        type=int,
        default=DEFAULT_EXPECTED_RNA1,
        help="0 disabilita il controllo sul numero di RNA1.",
    )
    parser.add_argument(
        "--expected_source_rows",
        type=int,
        default=DEFAULT_EXPECTED_SOURCE_ROWS,
        help="0 disabilita il controllo sul totale dei pickle.",
    )
    parser.add_argument(
        "--expected_test_source_rows",
        type=int,
        default=DEFAULT_EXPECTED_TEST_SOURCE_ROWS,
        help="0 disabilita il controllo sulle righe del test filtrato.",
    )
    parser.add_argument(
        "--expected_unique_test_pairs",
        type=int,
        default=DEFAULT_EXPECTED_UNIQUE_TEST_PAIRS,
        help="0 disabilita il controllo sulle coppie uniche del test.",
    )
    parser.add_argument(
        "--expected_types",
        type=int,
        default=DEFAULT_EXPECTED_TYPES,
    )
    parser.add_argument(
        "--expected_directional_types",
        type=int,
        default=DEFAULT_EXPECTED_DIRECTIONAL_TYPES,
    )
    parser.add_argument(
        "--expected_reference_types",
        type=int,
        default=DEFAULT_EXPECTED_REFERENCE_TYPES,
        help="Tipi non direzionali attesi nella heatmap completa.",
    )
    parser.add_argument(
        "--expected_reference_directional_types",
        type=int,
        default=DEFAULT_EXPECTED_REFERENCE_DIRECTIONAL_TYPES,
        help="Tipi direzionali attesi nella heatmap completa.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sostituisce soltanto i quattro output dello stesso prefisso.",
    )
    parser.add_argument(
        "--self_test",
        action="store_true",
        help="Esegue un test sintetico in una cartella temporanea.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    run_analysis(
        details_path=args.details,
        train_pickle=args.train_pickle,
        test_pickle=args.test_pickle,
        test_pairs_path=args.test_pairs,
        heatmap_counts=args.heatmap_counts,
        output_prefix=args.output_prefix,
        expected_details_rows=args.expected_details_rows,
        expected_rna1=args.expected_rna1,
        expected_source_rows=args.expected_source_rows,
        expected_test_source_rows=args.expected_test_source_rows,
        expected_unique_test_pairs=args.expected_unique_test_pairs,
        expected_types=args.expected_types,
        expected_directional_types=args.expected_directional_types,
        expected_reference_types=args.expected_reference_types,
        expected_reference_directional_types=(
            args.expected_reference_directional_types
        ),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
