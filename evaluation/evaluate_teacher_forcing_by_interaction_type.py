"""Teacher-forcing esatto sul test B4, globale e per tipo d'interazione.

Questo script e' separato sia dalla valutazione storica globale
``evaluate_teacher_forcing.py`` sia dagli script di generazione libera.
Non genera sequenze e non calcola allineamenti.

Le metriche sono accumulate sui soli token target validi:

    loss = NLL totale / numero totale di token non-PAD
    token_accuracy = token corretti / numero totale di token non-PAD
    perplexity = exp(loss)

I PAD sono esclusi da NLL, accuracy e perplexity perche' sono riempitivi
introdotti soltanto per uniformare le lunghezze dei tensori.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Lo script e' pubblicato in ``evaluation/``. Tutti i path di default sono
# quindi ancorati alla root del repository e non dipendono dalla directory
# dalla quale viene lanciato Python.
DEFAULT_MODEL_DIR = PROJECT_ROOT / "ckpt" / "dry_run" / "final_model"
DEFAULT_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "dry_run"
    / "test_filtered_no_aug_pident95_qcov90.tsv"
)
DEFAULT_HEATMAP_COUNTS = (
    PROJECT_ROOT / "heatmap_interaction_types_nonaug_pkl_split_counts.tsv"
)
DEFAULT_OUTPUT_PREFIX = (
    PROJECT_ROOT / "B4_teacher_forcing_test_by_interaction_type"
)

DEFAULT_EXPECTED_ROWS = 1_544
DEFAULT_EXPECTED_UNIQUE_PAIRS = 1_533
DEFAULT_EXPECTED_TYPES = 14
DEFAULT_EXPECTED_DIRECTIONAL_TYPES = 17
DEFAULT_EXPECTED_HEATMAP_TYPES = 15
DEFAULT_BATCH_SIZE = 48

TEST_COLUMNS = [
    "RNA_sequence_x",
    "RNA_sequence_y",
    "Category_x",
    "Category_y",
    "Category_Couple",
]

HEATMAP_COLUMNS = [
    "Category_x",
    "Category_y",
    "train",
    "test",
    "total",
]

DETAIL_METRIC_COLUMNS = [
    "n_valid_tokens",
    "n_correct_tokens",
    "nll_sum",
    "loss",
    "token_accuracy",
    "perplexity",
]

SUMMARY_COLUMNS = [
    "type_order",
    "interaction_type",
    "category_mode",
    "directional_subtypes",
    "n_pairs",
    "n_unique_pairs",
    "n_valid_tokens",
    "n_correct_tokens",
    "nll_sum",
    "loss",
    "token_accuracy",
    "perplexity",
]

OVERALL_COLUMNS = [
    "scope",
    "n_pairs",
    "n_unique_pairs",
    "n_interaction_types",
    "n_directional_types",
    "n_valid_tokens",
    "n_correct_tokens",
    "nll_sum",
    "loss",
    "token_accuracy",
    "perplexity",
]


@dataclass(frozen=True)
class OutputPaths:
    details: Path
    summary_types: Path
    summary_directional: Path
    overall: Path
    metadata: Path


def require_columns(
    frame: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label}: colonne mancanti: {missing}")


def require_expected_count(actual: int, expected: int, label: str) -> None:
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


def safe_perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def load_heatmap_reference(
    path: str | Path,
    expected_types: int,
) -> pd.DataFrame:
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Conteggi heatmap non trovati: {source_path}"
        )

    heatmap = pd.read_csv(source_path, sep="\t", keep_default_na=False)
    require_columns(heatmap, HEATMAP_COLUMNS, "Heatmap")
    heatmap = clean_text_columns(
        heatmap,
        ["Category_x", "Category_y"],
        "Heatmap",
    )
    for column in ["train", "test", "total"]:
        heatmap[column] = pd.to_numeric(heatmap[column], errors="raise")
        values = heatmap[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Heatmap: valori non finiti in {column}")
        if not np.allclose(values, np.rint(values), rtol=0.0, atol=0.0):
            raise ValueError(f"Heatmap: valori non interi in {column}")
        heatmap[column] = heatmap[column].astype("int64")

    if heatmap.duplicated(["Category_x", "Category_y"]).any():
        raise ValueError("Heatmap: coppie direzionali duplicate")
    if not (heatmap["train"] + heatmap["test"]).equals(heatmap["total"]):
        raise ValueError("Heatmap: train + test non coincide con total")

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

    rows: list[dict[str, Any]] = []
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
        rows.append(
            {
                "_interaction_key": interaction_key,
                "interaction_type": str(preferred["Category_Couple"]),
                "heatmap_test_pairs": int(group["test"].sum()),
                "heatmap_total_pairs": int(group["total"].sum()),
                "directional_subtypes": " | ".join(directional_subtypes),
            }
        )

    reference = pd.DataFrame(rows)
    require_expected_count(
        len(reference),
        expected_types,
        "Tipi non direzionali nella heatmap",
    )
    return reference


def load_and_validate_test(
    test_file: str | Path,
    heatmap_counts: str | Path,
    expected_rows: int,
    expected_unique_pairs: int,
    expected_types: int,
    expected_directional_types: int,
    expected_heatmap_types: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_path = Path(test_file)
    if not source_path.is_file():
        raise FileNotFoundError(f"Test TSV non trovato: {source_path}")

    test = pd.read_csv(source_path, sep="\t", keep_default_na=False)
    require_columns(test, TEST_COLUMNS, "Test")
    test = clean_text_columns(test, TEST_COLUMNS, "Test")
    test = test[TEST_COLUMNS].copy()
    require_expected_count(len(test), expected_rows, "Righe del test")

    expected_couple = test["Category_x"] + " - " + test["Category_y"]
    mismatch = expected_couple.ne(test["Category_Couple"])
    if mismatch.any():
        raise ValueError(
            "Test: "
            f"{int(mismatch.sum())} Category_Couple incoerenti con Category_x/y"
        )

    pair_categories = test[TEST_COLUMNS].drop_duplicates()
    ambiguity = (
        pair_categories.groupby(
            ["RNA_sequence_x", "RNA_sequence_y"],
            sort=False,
        )
        .size()
        .gt(1)
    )
    if ambiguity.any():
        raise ValueError(
            "Test: "
            f"{int(ambiguity.sum())} coppie RNA1/RNA2 con categorie ambigue"
        )

    unique_pairs = test[
        ["RNA_sequence_x", "RNA_sequence_y"]
    ].drop_duplicates()
    require_expected_count(
        len(unique_pairs),
        expected_unique_pairs,
        "Coppie RNA1/RNA2 uniche nel test",
    )

    reference = load_heatmap_reference(
        heatmap_counts,
        expected_types=expected_heatmap_types,
    )
    test["_interaction_key"] = [
        canonical_interaction_key(category_x, category_y)
        for category_x, category_y in zip(
            test["Category_x"],
            test["Category_y"],
        )
    ]
    lookup = reference.set_index("_interaction_key")
    test["interaction_type"] = test["_interaction_key"].map(
        lookup["interaction_type"]
    )
    test["directional_subtypes"] = test["_interaction_key"].map(
        lookup["directional_subtypes"]
    )
    if test["interaction_type"].isna().any():
        raise ValueError(
            "Test: categorie presenti nel test ma assenti dalla heatmap"
        )

    require_expected_count(
        int(test["interaction_type"].nunique()),
        expected_types,
        "Tipi non direzionali nel test",
    )
    require_expected_count(
        int(test["Category_Couple"].nunique()),
        expected_directional_types,
        "Tipi direzionali nel test",
    )

    present_keys = set(test["_interaction_key"])
    test_reference = reference.loc[
        reference["_interaction_key"].isin(present_keys)
    ].copy()
    test_reference = test_reference.sort_values(
        ["heatmap_total_pairs", "interaction_type"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    test_reference["type_order"] = np.arange(
        1,
        len(test_reference) + 1,
        dtype=np.int64,
    )
    type_order = test_reference.set_index("_interaction_key")["type_order"]
    test["type_order"] = test["_interaction_key"].map(type_order).astype("int64")
    test.insert(0, "row_index", np.arange(len(test), dtype=np.int64))

    return test, test_reference


def batch_row_metrics_from_arrays(
    token_nll: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    pad_token_id: int,
) -> list[dict[str, float | int]]:
    token_nll = np.asarray(token_nll, dtype=np.float64)
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if token_nll.shape != labels.shape or predictions.shape != labels.shape:
        raise ValueError(
            "Metriche batch: token_nll, predictions e labels devono avere "
            f"la stessa forma; trovate {token_nll.shape}, "
            f"{predictions.shape}, {labels.shape}"
        )
    if not np.isfinite(token_nll).all():
        raise ValueError("Metriche batch: NLL non finita")

    valid_mask = labels != int(pad_token_id)
    rows: list[dict[str, float | int]] = []
    for row_index in range(labels.shape[0]):
        row_valid = valid_mask[row_index]
        n_valid = int(row_valid.sum())
        if n_valid <= 0:
            raise ValueError(
                f"Metriche batch: riga {row_index} senza token target validi"
            )
        nll_sum = float(token_nll[row_index, row_valid].sum())
        n_correct = int(
            (
                predictions[row_index, row_valid]
                == labels[row_index, row_valid]
            ).sum()
        )
        loss = nll_sum / n_valid
        accuracy = n_correct / n_valid
        rows.append(
            {
                "n_valid_tokens": n_valid,
                "n_correct_tokens": n_correct,
                "nll_sum": nll_sum,
                "loss": loss,
                "token_accuracy": accuracy,
                "perplexity": safe_perplexity(loss),
            }
        )
    return rows


def aggregate_group(
    group: pd.DataFrame,
    interaction_type: str,
    category_mode: str,
    type_order: int,
    directional_subtypes: str,
) -> dict[str, Any]:
    n_valid_tokens = int(group["n_valid_tokens"].sum())
    n_correct_tokens = int(group["n_correct_tokens"].sum())
    nll_sum = float(group["nll_sum"].sum())
    if n_valid_tokens <= 0:
        raise ValueError(
            f"Aggregazione {interaction_type}: nessun token valido"
        )
    loss = nll_sum / n_valid_tokens
    token_accuracy = n_correct_tokens / n_valid_tokens
    return {
        "type_order": int(type_order),
        "interaction_type": interaction_type,
        "category_mode": category_mode,
        "directional_subtypes": directional_subtypes,
        "n_pairs": int(len(group)),
        "n_unique_pairs": int(
            group[
                ["RNA_sequence_x", "RNA_sequence_y"]
            ].drop_duplicates().shape[0]
        ),
        "n_valid_tokens": n_valid_tokens,
        "n_correct_tokens": n_correct_tokens,
        "nll_sum": nll_sum,
        "loss": loss,
        "token_accuracy": token_accuracy,
        "perplexity": safe_perplexity(loss),
    }


def aggregate_by_type(
    details: pd.DataFrame,
    test_reference: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for reference_row in test_reference.to_dict("records"):
        interaction_type = str(reference_row["interaction_type"])
        group = details.loc[
            details["interaction_type"].eq(interaction_type)
        ]
        if group.empty:
            raise ValueError(
                f"Summary: nessuna coppia per {interaction_type}"
            )
        rows.append(
            aggregate_group(
                group,
                interaction_type=interaction_type,
                category_mode="non_directional_test_only",
                type_order=int(reference_row["type_order"]),
                directional_subtypes=str(
                    reference_row["directional_subtypes"]
                ),
            )
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def aggregate_directional(details: pd.DataFrame) -> pd.DataFrame:
    counts = (
        details.groupby("Category_Couple", sort=False)
        .size()
        .rename("n_pairs")
        .reset_index()
        .sort_values(
            ["n_pairs", "Category_Couple"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []
    for type_order, category_couple in enumerate(
        counts["Category_Couple"],
        start=1,
    ):
        group = details.loc[
            details["Category_Couple"].eq(category_couple)
        ]
        rows.append(
            aggregate_group(
                group,
                interaction_type=str(category_couple),
                category_mode="directional_test_only",
                type_order=type_order,
                directional_subtypes=str(category_couple),
            )
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_overall(details: pd.DataFrame) -> pd.DataFrame:
    n_valid_tokens = int(details["n_valid_tokens"].sum())
    n_correct_tokens = int(details["n_correct_tokens"].sum())
    nll_sum = float(details["nll_sum"].sum())
    if n_valid_tokens <= 0:
        raise ValueError("Overall: nessun token valido")
    loss = nll_sum / n_valid_tokens
    token_accuracy = n_correct_tokens / n_valid_tokens
    row = {
        "scope": "ALL_TEST_ROWS",
        "n_pairs": int(len(details)),
        "n_unique_pairs": int(
            details[
                ["RNA_sequence_x", "RNA_sequence_y"]
            ].drop_duplicates().shape[0]
        ),
        "n_interaction_types": int(details["interaction_type"].nunique()),
        "n_directional_types": int(details["Category_Couple"].nunique()),
        "n_valid_tokens": n_valid_tokens,
        "n_correct_tokens": n_correct_tokens,
        "nll_sum": nll_sum,
        "loss": loss,
        "token_accuracy": token_accuracy,
        "perplexity": safe_perplexity(loss),
    }
    return pd.DataFrame([row], columns=OVERALL_COLUMNS)


def validate_aggregates(
    details: pd.DataFrame,
    summary_types: pd.DataFrame,
    summary_directional: pd.DataFrame,
    overall: pd.DataFrame,
    expected_types: int,
    expected_directional_types: int,
) -> None:
    require_expected_count(
        len(summary_types),
        expected_types,
        "Righe summary non direzionale",
    )
    require_expected_count(
        len(summary_directional),
        expected_directional_types,
        "Righe summary direzionale",
    )
    if int(summary_types["n_pairs"].sum()) != len(details):
        raise RuntimeError(
            "Summary non direzionale: totale coppie incoerente"
        )
    if int(summary_directional["n_pairs"].sum()) != len(details):
        raise RuntimeError("Summary direzionale: totale coppie incoerente")

    overall_row = overall.iloc[0]
    expected_tokens = int(details["n_valid_tokens"].sum())
    expected_correct = int(details["n_correct_tokens"].sum())
    expected_nll = float(details["nll_sum"].sum())
    if int(overall_row["n_valid_tokens"]) != expected_tokens:
        raise RuntimeError("Overall: totale token incoerente")
    if int(overall_row["n_correct_tokens"]) != expected_correct:
        raise RuntimeError("Overall: totale token corretti incoerente")
    if not math.isclose(
        float(overall_row["nll_sum"]),
        expected_nll,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Overall: NLL totale incoerente")

    for summary, label in [
        (summary_types, "non direzionale"),
        (summary_directional, "direzionale"),
    ]:
        if int(summary["n_valid_tokens"].sum()) != expected_tokens:
            raise RuntimeError(f"Summary {label}: totale token incoerente")
        if int(summary["n_correct_tokens"].sum()) != expected_correct:
            raise RuntimeError(
                f"Summary {label}: totale token corretti incoerente"
            )
        if not math.isclose(
            float(summary["nll_sum"].sum()),
            expected_nll,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"Summary {label}: NLL totale incoerente")


def make_output_paths(
    prefix: str | Path,
    n_types: int,
    n_directional_types: int,
) -> OutputPaths:
    prefix_path = Path(prefix)
    return OutputPaths(
        details=Path(f"{prefix_path}_details.tsv"),
        summary_types=Path(
            f"{prefix_path}_summary_{n_types}_types.tsv"
        ),
        summary_directional=Path(
            f"{prefix_path}_summary_"
            f"{n_directional_types}_directional_types.tsv"
        ),
        overall=Path(f"{prefix_path}_overall.tsv"),
        metadata=Path(f"{prefix_path}_metadata.json"),
    )


def check_output_targets(paths: OutputPaths, overwrite: bool) -> None:
    output_paths = list(paths.__dict__.values())
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("Output: path duplicati")
    existing = [path for path in output_paths if path.exists()]
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


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
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
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def resolve_device(torch_module: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    device = torch_module.device(requested)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError(
            f"Device richiesto {requested}, ma CUDA non e' disponibile"
        )
    return device


def resolve_precision(
    torch_module: Any,
    device: Any,
    requested: str,
) -> str:
    if requested == "auto":
        if (
            device.type == "cuda"
            and hasattr(torch_module.cuda, "is_bf16_supported")
            and torch_module.cuda.is_bf16_supported()
        ):
            return "bf16"
        return "fp32"
    if requested == "bf16":
        if device.type != "cuda":
            raise ValueError("BF16 e' supportato qui soltanto su CUDA")
        if (
            hasattr(torch_module.cuda, "is_bf16_supported")
            and not torch_module.cuda.is_bf16_supported()
        ):
            raise ValueError("La GPU selezionata non supporta BF16")
    return requested


def evaluate_model(
    test: pd.DataFrame,
    model_dir: str | Path,
    batch_size: int,
    device_name: str,
    precision_name: str,
    live_every: int,
) -> tuple[pd.DataFrame, str, str, int]:
    if batch_size <= 0:
        raise ValueError("batch_size deve essere maggiore di zero")
    if live_every < 0:
        raise ValueError("live_every non puo' essere negativo")

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch non disponibile. Esegui lo script nell'ambiente B4 "
            "usato per il modello."
        ) from exc

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from model import NucConfig, NucTransformer
    from tokenizer import tokenize_batch, tokenizer

    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Checkpoint non trovato: {model_path}")

    device = resolve_device(torch, device_name)
    precision = resolve_precision(torch, device, precision_name)

    config = NucConfig()
    config.pad_token_id = tokenizer.vocab["PAD"]
    config.eos_token_id = tokenizer.vocab["EOS"]
    config.vocab_size = len(tokenizer.vocab)
    model = NucTransformer.from_pretrained(model_path, config=config)
    model.to(device)
    model.eval()

    print(f"Checkpoint            : {model_path}")
    print(f"Device                : {device}")
    print(f"Precisione forward    : {precision}")
    print(f"PAD token id          : {tokenizer.pad_token_id}")
    print("PAD nelle metriche    : ESCLUSO")

    metric_rows: list[dict[str, float | int]] = []
    processed = 0
    with torch.inference_mode():
        for start in range(0, len(test), batch_size):
            stop = min(start + batch_size, len(test))
            batch = test.iloc[start:stop]
            encoded = tokenize_batch(
                {
                    "RNA_sequence_x": batch["RNA_sequence_x"].tolist(),
                    "RNA_sequence_y": batch["RNA_sequence_y"].tolist(),
                }
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            decoder_input_ids = encoded["decoder_input_ids"].to(device)
            labels = encoded["labels"].to(device)

            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                )
                if precision == "bf16"
                else nullcontext()
            )
            with autocast_context:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    labels=None,
                )
                logits = outputs["logits"]

            token_nll = functional.cross_entropy(
                logits.float().transpose(1, 2),
                labels,
                reduction="none",
            )
            predictions = logits.argmax(dim=-1)
            metric_rows.extend(
                batch_row_metrics_from_arrays(
                    token_nll=token_nll.detach().cpu().numpy(),
                    predictions=predictions.detach().cpu().numpy(),
                    labels=labels.detach().cpu().numpy(),
                    pad_token_id=int(tokenizer.pad_token_id),
                )
            )
            processed = stop

            if live_every > 0 and (
                processed == len(test)
                or processed % live_every < batch_size
            ):
                partial = pd.DataFrame(metric_rows)
                partial_tokens = int(partial["n_valid_tokens"].sum())
                partial_correct = int(partial["n_correct_tokens"].sum())
                partial_nll = float(partial["nll_sum"].sum())
                partial_loss = partial_nll / partial_tokens
                print(
                    f"[{processed}/{len(test)}] "
                    f"loss={partial_loss:.6f} "
                    f"accuracy={100.0 * partial_correct / partial_tokens:.2f}% "
                    f"ppl={safe_perplexity(partial_loss):.6f}"
                )

    if len(metric_rows) != len(test):
        raise RuntimeError(
            f"Valutazione incompleta: {len(metric_rows)}/{len(test)} righe"
        )
    return (
        pd.DataFrame(metric_rows),
        str(device),
        precision,
        int(tokenizer.pad_token_id),
    )


def print_results(
    summary_types: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    overall_row = overall.iloc[0]
    print("\n=== TEACHER FORCING: TOTALE TEST ===")
    print(f"Coppie              : {int(overall_row['n_pairs'])}")
    print(f"Coppie uniche       : {int(overall_row['n_unique_pairs'])}")
    print(f"Token non-PAD       : {int(overall_row['n_valid_tokens'])}")
    print(f"Loss                : {float(overall_row['loss']):.6f}")
    print(
        "Token accuracy      : "
        f"{100.0 * float(overall_row['token_accuracy']):.2f}%"
    )
    print(
        f"Perplexity          : {float(overall_row['perplexity']):.6f}"
    )

    print("\n=== TEACHER FORCING: 14 TIPI DEL TEST ===")
    for row in summary_types.itertuples(index=False):
        print(
            f"[{int(row.type_order):02d}/{len(summary_types)}] "
            f"{row.interaction_type} | "
            f"coppie={int(row.n_pairs)} | "
            f"token={int(row.n_valid_tokens)} | "
            f"loss={float(row.loss):.6f} | "
            f"accuracy={100.0 * float(row.token_accuracy):.2f}% | "
            f"ppl={float(row.perplexity):.6f}"
        )


def run_evaluation(args: argparse.Namespace) -> OutputPaths:
    outputs = make_output_paths(
        args.output_prefix,
        n_types=args.expected_types,
        n_directional_types=args.expected_directional_types,
    )
    check_output_targets(outputs, overwrite=args.overwrite)

    print("=== B4 TEACHER FORCING PER TIPO DI INTERAZIONE ===")
    print(f"Test TSV              : {Path(args.test_file)}")
    print(f"Heatmap categorie     : {Path(args.heatmap_counts)}")
    print(f"Prefisso output       : {Path(args.output_prefix)}")

    test, test_reference = load_and_validate_test(
        test_file=args.test_file,
        heatmap_counts=args.heatmap_counts,
        expected_rows=args.expected_rows,
        expected_unique_pairs=args.expected_unique_pairs,
        expected_types=args.expected_types,
        expected_directional_types=args.expected_directional_types,
        expected_heatmap_types=args.expected_heatmap_types,
    )
    print(
        "Test validato         : "
        f"{len(test)} righe, "
        f"{test[['RNA_sequence_x', 'RNA_sequence_y']].drop_duplicates().shape[0]} "
        f"coppie uniche, "
        f"{test['interaction_type'].nunique()} tipi"
    )

    (
        metric_rows,
        resolved_device,
        resolved_precision,
        pad_token_id,
    ) = evaluate_model(
        test=test,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        device_name=args.device,
        precision_name=args.precision,
        live_every=args.live_every,
    )
    details = pd.concat(
        [
            test.reset_index(drop=True),
            metric_rows.reset_index(drop=True),
        ],
        axis=1,
    )
    require_columns(details, DETAIL_METRIC_COLUMNS, "Dettagli")
    if (details["n_valid_tokens"] <= 0).any():
        raise RuntimeError("Dettagli: righe senza token target validi")
    if not details["token_accuracy"].between(0.0, 1.0).all():
        raise RuntimeError("Dettagli: token_accuracy fuori [0, 1]")

    summary_types = aggregate_by_type(details, test_reference)
    summary_directional = aggregate_directional(details)
    overall = build_overall(details)
    validate_aggregates(
        details,
        summary_types,
        summary_directional,
        overall,
        expected_types=args.expected_types,
        expected_directional_types=args.expected_directional_types,
    )

    details_output = details.drop(columns=["_interaction_key"])
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(Path(args.model_dir).resolve()),
        "test_file": str(Path(args.test_file).resolve()),
        "heatmap_counts": str(Path(args.heatmap_counts).resolve()),
        "device": resolved_device,
        "precision": resolved_precision,
        "batch_size": int(args.batch_size),
        "pad_token_id": pad_token_id,
        "pad_excluded": True,
        "formulas": {
            "loss": "total_nll_non_pad / total_non_pad_tokens",
            "token_accuracy": (
                "correct_non_pad_tokens / total_non_pad_tokens"
            ),
            "perplexity": "exp(loss)",
        },
        "counts": {
            "test_rows": int(len(details)),
            "unique_pairs": int(
                details[
                    ["RNA_sequence_x", "RNA_sequence_y"]
                ].drop_duplicates().shape[0]
            ),
            "interaction_types": int(
                details["interaction_type"].nunique()
            ),
            "directional_types": int(
                details["Category_Couple"].nunique()
            ),
            "non_pad_tokens": int(details["n_valid_tokens"].sum()),
        },
    }

    write_tsv_atomic(details_output, outputs.details)
    write_tsv_atomic(summary_types, outputs.summary_types)
    write_tsv_atomic(summary_directional, outputs.summary_directional)
    write_tsv_atomic(overall, outputs.overall)
    write_json_atomic(metadata, outputs.metadata)

    print_results(summary_types, overall)
    print("\n=== OUTPUT SCRITTI E VERIFICATI ===")
    print(f"Dettagli per coppia   : {outputs.details}")
    print(f"Summary 14 tipi       : {outputs.summary_types}")
    print(f"Audit direzionale     : {outputs.summary_directional}")
    print(f"Totale test           : {outputs.overall}")
    print(f"Metadati/formule      : {outputs.metadata}")
    return outputs


def validate_inputs_only(args: argparse.Namespace) -> None:
    test, test_reference = load_and_validate_test(
        test_file=args.test_file,
        heatmap_counts=args.heatmap_counts,
        expected_rows=args.expected_rows,
        expected_unique_pairs=args.expected_unique_pairs,
        expected_types=args.expected_types,
        expected_directional_types=args.expected_directional_types,
        expected_heatmap_types=args.expected_heatmap_types,
    )
    print("VALIDAZIONE INPUT TEACHER FORCING: OK")
    print(f"Test TSV            : {Path(args.test_file)}")
    print(f"Righe test          : {len(test)}")
    print(
        "Coppie uniche      : "
        f"{test[['RNA_sequence_x', 'RNA_sequence_y']].drop_duplicates().shape[0]}"
    )
    print(f"Tipi non direzionali: {test['interaction_type'].nunique()}")
    print(f"Tipi direzionali   : {test['Category_Couple'].nunique()}")
    print("Categorie:")
    for row in test_reference.itertuples(index=False):
        n_pairs = int(
            test["interaction_type"].eq(row.interaction_type).sum()
        )
        print(
            f"  [{int(row.type_order):02d}/{len(test_reference)}] "
            f"{row.interaction_type}: {n_pairs} righe"
        )


def run_self_test() -> None:
    labels = np.array(
        [
            [3, 4, 0, 0],
            [5, 6, 3, 0],
        ],
        dtype=np.int64,
    )
    predictions = np.array(
        [
            [3, 3, 6, 6],
            [5, 4, 3, 1],
        ],
        dtype=np.int64,
    )
    token_nll = np.array(
        [
            [0.1, 0.3, 99.0, 99.0],
            [0.2, 0.4, 0.6, 99.0],
        ],
        dtype=np.float64,
    )
    rows = batch_row_metrics_from_arrays(
        token_nll=token_nll,
        predictions=predictions,
        labels=labels,
        pad_token_id=0,
    )
    if rows[0]["n_valid_tokens"] != 2:
        raise AssertionError("Self-test: PAD contato come token valido")
    if rows[0]["n_correct_tokens"] != 1:
        raise AssertionError("Self-test: token corretti riga 0 errati")
    if not math.isclose(float(rows[0]["nll_sum"]), 0.4):
        raise AssertionError("Self-test: NLL riga 0 include PAD")
    if not math.isclose(float(rows[0]["loss"]), 0.2):
        raise AssertionError("Self-test: loss riga 0 errata")
    if not math.isclose(float(rows[0]["token_accuracy"]), 0.5):
        raise AssertionError("Self-test: accuracy riga 0 errata")

    details = pd.DataFrame(
        [
            {
                "RNA_sequence_x": "AAAA",
                "RNA_sequence_y": "UUUU",
                "Category_Couple": "miRNA - snoRNA",
                "interaction_type": "miRNA - snoRNA",
                **rows[0],
            },
            {
                "RNA_sequence_x": "CCCC",
                "RNA_sequence_y": "GGGG",
                "Category_Couple": "snoRNA - miRNA",
                "interaction_type": "miRNA - snoRNA",
                **rows[1],
            },
        ]
    )
    reference = pd.DataFrame(
        [
            {
                "type_order": 1,
                "interaction_type": "miRNA - snoRNA",
                "directional_subtypes": (
                    "miRNA - snoRNA | snoRNA - miRNA"
                ),
            }
        ]
    )
    summary = aggregate_by_type(details, reference)
    overall = build_overall(details)
    expected_tokens = 5
    expected_correct = 3
    expected_nll = 1.6
    expected_loss = expected_nll / expected_tokens
    if int(overall.iloc[0]["n_valid_tokens"]) != expected_tokens:
        raise AssertionError("Self-test: totale token non-PAD errato")
    if int(overall.iloc[0]["n_correct_tokens"]) != expected_correct:
        raise AssertionError("Self-test: totale token corretti errato")
    if not math.isclose(float(overall.iloc[0]["nll_sum"]), expected_nll):
        raise AssertionError("Self-test: NLL totale errata")
    if not math.isclose(float(overall.iloc[0]["loss"]), expected_loss):
        raise AssertionError("Self-test: loss totale errata")
    if not math.isclose(
        float(overall.iloc[0]["token_accuracy"]),
        expected_correct / expected_tokens,
    ):
        raise AssertionError("Self-test: accuracy totale errata")
    if not math.isclose(
        float(overall.iloc[0]["perplexity"]),
        math.exp(expected_loss),
    ):
        raise AssertionError("Self-test: perplexity totale errata")
    if not math.isclose(
        float(summary.iloc[0]["loss"]),
        expected_loss,
    ):
        raise AssertionError("Self-test: loss di categoria errata")

    print("SELF-TEST TEACHER FORCING: OK")
    print("PAD esclusi da NLL, accuracy e perplexity: OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valuta il modello B4 in teacher forcing sul test completo e "
            "per le 14 categorie, escludendo sempre i PAD."
        )
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        default=DEFAULT_TEST_FILE,
    )
    parser.add_argument(
        "--heatmap_counts",
        type=Path,
        default=DEFAULT_HEATMAP_COUNTS,
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, cuda:1, ...",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "bf16"],
        default="auto",
    )
    parser.add_argument(
        "--live_every",
        type=int,
        default=100,
        help="Stampa un riepilogo progressivo ogni N righe; 0 disabilita.",
    )
    parser.add_argument(
        "--expected_rows",
        type=int,
        default=DEFAULT_EXPECTED_ROWS,
    )
    parser.add_argument(
        "--expected_unique_pairs",
        type=int,
        default=DEFAULT_EXPECTED_UNIQUE_PAIRS,
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
        "--expected_heatmap_types",
        type=int,
        default=DEFAULT_EXPECTED_HEATMAP_TYPES,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sostituisce soltanto gli output del nuovo script.",
    )
    parser.add_argument(
        "--self_test",
        action="store_true",
        help="Verifica formule e maschera PAD senza caricare il modello.",
    )
    parser.add_argument(
        "--validate_inputs_only",
        action="store_true",
        help="Valida il vero test e le 14 categorie senza caricare il modello.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.validate_inputs_only:
        validate_inputs_only(args)
        return
    run_evaluation(args)


if __name__ == "__main__":
    main()
