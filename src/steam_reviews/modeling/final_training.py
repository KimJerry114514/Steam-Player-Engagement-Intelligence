"""Full-cohort LightGBM+NLP refit and text export."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import nbformat
import pandas as pd
import pyarrow.dataset as ds
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


TARGET = "still_playing_d30"
MODEL_NAME = "lightgbm_structured_nlp"
RANDOM_SEED = 42
TRAIN_START = pd.Timestamp("2010-10-15", tz="UTC")
TRAIN_END = pd.Timestamp("2022-01-01", tz="UTC")
ENGINEERED_SOURCES = {
    "log_hours_at_review": "hours_at_review",
    "review_year": "review_created_utc",
    "review_month": "review_created_utc",
    "review_dayofweek": "review_created_utc",
    "review_quarter_of_year": "review_created_utc",
}


def prepare_training_frame(df_merged: pd.DataFrame, input_features: Sequence[str]) -> pd.DataFrame:
    """Select all dated, binary-labeled rows, including edited reviews."""
    features = list(input_features)
    if not features or len(features) != len(set(features)) or "language" not in features:
        raise ValueError("Expected unique input features including language")
    raw_features = [ENGINEERED_SOURCES.get(name, name) for name in features]
    required = set(raw_features) | {"review_created_utc", TARGET}
    missing = sorted(required.difference(df_merged.columns))
    if missing:
        raise ValueError(f"Merged source is missing columns: {missing}")
    dates = pd.to_datetime(df_merged["review_created_utc"], utc=True, errors="raise")
    labels = pd.to_numeric(df_merged[TARGET], errors="coerce")
    eligible = dates.ge(TRAIN_START) & dates.lt(TRAIN_END) & labels.isin([0, 1])
    columns = list(dict.fromkeys(["review_created_utc", TARGET, *raw_features]))
    training = df_merged.loc[eligible, columns].copy()
    if training.empty:
        raise ValueError("No eligible training rows in the requested time window")
    if "review_year" in features:
        training["review_year"] = dates.loc[eligible].dt.year.astype("float32")
    if "review_month" in features:
        training["review_month"] = dates.loc[eligible].dt.month.astype("float32")
    if "review_dayofweek" in features:
        training["review_dayofweek"] = dates.loc[eligible].dt.dayofweek.astype("float32")
    if "review_quarter_of_year" in features:
        training["review_quarter_of_year"] = dates.loc[eligible].dt.quarter.astype("float32")
    if "log_hours_at_review" in features:
        hours = pd.to_numeric(training["hours_at_review"], errors="coerce").clip(lower=0)
        training["log_hours_at_review"] = np.log1p(hours).astype("float32")
    for feature in features:
        if feature != "language":
            training[feature] = pd.to_numeric(training[feature], errors="coerce").astype("float32")
    training["language"] = training["language"].astype(object)
    training[TARGET] = labels.loc[eligible].astype("int8")
    return training[[TARGET, *features]]


def train_frame_and_export(
    *,
    training: pd.DataFrame,
    model_path: Path,
    metadata_path: Path,
    input_features: Sequence[str],
    rounds: int,
    threads: int,
    source_sha256: str,
    staging_signature: str,
) -> dict:
    """Fit the approved in-memory sklearn pipeline and export its Booster."""
    if rounds < 1 or threads < 1:
        raise ValueError("rounds and threads must be positive")
    features = list(input_features)
    numeric = [name for name in features if name != "language"]
    preprocessor = ColumnTransformer(
        [
            (
                "language",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
                ["language"],
            ),
            ("numeric", "passthrough", numeric),
        ],
        verbose_feature_names_out=False,
    )
    matrix = preprocessor.fit_transform(training[features])
    transformed_features = ["language", *numeric]
    parameters = {
        "objective": "binary",
        "n_estimators": int(rounds),
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 500,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_SEED,
        "n_jobs": int(threads),
        "verbosity": -1,
    }
    model = lgb.LGBMClassifier(**parameters)
    model.fit(matrix, training[TARGET].to_numpy(), feature_name=transformed_features)
    booster = model.booster_
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = output.with_name(output.name + ".tmp")
    booster.save_model(str(temporary_model))
    reloaded = lgb.Booster(model_file=str(temporary_model))
    if reloaded.current_iteration() != booster.current_iteration():
        raise AssertionError("Exported LightGBM text model did not reload with all trees")
    os.replace(temporary_model, output)
    language_pipeline = preprocessor.named_transformers_["language"]
    categories = language_pipeline.named_steps["encode"].categories_[0].tolist()
    fill_value = str(language_pipeline.named_steps["fill"].statistics_[0])
    metadata = {
        "model": MODEL_NAME,
        "rows": len(training),
        "positive_rows": int(training[TARGET].sum()),
        "input_features": features,
        "transformed_features": transformed_features,
        "language_categories": categories,
        "language_fill_value": fill_value,
        "unknown_language_code": -1,
        "rounds_requested": int(rounds),
        "rounds_trained": int(reloaded.current_iteration()),
        "parameters": parameters,
        "source_sha256": source_sha256,
        "cv_reference_staging_signature": staging_signature,
        "training_cohort": {
            "start_utc": TRAIN_START.isoformat(),
            "end_exclusive_utc": TRAIN_END.isoformat(),
            "target": TARGET,
            "target_values": [0, 1],
            "edited_filter_applied": False,
        },
        "model_txt_sha256": _sha256_file(output),
        "package_versions": {
            "lightgbm": version("lightgbm"),
            "numpy": version("numpy"),
            "pyarrow": version("pyarrow"),
            "scikit-learn": version("scikit-learn"),
        },
    }
    manifest_path = Path(metadata_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return metadata


def language_contract(dataset: ds.Dataset, batch_size: int = 65_536) -> tuple[list[str], str]:
    """Reproduce the mode imputer and sorted OrdinalEncoder categories."""
    counts: Counter[str] = Counter()
    for batch in dataset.scanner(columns=["language"], batch_size=batch_size).to_batches():
        values = batch.column(0).to_pandas().dropna().value_counts()
        counts.update({str(language): int(count) for language, count in values.items()})
    if not counts:
        raise ValueError("No nonmissing language values in the training cohort")
    categories = sorted(counts)
    most_frequent = min(categories, key=lambda language: (-counts[language], language))
    return categories, most_frequent


def write_training_arrays(
    dataset: ds.Dataset,
    input_features: Sequence[str],
    directory: Path,
    expected_rows: int,
    categories: Sequence[str],
    fill_value: str,
    batch_size: int = 32_768,
) -> tuple[Path, Path, int]:
    """Stage only the selected model's transformed columns as disk-backed arrays."""
    if len(set(input_features)) != len(input_features) or "language" not in input_features:
        raise ValueError("Expected a unique feature list containing language")
    if expected_rows < 1 or fill_value not in categories:
        raise ValueError("Invalid training row count or language imputation value")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    matrix_path = destination / "training-matrix.npy"
    labels_path = destination / "training-labels.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=np.float32, shape=(expected_rows, len(input_features))
    )
    labels = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.int8, shape=(expected_rows,)
    )
    numeric = [feature for feature in input_features if feature != "language"]
    category_codes = {name: code for code, name in enumerate(categories)}
    offset = 0
    positive_rows = 0
    scanner = dataset.scanner(
        columns=[TARGET, "language", *numeric], batch_size=batch_size, use_threads=True
    )
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        row_count = len(frame)
        end = offset + row_count
        if end > expected_rows:
            raise AssertionError("Staged data exceeds the audited full-cohort row count")
        label_values = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(dtype=np.int8)
        if not np.isin(label_values, [0, 1]).all():
            raise AssertionError("Staging contains a non-binary target")
        language = frame["language"].fillna(fill_value).map(category_codes)
        if language.isna().any():
            raise AssertionError("Language categories changed during staging")
        matrix[offset:end, 0] = language.to_numpy(dtype=np.float32)
        matrix[offset:end, 1:] = frame[numeric].to_numpy(dtype=np.float32, na_value=np.nan)
        labels[offset:end] = label_values
        positive_rows += int(label_values.sum())
        offset = end
    matrix.flush()
    labels.flush()
    del matrix, labels
    if offset != expected_rows:
        raise AssertionError(f"Expected {expected_rows:,} rows but staged {offset:,}")
    return matrix_path, labels_path, positive_rows


class MemmapSequence(lgb.Sequence):
    """Supply bounded slices of the staged array to LightGBM."""

    batch_size = 8192

    def __init__(self, matrix_path: Path):
        self.data = np.load(matrix_path, mmap_mode="r")

    def __getitem__(self, index):
        return np.asarray(self.data[index], dtype=np.float64)

    def __len__(self):
        return len(self.data)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_and_export(
    *,
    matrix_path: Path,
    labels_path: Path,
    model_path: Path,
    metadata_path: Path,
    input_features: Sequence[str],
    categories: Sequence[str],
    fill_value: str,
    rounds: int,
    threads: int,
    source_sha256: str,
    staging_signature: str,
) -> dict:
    """Fit the approved LightGBM configuration and save a reloadable .txt Booster."""
    if rounds < 1 or threads < 1:
        raise ValueError("rounds and threads must be positive")
    sequence = MemmapSequence(matrix_path)
    labels = np.load(labels_path, mmap_mode="r")
    if len(labels) != len(sequence):
        raise AssertionError("Training matrix and labels have different row counts")
    transformed_features = ["language", *[name for name in input_features if name != "language"]]
    if sequence.data.shape[1] != len(transformed_features):
        raise AssertionError("Training matrix column count does not match feature list")
    dataset = lgb.Dataset(
        sequence,
        label=labels,
        feature_name=transformed_features,
        free_raw_data=True,
    )
    parameters = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 500,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "feature_fraction": 0.8,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "seed": RANDOM_SEED,
        "num_threads": threads,
        "verbosity": -1,
    }
    booster = lgb.train(parameters, dataset, num_boost_round=rounds)
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = output.with_name(output.name + ".tmp")
    booster.save_model(str(temporary_model))
    reloaded = lgb.Booster(model_file=str(temporary_model))
    if reloaded.current_iteration() != booster.current_iteration():
        raise AssertionError("Exported LightGBM text model did not reload with all trees")
    os.replace(temporary_model, output)
    metadata = {
        "model": MODEL_NAME,
        "rows": len(sequence),
        "positive_rows": int(np.count_nonzero(labels)),
        "input_features": list(input_features),
        "transformed_features": transformed_features,
        "language_categories": list(categories),
        "language_fill_value": fill_value,
        "unknown_language_code": -1,
        "rounds_requested": int(rounds),
        "rounds_trained": int(reloaded.current_iteration()),
        "parameters": parameters,
        "source_sha256": source_sha256,
        "staging_signature": staging_signature,
        "model_txt_sha256": _sha256_file(output),
        "package_versions": {
            "lightgbm": version("lightgbm"),
            "numpy": version("numpy"),
            "pyarrow": version("pyarrow"),
            "scikit-learn": version("scikit-learn"),
        },
    }
    manifest_path = Path(metadata_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return metadata


def check_resources(row_count: int, feature_count: int, output_dir: Path) -> dict[str, float]:
    """Stop before staging when available disk or RAM is too low."""
    matrix_gib = row_count * feature_count * 4 / 1024**3
    available_ram_gib = psutil.virtual_memory().available / 1024**3
    free_disk_gib = psutil.disk_usage(output_dir).free / 1024**3
    required_disk_gib = matrix_gib + 2.0
    if available_ram_gib < 5.0:
        raise RuntimeError(
            f"Full training needs at least 5.0 GiB available RAM; found {available_ram_gib:.2f} GiB"
        )
    if free_disk_gib < required_disk_gib:
        raise RuntimeError(
            f"Full training needs {required_disk_gib:.2f} GiB free disk; found {free_disk_gib:.2f} GiB"
        )
    return {
        "matrix_gib": matrix_gib,
        "available_ram_gib": available_ram_gib,
        "free_disk_gib": free_disk_gib,
    }


def load_approved_contract(project_root: Path) -> dict:
    """Check full-CV evidence and return the exact winning-model configuration."""
    import sys

    models_dir = Path(project_root) / "src"
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    from steam_reviews.modeling.training_cv_disk import (
        feature_signature,
        model_signature,
        sha256_file,
    )

    artifact_dir = Path(project_root) / "results" / "cross_validation"
    source_notebook = nbformat.read(Path(project_root) / "notebooks" / "02_model_comparison_cv.ipynb", as_version=4)
    definitions = {}
    for cell in source_notebook.cells:
        if cell.cell_type != "code":
            continue
        for statement in ast.parse(cell.source).body:
            if not isinstance(statement, ast.Assign):
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in {"MODEL_FEATURES", "MODEL_PARAMETERS"}:
                    definitions[target.id] = ast.literal_eval(statement.value)
    if set(definitions) != {"MODEL_FEATURES", "MODEL_PARAMETERS"}:
        raise ValueError("Approved model definitions are missing from 02_model_comparison_cv.ipynb")
    model_features = definitions["MODEL_FEATURES"]
    model_parameters = definitions["MODEL_PARAMETERS"]
    run = json.loads((artifact_dir / "run_manifest.json").read_text(encoding="utf-8"))
    staging_dir = artifact_dir / "staging" / "full"
    stage = json.loads((staging_dir / "staging_manifest.json").read_text(encoding="utf-8"))
    source = Path(project_root) / "data" / "processed" / "negative_reviews_merged.parquet"
    if not run.get("full_run_completed") or run.get("folds") != 10:
        raise ValueError("Completed full 10-fold CV evidence is required")
    if sha256_file(source) != run["source_sha256"] or stage["source_sha256"] != run["source_sha256"]:
        raise ValueError("Training source differs from the completed CV source")
    if run["staging_signature"] != stage["signature"]:
        raise ValueError("Archived staging signature does not match CV")
    if sha256_file(artifact_dir / "fold_assignments.parquet") != run["fold_assignment_sha256"]:
        raise ValueError("Archived fold assignments do not match completed CV")
    if stage["row_count"] != run["rows"] or stage["positive_rows"] != run["positive_rows"]:
        raise ValueError("Full-cohort counts do not match completed CV")
    features = model_features[MODEL_NAME]
    if len(features) != 96:
        raise ValueError("Approved LightGBM+NLP feature count changed")
    feature_hash = feature_signature(features)
    parameter_hash = model_signature(MODEL_NAME, model_parameters[MODEL_NAME])
    checkpoint_dir = artifact_dir / "checkpoints" / "full"
    iterations = []
    for fold in range(10):
        metadata_path = checkpoint_dir / f"{MODEL_NAME}_fold_{fold}.json"
        if not metadata_path.is_file():
            raise ValueError(f"Fold {fold} checkpoint is missing")
        checkpoint = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "model": MODEL_NAME,
            "fold_id": fold,
            "source_sha256": run["source_sha256"],
            "fold_assignment_sha256": run["fold_assignment_sha256"],
            "staging_signature": stage["signature"],
            "feature_signature": feature_hash,
            "model_signature": parameter_hash,
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Fold {fold} checkpoint does not match approved model")
        iteration = checkpoint.get("best_iteration")
        if not isinstance(iteration, int) or iteration < 1:
            raise ValueError(f"Fold {fold} has no valid best_iteration")
        iterations.append(iteration)
    return {
        "source_path": source,
        "source_sha256": run["source_sha256"],
        "staging_dir": staging_dir,
        "staging_signature": stage["signature"],
        "row_count": stage["row_count"],
        "positive_rows": stage["positive_rows"],
        "input_features": features,
        "fold_iterations": iterations,
        "final_rounds": int(np.median(iterations)),
    }
