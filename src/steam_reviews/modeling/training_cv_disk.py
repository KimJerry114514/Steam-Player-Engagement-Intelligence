"""Disk-backed helpers for the full-data training cross-validation notebook."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold


RANDOM_SEED = 42
FULL_FOLDS = 10
SMOKE_FOLDS = 3
STAGING_FORMAT_VERSION = 1
TARGET = "still_playing_d30"
DATE_COLUMN = "review_created_utc"
TRAIN_START = pd.Timestamp("2010-10-15", tz="UTC")
TRAIN_END = pd.Timestamp("2022-01-01", tz="UTC")
OOF_METADATA_COLUMNS = [
    "stable_row_id",
    "recommendationid",
    DATE_COLUMN,
    TARGET,
    "fold_id",
]
ENGINEERED_FEATURE_SOURCES = {
    "review_year": DATE_COLUMN,
    "review_month": DATE_COLUMN,
    "review_dayofweek": DATE_COLUMN,
    "review_quarter_of_year": DATE_COLUMN,
    "log1p_hours_at_review": "hours_at_review",
    "log_hours_at_review": "hours_at_review",
}


def _env_bool(environ: Mapping[str, str], name: str, default: bool = False) -> bool:
    value = environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off")


@dataclass(frozen=True)
class RunConfig:
    project_root: Path
    run_mode: str
    random_seed: int
    n_outer_folds: int
    smoke_rows: int
    source_path: Path
    artifact_dir: Path
    full_preflight_only: bool
    run_full_refits: bool
    force_rebuild_staging: bool
    min_available_ram_gb: float
    min_free_disk_gb: float
    threads: int

    @property
    def staging_dir(self) -> Path:
        return self.artifact_dir / "staging" / self.run_mode

    @property
    def checkpoint_dir(self) -> Path:
        return self.artifact_dir / "checkpoints" / self.run_mode


def load_run_config(
    project_root: Path,
    environ: Mapping[str, str] | None = None,
) -> RunConfig:
    env = os.environ if environ is None else environ
    root = Path(project_root).resolve()
    run_mode = env.get("TRAINING_CV_RUN_MODE", "full").strip().lower()
    if run_mode not in {"smoke", "full"}:
        raise ValueError("TRAINING_CV_RUN_MODE must be 'smoke' or 'full'")
    n_outer_folds = FULL_FOLDS if run_mode == "full" else SMOKE_FOLDS
    source_path = Path(
        env.get(
            "TRAINING_CV_DATA_PATH",
            str(root / "data" / "processed" / "negative_reviews_merged.parquet"),
        )
    ).resolve()
    artifact_dir = Path(
        env.get(
            "TRAINING_CV_ARTIFACT_DIR",
            str(root / "runs" / "cross_validation"),
        )
    ).resolve()
    threads = int(env.get("TRAINING_CV_THREADS", "4"))
    if threads < 1:
        raise ValueError("TRAINING_CV_THREADS must be positive")
    smoke_rows = int(env.get("TRAINING_CV_SMOKE_ROWS", "20000"))
    if smoke_rows < n_outer_folds * 2:
        raise ValueError("TRAINING_CV_SMOKE_ROWS is too small for stratified folds")
    return RunConfig(
        project_root=root,
        run_mode=run_mode,
        random_seed=RANDOM_SEED,
        n_outer_folds=n_outer_folds,
        smoke_rows=smoke_rows,
        source_path=source_path,
        artifact_dir=artifact_dir,
        full_preflight_only=_env_bool(env, "TRAINING_CV_FULL_PREFLIGHT_ONLY"),
        run_full_refits=_env_bool(env, "TRAINING_CV_RUN_FULL_REFITS"),
        force_rebuild_staging=_env_bool(env, "TRAINING_CV_FORCE_REBUILD_STAGING"),
        min_available_ram_gb=float(env.get("TRAINING_CV_MIN_AVAILABLE_RAM_GB", "12")),
        min_free_disk_gb=float(env.get("TRAINING_CV_MIN_FREE_DISK_GB", "20")),
        threads=threads,
    )


def generate_fold_ids(labels: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int8)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("labels must contain exact binary values 0 and 1")
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(len(y), dtype=np.int8)
    for fold_id, (_, held_out) in enumerate(splitter.split(dummy, y)):
        fold_ids[held_out] = fold_id
    if set(fold_ids.tolist()) != set(range(n_splits)):
        raise AssertionError("fold generation did not assign every requested fold")
    return fold_ids


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Unsupported signature value: {type(value).__name__}")


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_signature(feature_names: Sequence[str]) -> str:
    return stable_json_sha256(list(feature_names))


def model_signature(model_name: str, parameters: Mapping[str, Any]) -> str:
    return stable_json_sha256({"model_name": model_name, "parameters": dict(parameters)})


@dataclass(frozen=True)
class StagingManifest:
    format_version: int
    signature: str
    source_path: str
    source_sha256: str
    run_mode: str
    random_seed: int
    n_folds: int
    row_count: int
    positive_rows: int
    fold_assignment_sha256: str
    staged_columns: list[str]
    dtypes: dict[str, str]
    partition_rows: dict[str, int]
    max_modeling_date: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: Path) -> "StagingManifest":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _expected_staging_signature(
    config: RunConfig,
    source_path: Path,
    source_sha256: str,
    model_features: Mapping[str, Sequence[str]],
    required_metadata: Sequence[str],
) -> str:
    return stable_json_sha256(
        {
            "format_version": STAGING_FORMAT_VERSION,
            "run_mode": config.run_mode,
            "random_seed": config.random_seed,
            "n_folds": config.n_outer_folds,
            "smoke_rows": config.smoke_rows if config.run_mode == "smoke" else None,
            "source_path": str(Path(source_path).resolve()),
            "source_sha256": source_sha256,
            "model_features": {name: list(values) for name, values in model_features.items()},
            "required_metadata": list(required_metadata),
            "train_start": TRAIN_START.isoformat(),
            "train_end": TRAIN_END.isoformat(),
        }
    )


def validate_staging_manifest(staging_dir: Path, expected_signature: str) -> bool:
    root = Path(staging_dir)
    manifest_path = root / "staging_manifest.json"
    cohort_dir = root / "cohort"
    if not manifest_path.is_file() or not cohort_dir.is_dir():
        return False
    try:
        manifest = StagingManifest.from_path(manifest_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if manifest.signature != expected_signature:
        return False
    if manifest.format_version != STAGING_FORMAT_VERSION:
        return False
    if sum(manifest.partition_rows.values()) != manifest.row_count:
        return False
    if set(manifest.partition_rows) != {str(i) for i in range(manifest.n_folds)}:
        return False
    if not all(value > 0 for value in manifest.partition_rows.values()):
        return False
    try:
        for fold_id, expected_rows in manifest.partition_rows.items():
            parquet_files = list((cohort_dir / f"fold_id={fold_id}").glob("*.parquet"))
            if not parquet_files:
                return False
            actual_rows = sum(
                pq.ParquetFile(path).metadata.num_rows for path in parquet_files
            )
            if actual_rows != expected_rows:
                return False

        artifact_dir = root.parent.parent
        suffix = "" if manifest.run_mode == "full" else "_smoke"
        fold_path = artifact_dir / f"fold_assignments{suffix}.parquet"
        fold_hash_path = artifact_dir / f"fold_assignments{suffix}.sha256"
        if not fold_path.is_file() or not fold_hash_path.is_file():
            return False
        if sha256_file(fold_path) != manifest.fold_assignment_sha256:
            return False
        expected_sidecar = f"{manifest.fold_assignment_sha256}  {fold_path.name}\n"
        if fold_hash_path.read_text(encoding="utf-8") != expected_sidecar:
            return False
    except (OSError, ValueError):
        return False
    return True


def _date_filter(dataset: ds.Dataset) -> Any:
    timestamp_type = dataset.schema.field(DATE_COLUMN).type
    return (
        ds.field(DATE_COLUMN) >= pa.scalar(TRAIN_START.to_pydatetime(), timestamp_type)
    ) & (ds.field(DATE_COLUMN) < pa.scalar(TRAIN_END.to_pydatetime(), timestamp_type))


def _choose_smoke_positions(labels: np.ndarray, requested_rows: int, seed: int) -> np.ndarray:
    if len(labels) <= requested_rows:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    priorities = rng.random(len(labels))
    positive_rows = int(round(requested_rows * float(labels.mean())))
    targets = {1: positive_rows, 0: requested_rows - positive_rows}
    chosen: list[np.ndarray] = []
    for label, count in targets.items():
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < count:
            raise ValueError(f"Not enough class {label} rows for the smoke sample")
        order = np.argsort(priorities[candidates], kind="stable")[:count]
        chosen.append(candidates[order])
    return np.sort(np.concatenate(chosen).astype(np.int64))


def _engineer_staged_features(frame: pd.DataFrame, feature_union: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], utc=True, errors="raise")
    if "review_year" in feature_union:
        result["review_year"] = result[DATE_COLUMN].dt.year.astype("float32")
    if "review_month" in feature_union:
        result["review_month"] = result[DATE_COLUMN].dt.month.astype("float32")
    if "review_dayofweek" in feature_union:
        result["review_dayofweek"] = result[DATE_COLUMN].dt.dayofweek.astype("float32")
    if "review_quarter_of_year" in feature_union:
        result["review_quarter_of_year"] = result[DATE_COLUMN].dt.quarter.astype("float32")
    if {"log1p_hours_at_review", "log_hours_at_review"} & set(feature_union):
        hours = pd.to_numeric(result["hours_at_review"], errors="coerce").clip(lower=0)
        logged = np.log1p(hours).astype("float32")
        if "log1p_hours_at_review" in feature_union:
            result["log1p_hours_at_review"] = logged
        if "log_hours_at_review" in feature_union:
            result["log_hours_at_review"] = logged
    for feature in feature_union:
        if feature != "language":
            result[feature] = pd.to_numeric(result[feature], errors="coerce").astype("float32")
    if "language" in result:
        result["language"] = result["language"].astype(object)
    result[TARGET] = pd.to_numeric(result[TARGET], errors="raise").astype("int8")
    return result


def _assert_pass_two_sequence(
    expected_labels: np.ndarray,
    expected_recommendations: np.ndarray,
    expected_dates: np.ndarray,
    observed_labels: np.ndarray,
    observed_recommendations: np.ndarray,
    observed_dates: np.ndarray,
) -> None:
    if not np.array_equal(expected_labels, observed_labels):
        raise AssertionError("Pass-two eligible label/order differs from pass one")
    if not pd.Series(expected_recommendations).equals(pd.Series(observed_recommendations)):
        raise AssertionError("Pass-two eligible identifier/order differs from pass one")
    expected_date_index = pd.DatetimeIndex(pd.to_datetime(expected_dates, utc=True))
    observed_date_index = pd.DatetimeIndex(pd.to_datetime(observed_dates, utc=True))
    if not expected_date_index.equals(observed_date_index):
        raise AssertionError("Pass-two eligible date/order differs from pass one")


def _promote_build(build_dir: Path, staging_dir: Path) -> None:
    old_dir = staging_dir.with_name(f"{staging_dir.name}.old-{uuid.uuid4().hex}")
    if staging_dir.exists():
        os.replace(staging_dir, old_dir)
    try:
        os.replace(build_dir, staging_dir)
    except Exception:
        if old_dir.exists() and not staging_dir.exists():
            os.replace(old_dir, staging_dir)
        raise
    if old_dir.exists():
        shutil.rmtree(old_dir)


def build_or_reuse_staging(
    config: RunConfig,
    source_path: Path,
    model_features: Mapping[str, Sequence[str]],
    required_metadata: Sequence[str],
    expected_source_sha256: str | None,
) -> StagingManifest:
    source = Path(source_path).resolve()
    source_sha256 = sha256_file(source)
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise ValueError(
            "Source SHA-256 differs from the audited dataset: "
            f"expected {expected_source_sha256}, got {source_sha256}"
        )
    expected_signature = _expected_staging_signature(
        config, source, source_sha256, model_features, required_metadata
    )
    if (
        not config.force_rebuild_staging
        and validate_staging_manifest(config.staging_dir, expected_signature)
    ):
        return StagingManifest.from_path(config.staging_dir / "staging_manifest.json")

    arrow_dataset = ds.dataset(source, format="parquet")
    source_columns = set(arrow_dataset.schema.names)
    minimal_columns = ["recommendationid", DATE_COLUMN, TARGET, "edited"]
    missing_minimal = sorted(set(minimal_columns) - source_columns)
    if missing_minimal:
        raise ValueError(f"Source is missing required cohort columns: {missing_minimal}")

    labels_parts: list[np.ndarray] = []
    recommendation_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []
    scanner = arrow_dataset.scanner(
        columns=minimal_columns,
        filter=_date_filter(arrow_dataset),
        batch_size=131_072,
        use_threads=True,
    )
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        dates = pd.to_datetime(frame[DATE_COLUMN], utc=True, errors="raise")
        target = pd.to_numeric(frame[TARGET], errors="coerce")
        edited = pd.to_numeric(frame["edited"], errors="coerce")
        eligible = target.isin([0, 1]) & edited.eq(0)
        if eligible.any():
            labels_parts.append(target.loc[eligible].astype("int8").to_numpy())
            recommendation_parts.append(frame.loc[eligible, "recommendationid"].to_numpy())
            date_parts.append(dates.loc[eligible].to_numpy())
    if not labels_parts:
        raise ValueError("No eligible binary, unedited, pre-2022 rows were found")
    all_labels = np.concatenate(labels_parts)
    all_recommendations = np.concatenate(recommendation_parts)
    all_dates = np.concatenate(date_parts)
    selected_positions = (
        _choose_smoke_positions(all_labels, config.smoke_rows, config.random_seed)
        if config.run_mode == "smoke"
        else np.arange(len(all_labels), dtype=np.int64)
    )
    selected_labels = all_labels[selected_positions]
    fold_ids = generate_fold_ids(
        selected_labels, config.n_outer_folds, config.random_seed
    )
    repeated = generate_fold_ids(
        selected_labels, config.n_outer_folds, config.random_seed
    )
    if not np.array_equal(fold_ids, repeated):
        raise AssertionError("Repeated fold generation was not deterministic")

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if config.run_mode == "full" else "_smoke"
    fold_path = config.artifact_dir / f"fold_assignments{suffix}.parquet"
    fold_frame = pd.DataFrame(
        {
            "stable_row_id": selected_positions.astype("int64"),
            "recommendationid": all_recommendations[selected_positions],
            "fold_id": fold_ids,
            "actual_label": selected_labels,
            DATE_COLUMN: pd.to_datetime(all_dates[selected_positions], utc=True),
        }
    ).sort_values("stable_row_id")
    fold_frame.to_parquet(fold_path, index=False)
    fold_assignment_sha256 = sha256_file(fold_path)
    (config.artifact_dir / f"fold_assignments{suffix}.sha256").write_text(
        f"{fold_assignment_sha256}  {fold_path.name}\n", encoding="utf-8"
    )

    feature_union = sorted({feature for values in model_features.values() for feature in values})
    raw_features = [
        feature for feature in feature_union if feature not in ENGINEERED_FEATURE_SOURCES
    ]
    feature_sources = [ENGINEERED_FEATURE_SOURCES[name] for name in feature_union if name in ENGINEERED_FEATURE_SOURCES]
    pass_two_columns = list(
        dict.fromkeys(list(required_metadata) + raw_features + feature_sources)
    )
    missing_pass_two = sorted(set(pass_two_columns) - source_columns)
    if missing_pass_two:
        raise ValueError(f"Source is missing staged feature columns: {missing_pass_two}")

    staging_dir = config.staging_dir
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = staging_dir.with_name(f"{staging_dir.name}.building-{uuid.uuid4().hex}")
    cohort_dir = build_dir / "cohort"
    cohort_dir.mkdir(parents=True)
    partition_rows = {str(fold): 0 for fold in range(config.n_outer_folds)}
    selected_positions_sorted = selected_positions
    primary_offset = 0
    file_counter = 0
    staged_dtypes: dict[str, str] = {}
    staged_columns = OOF_METADATA_COLUMNS[:-1] + feature_union

    scanner = arrow_dataset.scanner(
        columns=pass_two_columns,
        filter=_date_filter(arrow_dataset),
        batch_size=131_072,
        use_threads=True,
    )
    try:
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], utc=True, errors="raise")
            target = pd.to_numeric(frame[TARGET], errors="coerce")
            edited = pd.to_numeric(frame["edited"], errors="coerce")
            eligible = target.isin([0, 1]) & edited.eq(0)
            primary = frame.loc[eligible].copy()
            batch_start = primary_offset
            batch_ids = np.arange(
                primary_offset, primary_offset + len(primary), dtype=np.int64
            )
            primary_offset += len(primary)
            if len(primary) == 0:
                continue
            _assert_pass_two_sequence(
                all_labels[batch_start:primary_offset],
                all_recommendations[batch_start:primary_offset],
                all_dates[batch_start:primary_offset],
                target.loc[eligible].astype("int8").to_numpy(),
                frame.loc[eligible, "recommendationid"].to_numpy(),
                frame.loc[eligible, DATE_COLUMN].to_numpy(),
            )
            positions = np.searchsorted(selected_positions_sorted, batch_ids)
            in_sample = positions < len(selected_positions_sorted)
            in_sample[in_sample] &= (
                selected_positions_sorted[positions[in_sample]] == batch_ids[in_sample]
            )
            if not in_sample.any():
                continue
            primary = primary.iloc[np.flatnonzero(in_sample)].copy()
            selected_batch_ids = batch_ids[in_sample]
            selected_indices = np.searchsorted(selected_positions_sorted, selected_batch_ids)
            primary.insert(0, "stable_row_id", selected_batch_ids)
            primary["fold_id"] = fold_ids[selected_indices]
            primary = _engineer_staged_features(primary, feature_union)
            for fold_id, part in primary.groupby("fold_id", sort=True):
                fold = int(fold_id)
                output = part[OOF_METADATA_COLUMNS[:-1] + feature_union].copy()
                if not staged_dtypes:
                    staged_dtypes = {name: str(dtype) for name, dtype in output.dtypes.items()}
                partition_dir = cohort_dir / f"fold_id={fold}"
                partition_dir.mkdir(parents=True, exist_ok=True)
                output_path = partition_dir / f"part-{file_counter:06d}.parquet"
                output.to_parquet(output_path, index=False, compression="zstd")
                partition_rows[str(fold)] += len(output)
                file_counter += 1
        if primary_offset != len(all_labels):
            raise AssertionError("Pass-two eligible-row order/count differs from pass one")
        if sum(partition_rows.values()) != len(selected_positions):
            raise AssertionError("Staged row count differs from selected cohort")
        max_date = pd.to_datetime(all_dates[selected_positions], utc=True).max()
        manifest = StagingManifest(
            format_version=STAGING_FORMAT_VERSION,
            signature=expected_signature,
            source_path=str(source),
            source_sha256=source_sha256,
            run_mode=config.run_mode,
            random_seed=config.random_seed,
            n_folds=config.n_outer_folds,
            row_count=len(selected_positions),
            positive_rows=int(selected_labels.sum()),
            fold_assignment_sha256=fold_assignment_sha256,
            staged_columns=staged_columns,
            dtypes=staged_dtypes,
            partition_rows=partition_rows,
            max_modeling_date=pd.Timestamp(max_date).isoformat(),
        )
        (build_dir / "staging_manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _promote_build(build_dir, staging_dir)
    except Exception:
        if build_dir.exists():
            shutil.rmtree(build_dir)
        raise
    return manifest


def load_model_fold_data(
    staging_dir: Path,
    feature_names: Sequence[str],
    held_out_fold: int,
    n_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if held_out_fold not in range(n_folds):
        raise ValueError("held_out_fold is outside the configured fold range")
    dataset = ds.dataset(
        Path(staging_dir) / "cohort",
        format="parquet",
        partitioning="hive",
    )
    columns = list(dict.fromkeys(OOF_METADATA_COLUMNS + list(feature_names)))
    validation = dataset.to_table(
        columns=columns,
        filter=ds.field("fold_id") == held_out_fold,
    ).to_pandas()
    training = dataset.to_table(
        columns=columns,
        filter=ds.field("fold_id") != held_out_fold,
    ).to_pandas()
    for frame, role in ((training, "training"), (validation, "validation")):
        frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], utc=True, errors="raise")
        if "language" in frame:
            frame["language"] = frame["language"].astype(object)
        if frame.empty:
            raise AssertionError(f"{role} fold frame is empty")
        if not frame["stable_row_id"].is_unique:
            raise AssertionError(f"{role} stable_row_id values are not unique")
        if not frame[TARGET].isin([0, 1]).all():
            raise AssertionError(f"{role} target contains non-binary values")
        if frame[DATE_COLUMN].max() >= TRAIN_END:
            raise AssertionError(f"{role} contains post-2021 rows")
    if set(training["fold_id"].unique()) != set(range(n_folds)) - {held_out_fold}:
        raise AssertionError("training rows do not cover every non-held-out fold")
    if set(validation["fold_id"].unique()) != {held_out_fold}:
        raise AssertionError("validation rows do not exclusively use the held-out fold")
    training = training.sort_values("stable_row_id").reset_index(drop=True)
    validation = validation.sort_values("stable_row_id").reset_index(drop=True)
    return training, validation


def staging_disk_bytes(staging_dir: Path) -> int:
    return sum(path.stat().st_size for path in Path(staging_dir).rglob("*") if path.is_file())


def _stable_ids_sha256(stable_ids: np.ndarray | Sequence[int]) -> str:
    values = np.asarray(stable_ids, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def expected_checkpoint_signature(
    model_name: str,
    fold_id: int,
    source_sha256: str,
    fold_assignment_sha256: str,
    staging_signature: str,
    feature_names: Sequence[str],
    model_parameters: Mapping[str, Any],
    train_rows: int,
    validation_stable_ids: np.ndarray | Sequence[int],
) -> dict[str, Any]:
    validation_ids = np.asarray(validation_stable_ids, dtype=np.int64)
    return {
        "checkpoint_format_version": 1,
        "model": model_name,
        "fold_id": int(fold_id),
        "source_sha256": source_sha256,
        "fold_assignment_sha256": fold_assignment_sha256,
        "staging_signature": staging_signature,
        "feature_signature": feature_signature(feature_names),
        "model_signature": model_signature(model_name, model_parameters),
        "train_rows": int(train_rows),
        "validation_rows": int(len(validation_ids)),
        "validation_stable_ids_sha256": _stable_ids_sha256(validation_ids),
    }


def validate_checkpoint(
    checkpoint_path: Path,
    metadata_path: Path,
    expected: Mapping[str, Any],
) -> bool:
    checkpoint = Path(checkpoint_path)
    metadata_file = Path(metadata_path)
    if not checkpoint.is_file() or not metadata_file.is_file():
        return False
    try:
        actual = json.loads(metadata_file.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if actual.get(key) != value:
                return False
        probability_column = f"prob_{expected['model']}"
        frame = pd.read_parquet(
            checkpoint,
            columns=["stable_row_id", "actual_label", probability_column],
        )
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if len(frame) != int(expected["validation_rows"]):
        return False
    if not frame["stable_row_id"].is_unique:
        return False
    if _stable_ids_sha256(frame["stable_row_id"].to_numpy()) != expected[
        "validation_stable_ids_sha256"
    ]:
        return False
    if not frame["actual_label"].isin([0, 1]).all():
        return False
    probability = pd.to_numeric(frame[probability_column], errors="coerce").to_numpy()
    return bool(
        np.isfinite(probability).all()
        and ((probability >= 0.0) & (probability <= 1.0)).all()
    )


def estimate_full_resources(
    staging_bytes: int,
    row_count: int,
    largest_model_features: int,
    sample_bytes_per_row: float,
    available_ram_gb: float,
    free_disk_gb: float,
) -> dict[str, float | int]:
    gib = 1024**3
    projected_frame_gb = sample_bytes_per_row * row_count / gib
    feature_headroom = max(1.0, largest_model_features / 100.0)
    estimated_peak_gb = projected_frame_gb * (2.5 + 0.5 * feature_headroom) + 2.0
    staging_gb = staging_bytes / gib
    estimated_output_gb = max(0.25, row_count * 64 / gib * 1.5)
    return {
        "staging_bytes": int(staging_bytes),
        "staging_gb": float(staging_gb),
        "row_count": int(row_count),
        "largest_model_features": int(largest_model_features),
        "sample_bytes_per_row": float(sample_bytes_per_row),
        "estimated_peak_gb": float(estimated_peak_gb),
        "estimated_output_gb": float(estimated_output_gb),
        "available_ram_gb": float(available_ram_gb),
        "free_disk_gb": float(free_disk_gb),
    }


def assert_resources_safe(
    report: Mapping[str, float | int],
    min_available_ram_gb: float,
    min_free_disk_gb: float,
) -> None:
    available_ram = float(report["available_ram_gb"])
    estimated_peak = float(report["estimated_peak_gb"])
    required_ram = max(float(min_available_ram_gb), estimated_peak * 1.15)
    if available_ram < required_ram:
        raise RuntimeError(
            "Insufficient available RAM before full training: "
            f"need at least {required_ram:.2f} GiB, available {available_ram:.2f} GiB. "
            "Close memory-heavy applications or use a larger-memory machine."
        )
    free_disk = float(report["free_disk_gb"])
    required_disk = max(
        float(min_free_disk_gb),
        float(report["staging_gb"]) + 2.0 * float(report["estimated_output_gb"]),
    )
    if free_disk < required_disk:
        raise RuntimeError(
            "Insufficient free disk before full training: "
            f"need at least {required_disk:.2f} GiB, available {free_disk:.2f} GiB."
        )


def run_after_preflight(
    config: RunConfig,
    report: Mapping[str, float | int],
    fit_callable: Callable[[], Any],
) -> Any:
    assert_resources_safe(
        report,
        min_available_ram_gb=config.min_available_ram_gb,
        min_free_disk_gb=config.min_free_disk_gb,
    )
    if config.full_preflight_only:
        return None
    return fit_callable()
