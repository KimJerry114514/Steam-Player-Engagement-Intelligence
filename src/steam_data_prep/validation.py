from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from .audit import _atomic_json
from .config import PipelinePaths
from .merge import MergeMetrics
from .reviews import REVIEW_SOURCE_COLUMNS, _sql_literal


@dataclass(frozen=True)
class DuplicateMetrics:
    total_rows: int
    exact_duplicate_groups: int
    exact_duplicate_instances: int
    exact_redundant_rows: int
    fallback_duplicate_groups: int
    fallback_duplicate_instances: int
    fallback_redundant_rows: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationMetrics:
    review_rows: int
    enriched_rows: int
    unique_review_appids: int
    unique_metadata_appids: int
    matched_review_rows: int
    unmatched_review_rows: int
    matched_unique_appids: int
    unmatched_unique_appids: int
    match_rate_by_row: float
    match_rate_by_unique_appid: float
    unmatched_appids: list[int]
    source_columns_preserved: bool
    review_schema: dict[str, str]
    enriched_schema: dict[str, str]
    timestamp_min_utc: str | None
    timestamp_max_utc: str | None
    missing_value_rates: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_outputs(
    paths: PipelinePaths, *, merge_metrics: MergeMetrics | None = None
) -> ValidationMetrics:
    review_pattern = str(paths.reviews_fact_dir / "**" / "*.parquet")
    enriched_pattern = str(paths.reviews_enriched_dir / "**" / "*.parquet")
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone = 'UTC'")
        review_description = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet({_sql_literal(review_pattern)}, "
            "hive_partitioning = true)"
        ).fetchall()
        enriched_description = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet({_sql_literal(enriched_pattern)}, "
            "hive_partitioning = true)"
        ).fetchall()
        if merge_metrics is None:
            merge_metrics = _derive_merge_metrics(connection, paths)
        timestamp_bounds = connection.execute(
            f"SELECT min(created_at_utc), max(created_at_utc) FROM "
            f"read_parquet({_sql_literal(enriched_pattern)}, hive_partitioning = true)"
        ).fetchone()
        enriched_names = [row[0] for row in enriched_description]
        missing_expressions = ", ".join(
            f"count_if({_quote_identifier(name)} IS NULL) / count(*)::DOUBLE"
            for name in enriched_names
        )
        missing_row = connection.execute(
            f"SELECT {missing_expressions} FROM "
            f"read_parquet({_sql_literal(enriched_pattern)}, hive_partitioning = true)"
        ).fetchone()
    finally:
        connection.close()
    if merge_metrics.review_rows != merge_metrics.enriched_rows:
        raise RuntimeError(
            "Validation failed: enriched row count differs from review fact row count"
        )
    unmatched = [
        int(value) for value in pq.read_table(paths.unmatched_parquet)["appid"].to_pylist()
    ]
    review_schema = {row[0]: row[1] for row in review_description}
    enriched_schema = {row[0]: row[1] for row in enriched_description}
    metrics = ValidationMetrics(
        review_rows=merge_metrics.review_rows,
        enriched_rows=merge_metrics.enriched_rows,
        unique_review_appids=merge_metrics.unique_review_appids,
        unique_metadata_appids=merge_metrics.unique_metadata_appids,
        matched_review_rows=merge_metrics.matched_review_rows,
        unmatched_review_rows=merge_metrics.unmatched_review_rows,
        matched_unique_appids=merge_metrics.matched_unique_appids,
        unmatched_unique_appids=merge_metrics.unmatched_unique_appids,
        match_rate_by_row=merge_metrics.match_rate_by_row,
        match_rate_by_unique_appid=merge_metrics.match_rate_by_unique_appid,
        unmatched_appids=unmatched,
        source_columns_preserved=all(column in enriched_schema for column in REVIEW_SOURCE_COLUMNS),
        review_schema=review_schema,
        enriched_schema=enriched_schema,
        timestamp_min_utc=str(timestamp_bounds[0]) if timestamp_bounds[0] is not None else None,
        timestamp_max_utc=str(timestamp_bounds[1]) if timestamp_bounds[1] is not None else None,
        missing_value_rates=dict(zip(enriched_names, missing_row, strict=True)),
    )
    paths.audit_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(paths.audit_dir / "validation_metrics.json", metrics.to_dict())
    return metrics


def _derive_merge_metrics(
    connection: duckdb.DuckDBPyConnection, paths: PipelinePaths
) -> MergeMetrics:
    review_scan = (
        f"read_parquet({_sql_literal(str(paths.reviews_fact_dir / '**' / '*.parquet'))}, "
        "hive_partitioning = true)"
    )
    enriched_scan = (
        f"read_parquet({_sql_literal(str(paths.reviews_enriched_dir / '**' / '*.parquet'))}, "
        "hive_partitioning = true)"
    )
    mapping_scan = f"read_parquet({_sql_literal(str(paths.mapping_parquet))})"
    row = connection.execute(
        f"""
        SELECT
            count(*),
            (SELECT count(*) FROM {enriched_scan}),
            count(DISTINCT r.appid),
            (SELECT count(DISTINCT appid) FROM {mapping_scan}),
            count(m.appid),
            count(*) - count(m.appid),
            count(DISTINCT CASE WHEN m.appid IS NOT NULL THEN r.appid END),
            count(DISTINCT CASE WHEN r.appid IS NOT NULL AND m.appid IS NULL
                THEN r.appid END)
        FROM {review_scan} r
        LEFT JOIN {mapping_scan} m USING (appid)
        """
    ).fetchone()
    return MergeMetrics(
        review_rows=row[0],
        enriched_rows=row[1],
        unique_review_appids=row[2],
        unique_metadata_appids=row[3],
        matched_review_rows=row[4],
        unmatched_review_rows=row[5],
        matched_unique_appids=row[6],
        unmatched_unique_appids=row[7],
        match_rate_by_row=0.0 if row[0] == 0 else row[4] / row[0],
        match_rate_by_unique_appid=0.0 if row[2] == 0 else row[6] / row[2],
        elapsed_seconds=0.0,
    )


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def analyze_review_duplicates(dataset_dir: Path, output_dir: Path) -> DuplicateMetrics:
    """Persist exact and fallback-key duplicate groups without deleting rows."""
    start = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    exact_path = output_dir / "exact_duplicate_groups.parquet"
    fallback_path = output_dir / "fallback_duplicate_groups.parquet"
    for path in (exact_path, fallback_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite generated output: {path}")
    pattern = str(dataset_dir / "**" / "*.parquet")
    columns = ", ".join(REVIEW_SOURCE_COLUMNS)
    connection = duckdb.connect()
    try:
        total_rows = connection.execute(
            f"SELECT count(*) FROM read_parquet({_sql_literal(pattern)}, hive_partitioning = true)"
        ).fetchone()[0]
        connection.execute(
            f"""
            COPY (
                SELECT {columns}, count(*) AS duplicate_count
                FROM read_parquet({_sql_literal(pattern)}, hive_partitioning = true)
                GROUP BY {columns}
                HAVING count(*) > 1
            ) TO {_sql_literal(str(exact_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        connection.execute(
            f"""
            COPY (
                SELECT appid, steamid, unix_timestamp_created, count(*) AS duplicate_count
                FROM read_parquet({_sql_literal(pattern)}, hive_partitioning = true)
                GROUP BY appid, steamid, unix_timestamp_created
                HAVING count(*) > 1
            ) TO {_sql_literal(str(fallback_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        exact = _duplicate_totals(connection, exact_path)
        fallback = _duplicate_totals(connection, fallback_path)
    finally:
        connection.close()
    return DuplicateMetrics(
        total_rows=total_rows,
        exact_duplicate_groups=exact[0],
        exact_duplicate_instances=exact[1],
        exact_redundant_rows=exact[2],
        fallback_duplicate_groups=fallback[0],
        fallback_duplicate_instances=fallback[1],
        fallback_redundant_rows=fallback[2],
        elapsed_seconds=round(time.monotonic() - start, 3),
    )


def _duplicate_totals(connection: duckdb.DuckDBPyConnection, path: Path) -> tuple[int, int, int]:
    row = connection.execute(
        f"""
        SELECT count(*), coalesce(sum(duplicate_count), 0),
            coalesce(sum(duplicate_count - 1), 0)
        FROM read_parquet({_sql_literal(str(path))})
        """
    ).fetchone()
    return row[0], row[1], row[2]
