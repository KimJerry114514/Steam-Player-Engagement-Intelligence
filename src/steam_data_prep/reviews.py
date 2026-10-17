from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from .audit import AuditResult, sha256_file

REVIEW_SOURCE_COLUMNS = [
    "steamid",
    "appid",
    "voted_up",
    "votes_up",
    "votes_funny",
    "weighted_vote_score",
    "playtime_forever",
    "playtime_at_review",
    "num_games_owned",
    "num_reviews",
    "review",
    "unix_timestamp_created",
    "unix_timestamp_updated",
]


@dataclass(frozen=True)
class ReviewMetrics:
    input_rows: int
    output_rows: int
    invalid_appids: int
    null_appids: int
    invalid_created_timestamps: int
    invalid_updated_timestamps: int
    min_review_year: int | None
    max_review_year: int | None
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def combine_reviews(
    audit: AuditResult,
    output_dir: Path,
    *,
    rows_per_file: int | None = None,
) -> ReviewMetrics:
    """Combine manifest-approved review CSVs into partitioned Parquet."""
    start = time.monotonic()
    _verify_manifest(audit)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite generated output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    raw_query = _raw_union_query(audit.review_files, rows_per_file=rows_per_file)
    typed_query = _typed_query(raw_query)
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone = 'UTC'")
        raw_metrics = connection.execute(
            f"""
            SELECT
                count(*) AS input_rows,
                count_if(appid IS NOT NULL AND try_cast(appid AS BIGINT) IS NULL),
                count_if(try_cast(appid AS BIGINT) IS NULL),
                count_if(unix_timestamp_created IS NOT NULL
                    AND try_cast(unix_timestamp_created AS BIGINT) IS NULL),
                count_if(unix_timestamp_updated IS NOT NULL
                    AND try_cast(unix_timestamp_updated AS BIGINT) IS NULL)
            FROM ({raw_query})
            """
        ).fetchone()
        connection.execute(
            f"""
            COPY ({typed_query}) TO {_sql_literal(str(temp_dir))}
            (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (review_year), ROW_GROUP_SIZE 100000)
            """
        )
        output_rows = sum(
            pq.ParquetFile(path).metadata.num_rows for path in temp_dir.rglob("*.parquet")
        )
        year_bounds = connection.execute(
            f"SELECT min(review_year), max(review_year) FROM ({typed_query})"
        ).fetchone()
        if raw_metrics[0] != output_rows:
            raise RuntimeError(
                f"Review fact row mismatch: input={raw_metrics[0]}, output={output_rows}"
            )
        temp_dir.replace(output_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    finally:
        connection.close()

    return ReviewMetrics(
        input_rows=raw_metrics[0],
        output_rows=output_rows,
        invalid_appids=raw_metrics[1],
        null_appids=raw_metrics[2],
        invalid_created_timestamps=raw_metrics[3],
        invalid_updated_timestamps=raw_metrics[4],
        min_review_year=year_bounds[0],
        max_review_year=year_bounds[1],
        elapsed_seconds=round(time.monotonic() - start, 3),
    )


def _verify_manifest(audit: AuditResult) -> None:
    review_entries = {
        entry["path"]: entry for entry in audit.manifest if entry["kind"] == "review_csv"
    }
    for path in audit.review_files:
        matches = [entry for key, entry in review_entries.items() if key.endswith(path.name)]
        if len(matches) != 1 or sha256_file(path) != matches[0]["sha256"]:
            raise RuntimeError(f"Source file changed after audit: {path}")


def _raw_union_query(review_files: list[Path], *, rows_per_file: int | None) -> str:
    selects = []
    columns = (
        "{"
        + ", ".join(f"{_sql_literal(column)}: 'VARCHAR'" for column in REVIEW_SOURCE_COLUMNS)
        + "}"
    )
    for path in review_files:
        limit = f" LIMIT {int(rows_per_file)}" if rows_per_file is not None else ""
        selects.append(
            "(SELECT *, "
            f"{_sql_literal(path.name)} AS source_file FROM read_csv("
            f"{_sql_literal(str(path))}, header = true, auto_detect = false, "
            f"delim = ',', quote = '\"', escape = '\"', columns = {columns}, "
            "strict_mode = false, ignore_errors = false, max_line_size = 100000000)"
            f"{limit})"
        )
    return " UNION ALL ".join(selects)


def _typed_query(raw_query: str) -> str:
    return f"""
        SELECT
            try_cast(steamid AS BIGINT) AS steamid,
            try_cast(appid AS BIGINT) AS appid,
            try_cast(voted_up AS BOOLEAN) AS voted_up,
            try_cast(votes_up AS BIGINT) AS votes_up,
            try_cast(votes_funny AS BIGINT) AS votes_funny,
            try_cast(weighted_vote_score AS DOUBLE) AS weighted_vote_score,
            try_cast(playtime_forever AS BIGINT) AS playtime_forever,
            try_cast(playtime_at_review AS BIGINT) AS playtime_at_review,
            try_cast(num_games_owned AS BIGINT) AS num_games_owned,
            try_cast(num_reviews AS BIGINT) AS num_reviews,
            review,
            try_cast(unix_timestamp_created AS BIGINT) AS unix_timestamp_created,
            try_cast(unix_timestamp_updated AS BIGINT) AS unix_timestamp_updated,
            source_file,
            to_timestamp(try_cast(unix_timestamp_created AS DOUBLE)) AS created_at_utc,
            to_timestamp(try_cast(unix_timestamp_updated AS DOUBLE)) AS updated_at_utc,
            try_cast(year(to_timestamp(try_cast(unix_timestamp_created AS DOUBLE))) AS INTEGER)
                AS review_year
        FROM ({raw_query})
    """


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
