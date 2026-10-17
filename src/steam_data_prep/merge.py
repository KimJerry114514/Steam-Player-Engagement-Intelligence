from __future__ import annotations

import csv
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from .reviews import _sql_literal


@dataclass(frozen=True)
class MergeMetrics:
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
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def merge_reviews(
    review_dir: Path,
    mapping_path: Path,
    output_dir: Path,
    *,
    unmatched_csv: Path,
    unmatched_parquet: Path,
) -> MergeMetrics:
    """Perform a validated many-to-one left join and publish partitioned Parquet."""
    start = time.monotonic()
    for target in (output_dir, unmatched_csv, unmatched_parquet):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite generated output: {target}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    unmatched_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    temp_unmatched = unmatched_parquet.with_name(f".{unmatched_parquet.name}.tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    reviews_pattern = str(review_dir / "**" / "*.parquet")
    reviews_scan = f"read_parquet({_sql_literal(reviews_pattern)}, hive_partitioning = true)"
    mapping_scan = f"read_parquet({_sql_literal(str(mapping_path))})"
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone = 'UTC'")
        mapping_counts = connection.execute(
            f"SELECT count(*), count(DISTINCT appid) FROM {mapping_scan}"
        ).fetchone()
        if mapping_counts[0] != mapping_counts[1]:
            raise RuntimeError(
                "Metadata mapping appid must be unique before merge: "
                f"rows={mapping_counts[0]}, unique={mapping_counts[1]}"
            )
        aggregate = connection.execute(
            f"""
            SELECT
                count(*) AS review_rows,
                count(DISTINCT r.appid) AS unique_review_appids,
                count(m.appid) AS matched_review_rows,
                count(*) - count(m.appid) AS unmatched_review_rows,
                count(DISTINCT CASE WHEN m.appid IS NOT NULL THEN r.appid END)
                    AS matched_unique_appids,
                count(DISTINCT CASE WHEN r.appid IS NOT NULL AND m.appid IS NULL
                    THEN r.appid END) AS unmatched_unique_appids
            FROM {reviews_scan} r
            LEFT JOIN {mapping_scan} m USING (appid)
            """
        ).fetchone()
        unmatched_query = f"""
            SELECT DISTINCT r.appid
            FROM {reviews_scan} r
            LEFT JOIN {mapping_scan} m USING (appid)
            WHERE r.appid IS NOT NULL AND m.appid IS NULL
            ORDER BY r.appid
        """
        connection.execute(
            f"COPY ({unmatched_query}) TO {_sql_literal(str(temp_unmatched))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        join_query = f"""
            SELECT r.*, m.game_name, m.genres, m.release_date, m.publishers
            FROM {reviews_scan} r
            LEFT JOIN {mapping_scan} m USING (appid)
        """
        connection.execute(
            f"""
            COPY ({join_query}) TO {_sql_literal(str(temp_dir))}
            (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (review_year), ROW_GROUP_SIZE 100000)
            """
        )
        enriched_rows = sum(
            pq.ParquetFile(path).metadata.num_rows for path in temp_dir.rglob("*.parquet")
        )
        if enriched_rows != aggregate[0]:
            raise RuntimeError(
                f"Left join changed row count: reviews={aggregate[0]}, enriched={enriched_rows}"
            )
        temp_dir.replace(output_dir)
        temp_unmatched.replace(unmatched_parquet)
        _write_unmatched_csv(unmatched_parquet, unmatched_csv)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_unmatched.unlink(missing_ok=True)
        raise
    finally:
        connection.close()

    unique_reviews = aggregate[1]
    return MergeMetrics(
        review_rows=aggregate[0],
        enriched_rows=enriched_rows,
        unique_review_appids=unique_reviews,
        unique_metadata_appids=mapping_counts[1],
        matched_review_rows=aggregate[2],
        unmatched_review_rows=aggregate[3],
        matched_unique_appids=aggregate[4],
        unmatched_unique_appids=aggregate[5],
        match_rate_by_row=_rate(aggregate[2], aggregate[0]),
        match_rate_by_unique_appid=_rate(aggregate[4], unique_reviews),
        elapsed_seconds=round(time.monotonic() - start, 3),
    )


def _write_unmatched_csv(parquet_path: Path, csv_path: Path) -> None:
    temp = csv_path.with_name(f".{csv_path.name}.tmp")
    appids = pq.read_table(parquet_path)["appid"].to_pylist()
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["appid"])
        writer.writerows([appid] for appid in appids)
    temp.replace(csv_path)


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
