from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import ijson
import pyarrow as pa
import pyarrow.parquet as pq

from .analysis_notebook import write_analysis_readiness_notebook
from .analysis_reporting import render_analysis_reports, render_figures
from .audit import _atomic_json
from .reviews import REVIEW_SOURCE_COLUMNS, _sql_literal

KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/forgemaster/steam-reviews-dataset"
DATASET_AUTHOR = "forgemaster"


@dataclass(frozen=True)
class AnalysisPaths:
    project_root: Path
    canonical_root: Path
    enriched_dir: Path
    canonical_mapping: Path
    canonical_unmatched: Path
    steamdb_json: Path
    source_dir: Path
    output_root: Path
    analysis_base_dir: Path
    negative_reviews_dir: Path
    unmatched_diagnostics: Path
    duplicate_diagnostics: Path
    audit_dir: Path
    tables_dir: Path
    figures_dir: Path
    metrics_path: Path
    notebook_path: Path
    audit_report: Path
    target_report: Path

    @classmethod
    def from_project_root(
        cls,
        root: str | Path,
        *,
        canonical_root: str | Path | None = None,
        output_root: str | Path | None = None,
    ) -> AnalysisPaths:
        project_root = Path(root).expanduser().resolve()
        canonical = (
            Path(canonical_root).expanduser().resolve()
            if canonical_root is not None
            else project_root / "runs" / "data_preparation" / "kaggle"
        )
        output = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else project_root / "runs" / "data_preparation" / "analysis"
        )
        source_dir = project_root / "data" / "raw" / "reviews"
        steamdb = project_root / "data" / "external" / "steamdb.json"
        if (
            output == canonical
            or canonical.is_relative_to(output)
            or output.is_relative_to(canonical)
        ):
            raise ValueError("Analysis output root must not overlap the canonical dataset")
        if (
            output == source_dir
            or source_dir.is_relative_to(output)
            or output.is_relative_to(source_dir)
        ):
            raise ValueError("Analysis output root must not overlap the raw source directory")
        return cls(
            project_root=project_root,
            canonical_root=canonical,
            enriched_dir=canonical / "reviews_enriched",
            canonical_mapping=canonical / "steam_games_mapping.parquet",
            canonical_unmatched=canonical / "unmatched_appids.parquet",
            steamdb_json=steamdb,
            source_dir=source_dir,
            output_root=output,
            analysis_base_dir=output / "reviews_analysis_base",
            negative_reviews_dir=output / "negative_reviews",
            unmatched_diagnostics=output / "unmatched_appid_diagnostics.parquet",
            duplicate_diagnostics=output / "duplicate_diagnostics.parquet",
            audit_dir=output / "audit",
            tables_dir=output / "audit" / "tables",
            figures_dir=output / "figures",
            metrics_path=output / "audit" / "analysis_readiness_metrics.json",
            notebook_path=output / "notebooks" / "02_analysis_readiness.ipynb",
            audit_report=output / "reports" / "analysis_readiness_audit.md",
            target_report=output / "reports" / "target_framework_recommendation.md",
        )


@dataclass(frozen=True)
class CatalogEvidence:
    steam_appid_present: bool
    kinds: tuple[str, ...]
    internal_id_collision: bool


@dataclass(frozen=True)
class AnalysisBaseMetrics:
    input_rows: int
    output_rows: int
    negative_reviews: int
    unmatched_reviews: int
    exact_duplicate_rows: int
    candidate_duplicate_rows: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_duplicate_group(
    distinct_payloads: int, distinct_updates: int, distinct_texts: int
) -> str:
    if distinct_payloads == 1:
        return "exact_duplicate"
    if distinct_updates > 1:
        return "likely_updated_version"
    if distinct_texts > 1:
        return "text_variant_same_update_time"
    return "other_payload_variant_ambiguous"


def scan_raw_catalog_appids(json_path: Path, target_appids: set[int]) -> dict[int, CatalogEvidence]:
    """Find target AppIDs only in semantically relevant raw-catalog identifier fields."""
    kinds: dict[int, set[str]] = {appid: set() for appid in target_appids}
    internal_collisions: set[int] = set()
    with json_path.open("rb") as handle:
        for record in ijson.items(handle, "item"):
            appid = _optional_int(record.get("steam_appid"))
            if appid in target_appids:
                kinds[appid].add(str(record.get("kind")))
            internal_id = _optional_int(record.get("id"))
            if internal_id in target_appids and internal_id != appid:
                internal_collisions.add(internal_id)
    return {
        appid: CatalogEvidence(
            steam_appid_present=bool(kinds[appid]),
            kinds=tuple(sorted(kinds[appid])),
            internal_id_collision=appid in internal_collisions,
        )
        for appid in sorted(target_appids)
    }


def build_analysis_base(paths: AnalysisPaths) -> AnalysisBaseMetrics:
    start = time.monotonic()
    _require_canonical_inputs(paths)
    _ensure_new(paths.analysis_base_dir)
    pattern = str(paths.enriched_dir / "**" / "*.parquet")
    temp_dir = paths.analysis_base_dir.with_name(
        f".{paths.analysis_base_dir.name}.tmp-{os.getpid()}"
    )
    temp_dir.parent.mkdir(parents=True, exist_ok=True)
    exact_partition = ", ".join(_quote_identifier(name) for name in REVIEW_SOURCE_COLUMNS)
    exact_order = ", ".join(
        [
            "source_file",
            "coalesce(review, '')",
            "unix_timestamp_updated DESC NULLS LAST",
        ]
    )
    candidate_partition = "appid, steamid, unix_timestamp_created"
    query = f"""
        WITH windowed AS (
            SELECT
                r.*,
                count(*) OVER (PARTITION BY {exact_partition}) AS exact_duplicate_count,
                row_number() OVER (
                    PARTITION BY {exact_partition}
                    ORDER BY {exact_order}
                ) AS exact_duplicate_rank,
                count(*) OVER (
                    PARTITION BY {candidate_partition}
                ) AS candidate_key_duplicate_count,
                row_number() OVER (
                    PARTITION BY {candidate_partition}
                    ORDER BY unix_timestamp_updated DESC NULLS LAST,
                        coalesce(review, '') DESC,
                        source_file,
                        playtime_forever DESC NULLS LAST,
                        votes_up DESC NULLS LAST
                ) AS candidate_key_duplicate_rank,
                count(DISTINCT review) OVER (
                    PARTITION BY {candidate_partition}
                ) AS candidate_distinct_review_count,
                count(DISTINCT unix_timestamp_updated) OVER (
                    PARTITION BY {candidate_partition}
                ) AS candidate_distinct_update_count,
                max(created_at_utc) OVER () AS dataset_max_created_at_utc
            FROM read_parquet({_sql_literal(pattern)}, hive_partitioning = true) r
        )
        SELECT
            * EXCLUDE (dataset_max_created_at_utc),
            game_name IS NOT NULL AS metadata_matched,
            exact_duplicate_count > 1 AS exact_duplicate_flag,
            candidate_key_duplicate_count > 1 AS candidate_key_duplicate_flag,
            candidate_distinct_review_count > 1 AS candidate_review_text_varies,
            candidate_distinct_update_count > 1 AS candidate_update_timestamp_varies,
            CAST(date_trunc('month', created_at_utc) AS DATE) AS review_month,
            playtime_forever - playtime_at_review AS additional_playtime_minutes,
            playtime_forever IS NOT NULL AND playtime_at_review IS NOT NULL
                AS playtime_inputs_present,
            coalesce(playtime_forever >= 0 AND playtime_at_review >= 0, false)
                AS playtime_inputs_nonnegative,
            coalesce(playtime_forever - playtime_at_review >= 0, false)
                AS additional_playtime_nonnegative,
            playtime_forever - playtime_at_review < 0 AS additional_playtime_negative,
            playtime_forever - playtime_at_review = 0 AS additional_playtime_zero,
            playtime_forever - playtime_at_review > 0 AS additional_playtime_positive,
            CASE WHEN release_date IS NOT NULL AND created_at_utc IS NOT NULL
                THEN date_diff('day', release_date, CAST(created_at_utc AS DATE)) END
                AS game_age_at_review_days,
            coalesce(release_date > CAST(created_at_utc AS DATE), false)
                AS review_before_release,
            NOT voted_up AS is_negative_review,
            coalesce(updated_at_utc < created_at_utc, false) AS updated_before_created,
            coalesce(updated_at_utc > dataset_max_created_at_utc, false)
                AS updated_after_dataset_creation_max
        FROM windowed
    """
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone = 'UTC'")
        input_rows = connection.execute(
            f"SELECT count(*) FROM read_parquet({_sql_literal(pattern)}, hive_partitioning = true)"
        ).fetchone()[0]
        connection.execute(
            f"COPY ({query}) TO {_sql_literal(str(temp_dir))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (review_year), "
            "ROW_GROUP_SIZE 100000)"
        )
        output_rows = _parquet_row_count(temp_dir)
        if output_rows != input_rows:
            raise RuntimeError(
                f"Analysis-base row mismatch: input={input_rows}, output={output_rows}"
            )
        temp_pattern = str(temp_dir / "**" / "*.parquet")
        row = connection.execute(
            f"""
            SELECT
                count(*) FILTER (WHERE is_negative_review),
                count(*) FILTER (WHERE NOT metadata_matched),
                count(*) FILTER (WHERE exact_duplicate_flag),
                count(*) FILTER (WHERE candidate_key_duplicate_flag)
            FROM read_parquet({_sql_literal(temp_pattern)}, hive_partitioning = true)
            """
        ).fetchone()
        temp_dir.replace(paths.analysis_base_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    finally:
        connection.close()
    return AnalysisBaseMetrics(
        input_rows=input_rows,
        output_rows=output_rows,
        negative_reviews=row[0],
        unmatched_reviews=row[1],
        exact_duplicate_rows=row[2],
        candidate_duplicate_rows=row[3],
        elapsed_seconds=round(time.monotonic() - start, 3),
    )


def build_negative_reviews(paths: AnalysisPaths) -> int:
    _ensure_new(paths.negative_reviews_dir)
    pattern = str(paths.analysis_base_dir / "**" / "*.parquet")
    temp_dir = paths.negative_reviews_dir.with_name(
        f".{paths.negative_reviews_dir.name}.tmp-{os.getpid()}"
    )
    temp_dir.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet({_sql_literal(pattern)}, hive_partitioning = true)
                WHERE is_negative_review
            ) TO {_sql_literal(str(temp_dir))}
            (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (review_year),
                ROW_GROUP_SIZE 100000)
            """
        )
        count = _parquet_row_count(temp_dir)
        temp_dir.replace(paths.negative_reviews_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    finally:
        connection.close()
    return count


def build_unmatched_diagnostics(paths: AnalysisPaths) -> dict[str, int]:
    _ensure_new(paths.unmatched_diagnostics)
    unmatched_table = pq.read_table(paths.canonical_unmatched, columns=["appid"])
    appids = {int(value) for value in unmatched_table["appid"].to_pylist()}
    evidence = scan_raw_catalog_appids(paths.steamdb_json, appids)
    evidence_table = pa.Table.from_pylist(
        [
            {
                "appid": appid,
                "raw_steam_appid_present": item.steam_appid_present,
                "raw_steam_appid_kinds": list(item.kinds),
                "excluded_by_steam_kind_filter": item.steam_appid_present
                and "steam" not in item.kinds,
                "raw_internal_id_collision": item.internal_id_collision,
            }
            for appid, item in evidence.items()
        ]
    )
    pattern = str(paths.analysis_base_dir / "**" / "*.parquet")
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone = 'UTC'")
        connection.register("catalog_evidence", evidence_table)
        connection.execute(
            f"""
            COPY (
                SELECT
                    b.appid,
                    count(*) AS review_count,
                    count(DISTINCT b.steamid) AS user_count,
                    min(b.created_at_utc) AS earliest_review_utc,
                    max(b.created_at_utc) AS latest_review_utc,
                    count(*) FILTER (WHERE b.voted_up) AS positive_review_count,
                    count(*) FILTER (WHERE NOT b.voted_up) AS negative_review_count,
                    count(*) FILTER (WHERE b.voted_up) / count(*)::DOUBLE
                        AS positive_review_share,
                    min(b.playtime_forever) AS min_playtime_forever,
                    approx_quantile(b.playtime_forever, 0.25) AS p25_playtime_forever,
                    approx_quantile(b.playtime_forever, 0.50) AS median_playtime_forever,
                    approx_quantile(b.playtime_forever, 0.90) AS p90_playtime_forever,
                    approx_quantile(b.playtime_forever, 0.99) AS p99_playtime_forever,
                    max(b.playtime_forever) AS max_playtime_forever,
                    avg(b.num_games_owned) AS mean_games_owned,
                    avg(b.num_reviews) AS mean_reviews_written,
                    avg(b.votes_up) AS mean_votes_up,
                    avg(b.votes_funny) AS mean_votes_funny,
                    avg(b.weighted_vote_score) AS mean_weighted_vote_score,
                    list_sort(list(DISTINCT b.source_file)) AS source_csv_files,
                    e.raw_steam_appid_present,
                    e.raw_steam_appid_kinds,
                    e.excluded_by_steam_kind_filter,
                    e.raw_internal_id_collision
                FROM read_parquet({_sql_literal(pattern)}, hive_partitioning = true) b
                JOIN catalog_evidence e USING (appid)
                GROUP BY b.appid, e.raw_steam_appid_present, e.raw_steam_appid_kinds,
                    e.excluded_by_steam_kind_filter, e.raw_internal_id_collision
                ORDER BY review_count DESC, b.appid
            ) TO {_sql_literal(str(paths.unmatched_diagnostics))}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        row = connection.execute(
            f"""
            SELECT count(*),
                count(*) FILTER (WHERE raw_steam_appid_present),
                count(*) FILTER (WHERE excluded_by_steam_kind_filter),
                count(*) FILTER (WHERE raw_internal_id_collision)
            FROM read_parquet({_sql_literal(str(paths.unmatched_diagnostics))})
            """
        ).fetchone()
    finally:
        connection.close()
    return {
        "unmatched_appids": row[0],
        "present_as_steam_appid_any_kind": row[1],
        "excluded_by_kind_filter": row[2],
        "internal_id_collisions": row[3],
    }


def build_duplicate_diagnostics(paths: AnalysisPaths) -> dict[str, int]:
    _ensure_new(paths.duplicate_diagnostics)
    pattern = str(paths.analysis_base_dir / "**" / "*.parquet")
    payload_columns = ", ".join(_quote_identifier(name) for name in REVIEW_SOURCE_COLUMNS)
    query = f"""
        WITH candidate_rows AS (
            SELECT *
            FROM read_parquet({_sql_literal(pattern)}, hive_partitioning = true)
            WHERE candidate_key_duplicate_flag
        ),
        payloads AS (
            SELECT
                appid, steamid, unix_timestamp_created,
                {payload_columns},
                count(*) AS exact_instance_count
            FROM candidate_rows
            GROUP BY appid, steamid, unix_timestamp_created, {payload_columns}
        ),
        payload_stats AS (
            SELECT
                appid, steamid, unix_timestamp_created,
                count(*) AS distinct_payload_count,
                count(*) FILTER (WHERE exact_instance_count > 1) AS exact_subgroup_count,
                sum(exact_instance_count) AS instance_count
            FROM payloads
            GROUP BY appid, steamid, unix_timestamp_created
        ),
        candidate_stats AS (
            SELECT
                appid, steamid, unix_timestamp_created,
                count(DISTINCT review) AS distinct_review_text_count,
                count(DISTINCT unix_timestamp_updated) AS distinct_update_timestamp_count,
                count(DISTINCT source_file) AS distinct_source_file_count,
                min(unix_timestamp_updated) AS min_unix_timestamp_updated,
                max(unix_timestamp_updated) AS max_unix_timestamp_updated,
                min(created_at_utc) AS created_at_utc,
                min(source_file) AS first_source_file,
                max(source_file) AS last_source_file
            FROM candidate_rows
            GROUP BY appid, steamid, unix_timestamp_created
        )
        SELECT
            s.*,
            p.instance_count,
            p.instance_count - 1 AS analytically_redundant_count,
            p.distinct_payload_count,
            p.exact_subgroup_count,
            CASE
                WHEN p.distinct_payload_count = 1 THEN 'exact_duplicate'
                WHEN s.distinct_update_timestamp_count > 1 THEN 'likely_updated_version'
                WHEN s.distinct_review_text_count > 1
                    THEN 'text_variant_same_update_time'
                ELSE 'other_payload_variant_ambiguous'
            END AS classification,
            'retain canonical; for sensitivity analysis select candidate rank 1 '
                || '(latest update, deterministic tie-breakers)' AS recommended_rule
        FROM candidate_stats s
        JOIN payload_stats p USING (appid, steamid, unix_timestamp_created)
        ORDER BY appid, steamid, unix_timestamp_created
    """
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute(
            f"COPY ({query}) TO {_sql_literal(str(paths.duplicate_diagnostics))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        rows = connection.execute(
            f"""
            SELECT classification, count(*)
            FROM read_parquet({_sql_literal(str(paths.duplicate_diagnostics))})
            GROUP BY classification ORDER BY classification
            """
        ).fetchall()
    finally:
        connection.close()
    return {str(label): count for label, count in rows}


def build_audit_tables(paths: AnalysisPaths) -> dict[str, Any]:
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(paths.analysis_base_dir / "**" / "*.parquet")
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute(
            f"CREATE VIEW analysis_base AS SELECT * FROM read_parquet("
            f"{_sql_literal(pattern)}, hive_partitioning = true)"
        )
        table_queries = _audit_table_queries(paths)
        for name, query in table_queries.items():
            _write_query_table(connection, paths.tables_dir, name, query)
        summary = _collect_summary_metrics(connection, paths)
    finally:
        connection.close()
    return summary


def run_analysis_readiness(paths: AnalysisPaths, *, force: bool = False) -> dict[str, Any]:
    _require_canonical_inputs(paths)
    generated = [
        paths.output_root,
        paths.notebook_path,
        paths.audit_report,
        paths.target_report,
    ]
    existing = [path for path in generated if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite generated Phase 2 outputs: "
            + ", ".join(str(path) for path in existing)
        )
    if force:
        for path in existing:
            _remove_generated(path)
    base = build_analysis_base(paths)
    negative_count = build_negative_reviews(paths)
    unmatched = build_unmatched_diagnostics(paths)
    duplicates = build_duplicate_diagnostics(paths)
    summary = build_audit_tables(paths)
    provenance = _collect_provenance(paths)
    metrics = {
        "analysis_base": base.to_dict(),
        "negative_review_rows": negative_count,
        "unmatched_catalog_evidence": unmatched,
        "duplicate_classifications": duplicates,
        "summary": summary,
        "provenance": provenance,
    }
    paths.audit_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(paths.metrics_path, metrics)
    render_figures(paths)
    render_analysis_reports(paths, metrics)
    write_analysis_readiness_notebook(paths)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Steam review analysis readiness")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--canonical-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = AnalysisPaths.from_project_root(
        args.project_root,
        canonical_root=args.canonical_root,
        output_root=args.output_root,
    )
    result = run_analysis_readiness(paths, force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _audit_table_queries(paths: AnalysisPaths) -> dict[str, str]:
    unmatched = _sql_literal(str(paths.unmatched_diagnostics))
    duplicates = _sql_literal(str(paths.duplicate_diagnostics))
    split_rows = _split_rows_query()
    queries = {
        "temporal_by_year": """
            SELECT review_year, count(*) AS review_count,
                count(*) FILTER (WHERE voted_up) AS positive_reviews,
                count(*) FILTER (WHERE NOT voted_up) AS negative_reviews,
                count(DISTINCT appid) AS unique_games
            FROM analysis_base GROUP BY review_year ORDER BY review_year
        """,
        "temporal_by_month": """
            SELECT review_month, count(*) AS review_count,
                count(*) FILTER (WHERE voted_up) AS positive_reviews,
                count(*) FILTER (WHERE NOT voted_up) AS negative_reviews,
                count(DISTINCT appid) AS unique_games
            FROM analysis_base GROUP BY review_month ORDER BY review_month
        """,
        "source_csv_coverage": """
            SELECT source_file, count(*) AS review_count,
                count(DISTINCT appid) AS unique_games,
                min(created_at_utc) AS earliest_created_utc,
                max(created_at_utc) AS latest_created_utc,
                min(updated_at_utc) AS earliest_updated_utc,
                max(updated_at_utc) AS latest_updated_utc
            FROM analysis_base GROUP BY source_file ORDER BY source_file
        """,
        "match_by_year": """
            SELECT review_year, count(*) AS review_count,
                count(*) FILTER (WHERE metadata_matched) AS matched_reviews,
                count(*) FILTER (WHERE NOT metadata_matched) AS unmatched_reviews,
                count(*) FILTER (WHERE metadata_matched) / count(*)::DOUBLE AS match_rate
            FROM analysis_base GROUP BY review_year ORDER BY review_year
        """,
        "match_by_sentiment": """
            SELECT voted_up, count(*) AS review_count,
                count(*) FILTER (WHERE metadata_matched) AS matched_reviews,
                count(*) FILTER (WHERE NOT metadata_matched) AS unmatched_reviews,
                count(*) FILTER (WHERE metadata_matched) / count(*)::DOUBLE AS match_rate
            FROM analysis_base GROUP BY voted_up ORDER BY voted_up
        """,
        "match_by_review_volume_decile": """
            WITH games AS (
                SELECT appid, bool_or(metadata_matched) AS metadata_matched,
                    count(*) AS review_count
                FROM analysis_base GROUP BY appid
            ), ranked AS (
                SELECT *, ntile(10) OVER (ORDER BY review_count, appid) AS volume_decile
                FROM games
            )
            SELECT volume_decile, count(*) AS game_count, sum(review_count) AS review_count,
                count(*) FILTER (WHERE metadata_matched) AS matched_games,
                sum(review_count) FILTER (WHERE metadata_matched) AS matched_reviews,
                count(*) FILTER (WHERE metadata_matched) / count(*)::DOUBLE
                    AS game_match_rate,
                sum(review_count) FILTER (WHERE metadata_matched) / sum(review_count)::DOUBLE
                    AS row_match_rate,
                min(review_count) AS min_game_reviews,
                max(review_count) AS max_game_reviews
            FROM ranked GROUP BY volume_decile ORDER BY volume_decile
        """,
        "matched_unmatched_comparison": """
            SELECT metadata_matched, count(*) AS review_count,
                count(DISTINCT appid) AS appid_count,
                count(*) FILTER (WHERE voted_up) / count(*)::DOUBLE AS positive_share,
                approx_quantile(playtime_forever, 0.50) AS median_playtime_forever,
                approx_quantile(playtime_forever, 0.90) AS p90_playtime_forever,
                approx_quantile(num_games_owned, 0.50) AS median_games_owned,
                approx_quantile(num_reviews, 0.50) AS median_reviews_written,
                approx_quantile(votes_up, 0.50) AS median_votes_up,
                approx_quantile(votes_up, 0.99) AS p99_votes_up,
                approx_quantile(votes_funny, 0.99) AS p99_votes_funny,
                approx_quantile(weighted_vote_score, 0.50)
                    AS median_weighted_vote_score
            FROM analysis_base GROUP BY metadata_matched ORDER BY metadata_matched
        """,
        "top_50_unmatched_appids": f"""
            SELECT * FROM read_parquet({unmatched})
            ORDER BY review_count DESC, appid LIMIT 50
        """,
        "unmatched_concentration": f"""
            WITH ranked AS (
                SELECT *, row_number() OVER (ORDER BY review_count DESC, appid) AS rank,
                    sum(review_count) OVER () AS all_unmatched_rows
                FROM read_parquet({unmatched})
            ), cutoffs(cutoff) AS (VALUES (10), (25), (50), (100))
            SELECT cutoff,
                sum(review_count) FILTER (WHERE rank <= cutoff) AS explained_rows,
                max(all_unmatched_rows) AS all_unmatched_rows,
                sum(review_count) FILTER (WHERE rank <= cutoff)
                    / max(all_unmatched_rows)::DOUBLE AS explained_share
            FROM ranked CROSS JOIN cutoffs GROUP BY cutoff ORDER BY cutoff
        """,
        "duplicate_classification_summary": f"""
            SELECT classification, count(*) AS candidate_group_count,
                sum(instance_count) AS instance_count,
                sum(analytically_redundant_count) AS analytically_redundant_count
            FROM read_parquet({duplicates}) GROUP BY classification ORDER BY classification
        """,
        "playtime_by_sentiment": """
            SELECT voted_up, count(*) AS review_count,
                count(*) FILTER (WHERE additional_playtime_minutes IS NULL) AS null_count,
                count(*) FILTER (WHERE additional_playtime_negative) AS negative_count,
                count(*) FILTER (WHERE additional_playtime_zero) AS zero_count,
                count(*) FILTER (WHERE additional_playtime_positive) AS positive_count,
                count(*) FILTER (WHERE additional_playtime_positive) / count(*)::DOUBLE
                    AS positive_share,
                min(additional_playtime_minutes) AS minimum,
                approx_quantile(additional_playtime_minutes, 0.25) AS p25,
                approx_quantile(additional_playtime_minutes, 0.50) AS median,
                approx_quantile(additional_playtime_minutes, 0.75) AS p75,
                approx_quantile(additional_playtime_minutes, 0.90) AS p90,
                approx_quantile(additional_playtime_minutes, 0.95) AS p95,
                approx_quantile(additional_playtime_minutes, 0.99) AS p99,
                max(additional_playtime_minutes) AS maximum
            FROM analysis_base GROUP BY voted_up ORDER BY voted_up
        """,
        "playtime_by_year_sentiment": """
            SELECT review_year, voted_up, count(*) AS review_count,
                count(*) FILTER (WHERE additional_playtime_negative) AS negative_count,
                count(*) FILTER (WHERE additional_playtime_zero) AS zero_count,
                count(*) FILTER (WHERE additional_playtime_positive) AS positive_count,
                count(*) FILTER (WHERE additional_playtime_positive) / count(*)::DOUBLE
                    AS positive_share,
                approx_quantile(additional_playtime_minutes, 0.50) AS median,
                approx_quantile(additional_playtime_minutes, 0.90) AS p90,
                approx_quantile(additional_playtime_minutes, 0.99) AS p99
            FROM analysis_base GROUP BY review_year, voted_up ORDER BY review_year, voted_up
        """,
        "playtime_by_game_sentiment": """
            SELECT appid, game_name, metadata_matched, voted_up,
                count(*) AS review_count,
                count(*) FILTER (WHERE additional_playtime_positive) / count(*)::DOUBLE
                    AS positive_share,
                approx_quantile(additional_playtime_minutes, 0.50) AS median,
                approx_quantile(additional_playtime_minutes, 0.90) AS p90,
                approx_quantile(additional_playtime_minutes, 0.99) AS p99
            FROM analysis_base GROUP BY appid, game_name, metadata_matched, voted_up
            ORDER BY review_count DESC, appid, voted_up
        """,
        "helpfulness_by_sentiment": """
            SELECT voted_up, count(*) AS review_count,
                count(*) FILTER (WHERE votes_up IS NULL) AS votes_up_nulls,
                count(*) FILTER (WHERE votes_up = 0) / count(*)::DOUBLE
                    AS votes_up_zero_share,
                approx_quantile(votes_up, 0.50) AS votes_up_median,
                approx_quantile(votes_up, 0.90) AS votes_up_p90,
                approx_quantile(votes_up, 0.99) AS votes_up_p99,
                max(votes_up) AS votes_up_max,
                count(*) FILTER (WHERE votes_funny IS NULL) AS votes_funny_nulls,
                count(*) FILTER (WHERE votes_funny = 0) / count(*)::DOUBLE
                    AS votes_funny_zero_share,
                approx_quantile(votes_funny, 0.99) AS votes_funny_p99,
                max(votes_funny) AS votes_funny_max,
                count(*) FILTER (WHERE weighted_vote_score IS NULL) AS weighted_score_nulls,
                count(*) FILTER (WHERE weighted_vote_score = 0) / count(*)::DOUBLE
                    AS weighted_score_zero_share,
                approx_quantile(weighted_vote_score, 0.50) AS weighted_score_median,
                approx_quantile(weighted_vote_score, 0.99) AS weighted_score_p99,
                max(weighted_vote_score) AS weighted_score_max
            FROM analysis_base GROUP BY voted_up ORDER BY voted_up
        """,
        "helpfulness_by_year": """
            SELECT review_year, count(*) AS review_count,
                avg(votes_up) AS mean_votes_up,
                approx_quantile(votes_up, 0.50) AS median_votes_up,
                approx_quantile(votes_up, 0.99) AS p99_votes_up,
                avg(votes_funny) AS mean_votes_funny,
                avg(weighted_vote_score) AS mean_weighted_vote_score,
                count(*) FILTER (WHERE votes_up = 0) / count(*)::DOUBLE
                    AS votes_up_zero_share
            FROM analysis_base GROUP BY review_year ORDER BY review_year
        """,
        "helpfulness_extremes": """
            SELECT
                count(*) FILTER (WHERE votes_funny = 4294967295)
                    AS votes_funny_uint32_max_count,
                count(*) FILTER (WHERE votes_funny >= 1000000)
                    AS votes_funny_at_least_one_million,
                count(*) FILTER (WHERE votes_up >= 10000)
                    AS votes_up_at_least_ten_thousand,
                max(votes_funny) AS votes_funny_max,
                max(votes_up) AS votes_up_max,
                min(weighted_vote_score) AS weighted_vote_score_min,
                max(weighted_vote_score) AS weighted_vote_score_max
            FROM analysis_base
        """,
        "chronological_split_summary": f"""
            WITH split_rows AS ({split_rows}),
            base_counts AS (
                SELECT design, split, min_date, max_date, advantage, limitation,
                    review_age_bias_remains,
                    count(*) AS review_count,
                    count(*) FILTER (WHERE is_negative_review) AS negative_review_count,
                    count(DISTINCT appid) AS unique_games,
                    count(*) FILTER (WHERE playtime_inputs_present)
                        AS playtime_target_available
                FROM split_rows
                GROUP BY design, split, min_date, max_date, advantage, limitation,
                    review_age_bias_remains
            ), genres AS (
                SELECT design, split, count(DISTINCT genre) AS genre_coverage
                FROM split_rows, UNNEST(genres) t(genre)
                GROUP BY design, split
            )
            SELECT b.*, coalesce(g.genre_coverage, 0) AS genre_coverage
            FROM base_counts b LEFT JOIN genres g USING (design, split)
            ORDER BY design, CASE split WHEN 'train' THEN 1 WHEN 'validation' THEN 2 ELSE 3 END
        """,
    }
    if paths.canonical_mapping.exists():
        mapping = _sql_literal(str(paths.canonical_mapping))
        queries["metadata_catalog_timing"] = f"""
            SELECT count(*) AS metadata_appids,
                count(*) FILTER (WHERE release_date > DATE '2021-04-23')
                    AS releases_after_review_window,
                min(release_date) AS earliest_release_date,
                max(release_date) AS latest_release_date
            FROM read_parquet({mapping})
        """
    return queries


def _split_rows_query() -> str:
    return """
        SELECT 'A: broad historical holdout' AS design,
            CASE
                WHEN created_at_utc < TIMESTAMP '2019-01-01' THEN 'train'
                WHEN created_at_utc < TIMESTAMP '2020-01-01' THEN 'validation'
                ELSE 'test'
            END AS split,
            CASE
                WHEN created_at_utc < TIMESTAMP '2019-01-01' THEN DATE '2010-10-15'
                WHEN created_at_utc < TIMESTAMP '2020-01-01' THEN DATE '2019-01-01'
                ELSE DATE '2020-01-01'
            END AS min_date,
            CASE
                WHEN created_at_utc < TIMESTAMP '2019-01-01' THEN DATE '2018-12-31'
                WHEN created_at_utc < TIMESTAMP '2020-01-01' THEN DATE '2019-12-31'
                ELSE DATE '2021-04-23'
            END AS max_date,
            'largest training history and a genuinely forward holdout' AS advantage,
            'period and platform shifts; test combines 2020 with partial 2021' AS limitation,
            true AS review_age_bias_remains,
            *
        FROM analysis_base
        UNION ALL
        SELECT 'B: recent narrow cohort' AS design,
            CASE
                WHEN created_at_utc < TIMESTAMP '2020-01-01' THEN 'train'
                WHEN created_at_utc < TIMESTAMP '2020-07-01' THEN 'validation'
                ELSE 'test'
            END AS split,
            CASE
                WHEN created_at_utc < TIMESTAMP '2020-01-01' THEN DATE '2019-01-01'
                WHEN created_at_utc < TIMESTAMP '2020-07-01' THEN DATE '2020-01-01'
                ELSE DATE '2020-07-01'
            END AS min_date,
            CASE
                WHEN created_at_utc < TIMESTAMP '2020-01-01' THEN DATE '2019-12-31'
                WHEN created_at_utc < TIMESTAMP '2020-07-01' THEN DATE '2020-06-30'
                ELSE DATE '2020-12-31'
            END AS max_date,
            'closer review eras and avoids the incomplete 2021 tail' AS advantage,
            'uses less history and still cannot equalize unknown exposure time' AS limitation,
            true AS review_age_bias_remains,
            *
        FROM analysis_base
        WHERE created_at_utc >= TIMESTAMP '2019-01-01'
          AND created_at_utc < TIMESTAMP '2021-01-01'
        UNION ALL
        SELECT 'C: expanding-window example' AS design,
            CASE
                WHEN created_at_utc < TIMESTAMP '2019-01-01' THEN 'train'
                WHEN created_at_utc < TIMESTAMP '2020-01-01' THEN 'validation'
                ELSE 'test'
            END AS split,
            CASE
                WHEN created_at_utc < TIMESTAMP '2019-01-01' THEN DATE '2010-10-15'
                WHEN created_at_utc < TIMESTAMP '2020-01-01' THEN DATE '2019-01-01'
                ELSE DATE '2020-01-01'
            END AS min_date,
            CASE
                WHEN created_at_utc < TIMESTAMP '2019-01-01' THEN DATE '2018-12-31'
                WHEN created_at_utc < TIMESTAMP '2020-01-01' THEN DATE '2019-12-31'
                ELSE DATE '2020-12-31'
            END AS max_date,
            'supports repeated forward checks when extended to adjacent yearly folds'
                AS advantage,
            'higher operational complexity and exposure bias persists within every fold'
                AS limitation,
            true AS review_age_bias_remains,
            *
        FROM analysis_base
        WHERE created_at_utc < TIMESTAMP '2021-01-01'
    """


def _collect_summary_metrics(
    connection: duckdb.DuckDBPyConnection, paths: AnalysisPaths
) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT count(*) AS review_count,
            count(*) FILTER (WHERE voted_up) AS positive_reviews,
            count(*) FILTER (WHERE NOT voted_up) AS negative_reviews,
            count(DISTINCT appid) AS unique_appids,
            min(created_at_utc) AS earliest_created_utc,
            max(created_at_utc) AS latest_created_utc,
            min(updated_at_utc) AS earliest_updated_utc,
            max(updated_at_utc) AS latest_updated_utc,
            count(*) FILTER (WHERE updated_after_dataset_creation_max)
                AS updates_after_latest_creation,
            count(*) FILTER (WHERE updated_before_created) AS updates_before_creation,
            count(*) FILTER (WHERE additional_playtime_minutes IS NULL)
                AS additional_playtime_nulls,
            count(*) FILTER (WHERE additional_playtime_negative)
                AS additional_playtime_negative,
            count(*) FILTER (WHERE additional_playtime_zero) AS additional_playtime_zero,
            count(*) FILTER (WHERE additional_playtime_positive) AS additional_playtime_positive,
            count(*) FILTER (WHERE review_before_release) AS reviews_before_release
        FROM analysis_base
        """
    ).fetchone()
    names = [item[0] for item in connection.description]
    overview = dict(zip(names, row, strict=True))
    concentration_rows = _query_dicts(
        connection,
        f"SELECT * FROM read_parquet("
        f"{_sql_literal(str(paths.tables_dir / 'unmatched_concentration.parquet'))}) "
        "ORDER BY cutoff",
    )
    playtime = _query_dicts(
        connection,
        f"SELECT * FROM read_parquet("
        f"{_sql_literal(str(paths.tables_dir / 'playtime_by_sentiment.parquet'))}) "
        "ORDER BY voted_up",
    )
    helpfulness = _query_dicts(
        connection,
        f"SELECT * FROM read_parquet("
        f"{_sql_literal(str(paths.tables_dir / 'helpfulness_by_sentiment.parquet'))}) "
        "ORDER BY voted_up",
    )
    return {
        "overview": overview,
        "unmatched_concentration": concentration_rows,
        "playtime_by_sentiment": playtime,
        "helpfulness_by_sentiment": helpfulness,
        "snapshot_date_available": False,
        "exposure_adjusted_measures_created": False,
    }


def _collect_provenance(paths: AnalysisPaths) -> dict[str, Any]:
    files = []
    for path in sorted(paths.source_dir.glob("reviews-*.csv")):
        stat = path.stat()
        files.append(
            {
                "file": path.name,
                "size_bytes": stat.st_size,
                "filesystem_modified_at": stat.st_mtime,
                "interpretation": "local packaging timestamp; not an observation date",
            }
        )
    return {
        "dataset_url": KAGGLE_DATASET_URL,
        "author": DATASET_AUTHOR,
        "access_date": date.today().isoformat(),
        "license": "license not specified",
        "publication_date": "not established from accessible source evidence",
        "dataset_update_date": "not established; Kaggle exposes only a relative update label",
        "review_collection_date": "not established",
        "common_observation_snapshot": False,
        "playtime_common_measurement_date": False,
        "continuous_collection_dates": "not established",
        "source_csv_files": files,
        "conclusion": (
            "No source statement ties an upload, update, or filesystem date to review or "
            "playtime observation. Exposure time therefore cannot be reconstructed."
        ),
    }


def _write_query_table(
    connection: duckdb.DuckDBPyConnection, directory: Path, name: str, query: str
) -> None:
    parquet_path = directory / f"{name}.parquet"
    csv_path = directory / f"{name}.csv"
    for path in (parquet_path, csv_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite generated table: {path}")
    connection.execute(
        f"COPY ({query}) TO {_sql_literal(str(parquet_path))} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.execute(f"COPY ({query}) TO {_sql_literal(str(csv_path))} (FORMAT CSV, HEADER true)")


def _query_dicts(connection: duckdb.DuckDBPyConnection, query: str) -> list[dict[str, Any]]:
    cursor = connection.execute(query)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _parquet_row_count(directory: Path) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in directory.rglob("*.parquet"))


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _ensure_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite generated output: {path}")


def _remove_generated(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _require_canonical_inputs(paths: AnalysisPaths) -> None:
    required = [paths.enriched_dir, paths.canonical_unmatched, paths.steamdb_json]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase 2 inputs: " + ", ".join(missing))


if __name__ == "__main__":
    raise SystemExit(main())
