from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import ijson
import pyarrow as pa
import pyarrow.parquet as pq

from .reviews import _sql_literal

GENRE_MAP = {
    "Free To Play": "Free to Play",
    "RPG": "Role-playing (RPG)",
    "Role-playing": "Role-playing (RPG)",
    "Sport": "Sports",
    "Platform": "Platformer",
    "Simulator": "Simulation",
    "Card Game": "Card & Board Game",
    "Real Time Strategy (RTS)": "Real-time",
    "Turn-based strategy (TBS)": "Turn-based",
}


@dataclass(frozen=True)
class GenreBuildMetrics:
    input_rows: int
    output_rows: int
    input_appids: int
    mapped_appids: int
    unmapped_appids: int
    mapped_rows: int
    unmapped_rows: int
    threshold_rule: str
    main_genres: tuple[str, ...]
    rare_genres: tuple[str, ...]
    genre_game_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_genres(genres: Any) -> tuple[str, ...]:
    if not isinstance(genres, (list, tuple, set)):
        return ()
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_genre in genres:
        if raw_genre is None:
            continue
        genre = GENRE_MAP.get(str(raw_genre), str(raw_genre))
        if genre not in seen:
            cleaned.append(genre)
            seen.add(genre)
    return tuple(cleaned)


def count_genres_by_game(
    mapping: dict[int, tuple[str, ...]],
    input_appids: Iterable[int],
    *,
    threshold: int,
) -> tuple[dict[str, int], tuple[str, ...], frozenset[str]]:
    counts: Counter[str] = Counter()
    for appid in set(input_appids):
        counts.update(mapping.get(appid, ()))
    ordered_counts = dict(sorted(counts.items()))
    main_genres = tuple(sorted(genre for genre, count in counts.items() if count > threshold))
    rare_genres = frozenset(genre for genre, count in counts.items() if count <= threshold)
    return ordered_counts, main_genres, rare_genres


def load_steamdb_genres(
    steamdb_path: Path, *, target_appids: set[int]
) -> dict[int, tuple[str, ...]]:
    mapping: dict[int, tuple[str, ...]] = {}
    with steamdb_path.open("rb") as handle:
        for record in ijson.items(handle, "item"):
            if record.get("kind") != "steam":
                continue
            raw_appid = record.get("steam_appid")
            if raw_appid is None:
                continue
            appid = int(raw_appid)
            if appid not in target_appids or appid in mapping:
                continue
            mapping[appid] = clean_genres(record.get("genres"))
    return mapping


def build_genre_features(
    input_path: Path,
    steamdb_path: Path,
    output_path: Path,
    *,
    threshold: int = 50,
) -> GenreBuildMetrics:
    input_path = input_path.expanduser().resolve()
    steamdb_path = steamdb_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite generated output: {output_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input Parquet not found: {input_path}")
    if not steamdb_path.is_file():
        raise FileNotFoundError(f"SteamDB JSON not found: {steamdb_path}")
    if threshold < 0:
        raise ValueError("Genre threshold must be nonnegative")

    connection = duckdb.connect()
    try:
        input_description = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet({_sql_literal(str(input_path))})"
        ).fetchall()
        input_columns = [row[0] for row in input_description]
        if "appid" not in input_columns:
            raise ValueError("Input Parquet must contain an appid column")
        input_rows, input_appids = connection.execute(
            f"SELECT count(*), count(DISTINCT appid) FROM "
            f"read_parquet({_sql_literal(str(input_path))})"
        ).fetchone()
        target_appids = {
            int(row[0])
            for row in connection.execute(
                f"SELECT DISTINCT appid FROM read_parquet({_sql_literal(str(input_path))}) "
                "WHERE appid IS NOT NULL"
            ).fetchall()
        }
        mapping = load_steamdb_genres(steamdb_path, target_appids=target_appids)
        counts, main_genres, rare_genres = count_genres_by_game(
            mapping,
            target_appids,
            threshold=threshold,
        )
        dummy_table = _dummy_mapping_table(mapping, main_genres, rare_genres)
        connection.register("genre_dummy_mapping", dummy_table)
        select_dummies = ",\n".join(
            f"CAST(coalesce(g.{_quote_identifier(_dummy_name(genre))}, 0) AS UTINYINT) "
            f"AS {_quote_identifier(_dummy_name(genre))}"
            for genre in main_genres
        )
        if select_dummies:
            select_dummies += ",\n"
        query = f"""
            SELECT
                r.*,
                {select_dummies}
                CAST(coalesce(g.genre_other, 0) AS UTINYINT) AS genre_other
            FROM read_parquet({_sql_literal(str(input_path))}) r
            LEFT JOIN genre_dummy_mapping g USING (appid)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
        try:
            connection.execute(
                f"COPY ({query}) TO {_sql_literal(str(temp_path))} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
            )
            output_rows = pq.ParquetFile(temp_path).metadata.num_rows
            if output_rows != input_rows:
                raise RuntimeError(
                    f"Genre output row mismatch: input={input_rows}, output={output_rows}"
                )
            output_columns = [
                row[0]
                for row in connection.execute(
                    f"DESCRIBE SELECT * FROM read_parquet({_sql_literal(str(temp_path))})"
                ).fetchall()
            ]
            expected_columns = [
                *input_columns,
                *(_dummy_name(genre) for genre in main_genres),
                "genre_other",
            ]
            if output_columns != expected_columns:
                raise RuntimeError("Genre output schema differs from the expected schema")
            invalid_dummy_rows = connection.execute(
                f"SELECT count(*) FROM read_parquet({_sql_literal(str(temp_path))}) "
                f"WHERE NOT ({_dummy_validation_expression(main_genres)})"
            ).fetchone()[0]
            if invalid_dummy_rows:
                raise RuntimeError(f"Found {invalid_dummy_rows} rows with invalid dummy values")
            temp_path.replace(output_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        matched_rows = connection.execute(
            f"""
            SELECT count(*)
            FROM read_parquet({_sql_literal(str(input_path))}) r
            JOIN genre_dummy_mapping g USING (appid)
            """
        ).fetchone()[0]
    finally:
        connection.close()

    mapped_appids = len(mapping)
    return GenreBuildMetrics(
        input_rows=input_rows,
        output_rows=output_rows,
        input_appids=input_appids,
        mapped_appids=mapped_appids,
        unmapped_appids=input_appids - mapped_appids,
        mapped_rows=matched_rows,
        unmapped_rows=input_rows - matched_rows,
        threshold_rule=f"> {threshold} distinct input AppIDs",
        main_genres=main_genres,
        rare_genres=tuple(sorted(rare_genres)),
        genre_game_counts=counts,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add SteamDB genre dummy variables")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path)
    parser.add_argument("--steamdb", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    input_path = args.input or root / "runs" / "data_preparation" / "negative_reviews.parquet"
    steamdb_path = args.steamdb or root / "data" / "external" / "steamdb.json"
    output_path = args.output or root / "runs" / "data_preparation" / "negative_reviews_new.parquet"
    metrics = build_genre_features(
        input_path,
        steamdb_path,
        output_path,
        threshold=args.threshold,
    )
    print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
    return 0


def _dummy_mapping_table(
    mapping: dict[int, tuple[str, ...]],
    main_genres: tuple[str, ...],
    rare_genres: frozenset[str],
) -> pa.Table:
    appids = sorted(mapping)
    data: dict[str, pa.Array] = {
        "appid": pa.array(appids, type=pa.int64()),
    }
    for genre in main_genres:
        data[_dummy_name(genre)] = pa.array(
            [int(genre in mapping[appid]) for appid in appids],
            type=pa.uint8(),
        )
    data["genre_other"] = pa.array(
        [int(any(genre in rare_genres for genre in mapping[appid])) for appid in appids],
        type=pa.uint8(),
    )
    return pa.table(data)


def _dummy_name(genre: str) -> str:
    return f"genre_{genre}"


def _dummy_validation_expression(main_genres: tuple[str, ...]) -> str:
    columns = [*(_dummy_name(genre) for genre in main_genres), "genre_other"]
    return " AND ".join(f"{_quote_identifier(column)} IN (0, 1)" for column in columns)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


if __name__ == "__main__":
    raise SystemExit(main())
