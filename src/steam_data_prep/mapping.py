from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import ijson
import pyarrow as pa
import pyarrow.parquet as pq

MAPPING_SCHEMA = pa.schema(
    [
        pa.field("appid", pa.int64(), nullable=False),
        pa.field("game_name", pa.string()),
        pa.field("genres", pa.list_(pa.string())),
        pa.field("release_date", pa.date32()),
        pa.field("publishers", pa.list_(pa.string())),
    ]
)


@dataclass(frozen=True)
class MappingMetrics:
    total_json_records: int
    steam_records: int
    eligible_records: int
    unique_appids: int
    duplicate_appid_count: int
    duplicate_candidate_rows: int
    invalid_release_dates: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_mapping(
    json_path: Path,
    parquet_path: Path,
    csv_path: Path,
    duplicate_audit_path: Path,
    *,
    batch_size: int = 10_000,
) -> MappingMetrics:
    """Stream SteamDB into a compact unique mapping and duplicate audit."""
    start = time.monotonic()
    _ensure_new_targets(parquet_path, csv_path, duplicate_audit_path)
    for target in (parquet_path, csv_path, duplicate_audit_path):
        target.parent.mkdir(parents=True, exist_ok=True)

    temp_parquet = parquet_path.with_name(f".{parquet_path.name}.tmp")
    temp_csv = csv_path.with_name(f".{csv_path.name}.tmp")
    temp_duplicates = duplicate_audit_path.with_name(f".{duplicate_audit_path.name}.tmp")

    total = steam = eligible = invalid_dates = duplicate_rows = 0
    duplicate_appids: set[int] = set()
    audited_first: set[int] = set()
    seen: dict[int, tuple[int, dict[str, Any]]] = {}
    batch: list[dict[str, Any]] = []
    writer = pq.ParquetWriter(temp_parquet, MAPPING_SCHEMA, compression="zstd")
    duplicate_fields = [
        "appid",
        "source_index",
        "selected",
        "game_name",
        "genres",
        "release_date",
        "publishers",
    ]
    try:
        with (
            json_path.open("rb") as source,
            temp_csv.open("w", encoding="utf-8", newline="") as csv_handle,
            temp_duplicates.open("w", encoding="utf-8", newline="") as dup_handle,
        ):
            mapping_csv = csv.DictWriter(csv_handle, fieldnames=MAPPING_SCHEMA.names)
            duplicate_csv = csv.DictWriter(dup_handle, fieldnames=duplicate_fields)
            mapping_csv.writeheader()
            duplicate_csv.writeheader()
            for source_index, record in enumerate(ijson.items(source, "item")):
                total += 1
                if record.get("kind") != "steam":
                    continue
                steam += 1
                raw_appid = record.get("steam_appid")
                if raw_appid is None:
                    continue
                eligible += 1
                appid = int(raw_appid)
                parsed_date, invalid = _parse_date(record.get("release_date"))
                invalid_dates += int(invalid)
                compact = {
                    "appid": appid,
                    "game_name": _optional_string(record.get("name")),
                    "genres": _string_list(record.get("genres")),
                    "release_date": parsed_date,
                    "publishers": _string_list(record.get("publishers")),
                }
                if appid in seen:
                    duplicate_appids.add(appid)
                    duplicate_rows += 1
                    if appid not in audited_first:
                        first_index, first = seen[appid]
                        duplicate_csv.writerow(_duplicate_row(first_index, True, first))
                        audited_first.add(appid)
                    duplicate_csv.writerow(_duplicate_row(source_index, False, compact))
                    continue

                seen[appid] = (source_index, compact)
                batch.append(compact)
                mapping_csv.writerow(_mapping_csv_row(compact))
                if len(batch) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(batch, schema=MAPPING_SCHEMA))
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=MAPPING_SCHEMA))
    except Exception:
        writer.close()
        _remove_temps(temp_parquet, temp_csv, temp_duplicates)
        raise
    else:
        writer.close()
        temp_parquet.replace(parquet_path)
        temp_csv.replace(csv_path)
        temp_duplicates.replace(duplicate_audit_path)

    return MappingMetrics(
        total_json_records=total,
        steam_records=steam,
        eligible_records=eligible,
        unique_appids=len(seen),
        duplicate_appid_count=len(duplicate_appids),
        duplicate_candidate_rows=duplicate_rows,
        invalid_release_dates=invalid_dates,
        elapsed_seconds=round(time.monotonic() - start, 3),
    )


def _parse_date(value: Any) -> tuple[date | None, bool]:
    if value in (None, ""):
        return None, False
    try:
        return date.fromisoformat(str(value)), False
    except ValueError:
        return None, True


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return [str(value)]
    return [str(item) for item in value]


def _mapping_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    release_date = record["release_date"]
    return {
        "appid": record["appid"],
        "game_name": record["game_name"],
        "genres": json.dumps(record["genres"], ensure_ascii=False),
        "release_date": release_date.isoformat() if release_date else "",
        "publishers": json.dumps(record["publishers"], ensure_ascii=False),
    }


def _duplicate_row(index: int, selected: bool, record: dict[str, Any]) -> dict[str, Any]:
    row = _mapping_csv_row(record)
    return {
        "appid": row["appid"],
        "source_index": index,
        "selected": str(selected).lower(),
        "game_name": row["game_name"],
        "genres": row["genres"],
        "release_date": row["release_date"],
        "publishers": row["publishers"],
    }


def _ensure_new_targets(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite generated outputs: " + ", ".join(existing))


def _remove_temps(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
