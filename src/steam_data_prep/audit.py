from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ijson
import pyarrow as pa
import pyarrow.csv as pacsv

from .config import PipelinePaths, discover_review_files


class InputAuditError(RuntimeError):
    """Raised when source inputs are unsafe for a pipeline run."""


@dataclass
class AuditResult:
    review_files: list[Path]
    review_columns: list[str]
    schemas_consistent: bool
    total_review_rows: int
    total_review_bytes: int
    total_input_bytes: int
    available_disk_bytes: int
    estimated_required_bytes: int
    json_total_records: int
    json_kind_counts: dict[str, int]
    json_eligible_steam_records: int
    manifest: list[dict[str, Any]]
    elapsed_seconds: float

    def write(self, audit_dir: Path) -> None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(audit_dir / "input_manifest.json", self.manifest)
        fieldnames = ["kind", "path", "size_bytes", "row_count", "sha256", "columns"]
        temp_csv = audit_dir / ".input_manifest.csv.tmp"
        with temp_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.manifest:
                row = {key: entry.get(key) for key in fieldnames}
                row["columns"] = json.dumps(row["columns"]) if row["columns"] else ""
                writer.writerow(row)
        temp_csv.replace(audit_dir / "input_manifest.csv")
        metrics = asdict(self)
        metrics["review_files"] = [str(path) for path in self.review_files]
        _atomic_json(audit_dir / "audit_metrics.json", metrics)


def check_disk_capacity(*, input_bytes: int, available_bytes: int, multiplier: float = 4.0) -> int:
    required = int(input_bytes * multiplier)
    if available_bytes < required:
        raise InputAuditError(
            "Insufficient disk space: "
            f"need at least {required:,} bytes, found {available_bytes:,} bytes"
        )
    return required


def run_audit(
    paths: PipelinePaths,
    *,
    rows_per_file: int | None = None,
    disk_multiplier: float = 4.0,
) -> AuditResult:
    start = time.monotonic()
    review_files = discover_review_files(paths.source_dir)
    if not review_files:
        raise InputAuditError(f"No original review CSVs found in {paths.source_dir}")
    if not paths.steamdb_json.is_file():
        raise InputAuditError(f"SteamDB JSON not found: {paths.steamdb_json}")

    manifest: list[dict[str, Any]] = []
    headers: list[list[str]] = []
    total_rows = 0
    total_review_bytes = 0
    for path in review_files:
        columns = _read_header(path)
        headers.append(columns)
        row_count = _count_csv_rows(path, columns, limit=rows_per_file)
        size = path.stat().st_size
        total_rows += row_count
        total_review_bytes += size
        manifest.append(
            {
                "kind": "review_csv",
                "path": str(path.relative_to(paths.project_root)),
                "size_bytes": size,
                "row_count": row_count,
                "sha256": sha256_file(path),
                "columns": columns,
            }
        )

    canonical = headers[0]
    inconsistent = [
        path.name
        for path, header in zip(review_files, headers, strict=False)
        if header != canonical
    ]
    if inconsistent:
        raise InputAuditError(
            "Review CSV schema inconsistency detected in: " + ", ".join(inconsistent)
        )

    json_stats = _audit_json(paths.steamdb_json)
    json_size = paths.steamdb_json.stat().st_size
    manifest.append(
        {
            "kind": "steamdb_json",
            "path": str(paths.steamdb_json.relative_to(paths.project_root)),
            "size_bytes": json_size,
            "row_count": json_stats["total_records"],
            "sha256": sha256_file(paths.steamdb_json),
            "columns": None,
        }
    )
    total_input_bytes = total_review_bytes + json_size
    available = shutil.disk_usage(paths.project_root).free
    required = check_disk_capacity(
        input_bytes=total_input_bytes,
        available_bytes=available,
        multiplier=disk_multiplier,
    )
    return AuditResult(
        review_files=review_files,
        review_columns=canonical,
        schemas_consistent=True,
        total_review_rows=total_rows,
        total_review_bytes=total_review_bytes,
        total_input_bytes=total_input_bytes,
        available_disk_bytes=available,
        estimated_required_bytes=required,
        json_total_records=json_stats["total_records"],
        json_kind_counts=json_stats["kind_counts"],
        json_eligible_steam_records=json_stats["eligible_steam_records"],
        manifest=manifest,
        elapsed_seconds=round(time.monotonic() - start, 3),
    )


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            return next(csv.reader(handle))
        except StopIteration as exc:
            raise InputAuditError(f"Empty review CSV: {path}") from exc


def _count_csv_rows(path: Path, columns: list[str], *, limit: int | None) -> int:
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True),
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
        convert_options=pacsv.ConvertOptions(
            column_types={column: pa.string() for column in columns},
            strings_can_be_null=True,
        ),
    )
    count = 0
    for batch in reader:
        count += batch.num_rows
        if limit is not None and count >= limit:
            return limit
    return count


def _audit_json(path: Path) -> dict[str, Any]:
    kinds: Counter[str] = Counter()
    total = 0
    eligible = 0
    with path.open("rb") as handle:
        try:
            for record in ijson.items(handle, "item"):
                total += 1
                kind = str(record.get("kind"))
                kinds[kind] += 1
                if kind == "steam" and record.get("steam_appid") is not None:
                    eligible += 1
        except ijson.JSONError as exc:
            raise InputAuditError(f"Invalid SteamDB JSON structure in {path}: {exc}") from exc
    return {
        "total_records": total,
        "kind_counts": dict(sorted(kinds.items())),
        "eligible_steam_records": eligible,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(path)
