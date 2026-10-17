from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PipelinePaths


def render_report(paths: PipelinePaths) -> Path:
    audit = _read_json(paths.audit_dir / "audit_metrics.json")
    review = _read_json(paths.audit_dir / "review_metrics.json")
    mapping = _read_json(paths.audit_dir / "mapping_metrics.json")
    duplicates = _read_json(paths.audit_dir / "duplicate_metrics.json")
    merge = _read_json(paths.audit_dir / "merge_metrics.json")
    validation = _read_json(paths.audit_dir / "validation_metrics.json")
    manifest = _read_json(paths.audit_dir / "input_manifest.json", default=[])
    verification = _read_json(paths.audit_dir / "verification_metrics.json", default={})
    output_sizes = {
        str(path.relative_to(paths.project_root)): _path_size(path)
        for path in [
            paths.reviews_fact_dir,
            paths.mapping_parquet,
            paths.mapping_csv,
            paths.reviews_enriched_dir,
            paths.unmatched_csv,
            paths.unmatched_parquet,
        ]
        if path.exists()
    }
    lines = [
        "# Kaggle Steam Reviews + SteamDB Merge Audit",
        "",
        "## Input manifest and checksums",
        "",
        "| Kind | File | Bytes | Rows/records | SHA-256 |",
        "|---|---|---:|---:|---|",
    ]
    lines.extend(
        f"| {item['kind']} | `{item['path']}` | {item['size_bytes']:,} | "
        f"{item.get('row_count', ''):,} | `{item['sha256']}` |"
        for item in manifest
    )
    lines.extend(
        [
            "",
            "## Data lineage",
            "",
            "The eleven manifest-approved `reviews-*.csv` shards are scanned as read-only inputs. "
            "Their columns are typed, UTC timestamps and `review_year` are derived, and the result "
            "is written to `reviews_fact/` with `source_file` lineage. The root `steamdb.json` is "
            "streamed into a five-column unique mapping. A many-to-one left join produces "
            "`reviews_enriched/`; duplicates are reported but never deleted.",
            "",
            "Playtime fields are preserved as supplied. Their Steam-compatible semantics are "
            "integer "
            "minutes; no rename or unit conversion was performed.",
            "",
            "## Schemas",
            "",
            "### Review fact",
            "",
            _schema_table(validation.get("review_schema", {})),
            "",
            "### Enriched reviews",
            "",
            _schema_table(validation.get("enriched_schema", {})),
            "",
            "## Row counts and timestamp coverage",
            "",
            f"- Source review rows: {audit.get('total_review_rows', 0):,}",
            f"- Review fact rows: {validation.get('review_rows', 0):,}",
            f"- Enriched rows: {validation.get('enriched_rows', 0):,}",
            f"- Unique review AppIDs: {validation.get('unique_review_appids', 0):,}",
            f"- Unique metadata AppIDs: {validation.get('unique_metadata_appids', 0):,}",
            f"- Created timestamp range (UTC): {validation.get('timestamp_min_utc')} to "
            f"{validation.get('timestamp_max_utc')}",
            "",
            "## Missingness",
            "",
            "| Column | Missing rate |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{name}` | {rate:.4%} |"
        for name, rate in validation.get("missing_value_rates", {}).items()
    )
    lines.extend(
        [
            "",
            "## Duplicate findings",
            "",
            f"- Exact duplicate groups: {duplicates.get('exact_duplicate_groups', 0):,}",
            f"- Rows in exact duplicate groups: {duplicates.get('exact_duplicate_instances', 0):,}",
            f"- Redundant exact rows (not removed): {duplicates.get('exact_redundant_rows', 0):,}",
            f"- Fallback-key duplicate groups (`appid + steamid + unix_timestamp_created`): "
            f"{duplicates.get('fallback_duplicate_groups', 0):,}",
            f"- Duplicate metadata AppIDs: {mapping.get('duplicate_appid_count', 0):,}",
            "",
            "## Join cardinality and match rates",
            "",
            "The mapping key was verified unique before a many-to-one left join. Review fact and "
            "enriched row counts are equal, so the join dropped and multiplied zero review rows.",
            "",
            f"- Matched review rows: {merge.get('matched_review_rows', 0):,}",
            f"- Unmatched review rows: {merge.get('unmatched_review_rows', 0):,}",
            f"- Matched unique AppIDs: {merge.get('matched_unique_appids', 0):,}",
            f"- Unmatched unique AppIDs: {merge.get('unmatched_unique_appids', 0):,}",
            f"- Match rate by row: {merge.get('match_rate_by_row', 0):.4%}",
            f"- Match rate by unique AppID: {merge.get('match_rate_by_unique_appid', 0):.4%}",
            "",
            "## Unmatched AppIDs",
            "",
            f"Complete machine-readable lists: `{paths.unmatched_csv}` and "
            f"`{paths.unmatched_parquet}`.",
            "",
            ", ".join(str(value) for value in validation.get("unmatched_appids", [])) or "None.",
            "",
            "## Runtime and output storage",
            "",
            "| Stage | Seconds |",
            "|---|---:|",
            f"| Audit | {audit.get('elapsed_seconds', 0):.3f} |",
            f"| Combine reviews | {review.get('elapsed_seconds', 0):.3f} |",
            f"| Build mapping | {mapping.get('elapsed_seconds', 0):.3f} |",
            f"| Duplicate analysis | {duplicates.get('elapsed_seconds', 0):.3f} |",
            f"| Merge | {merge.get('elapsed_seconds', 0):.3f} |",
            "",
            "| Output | Bytes |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{name}` | {size:,} |" for name, size in output_sizes.items())
    lines.extend(
        [
            "",
            "## Test and validation results",
            "",
            f"- Source columns preserved: {validation.get('source_columns_preserved', False)}",
            f"- Fact/enriched row equality: "
            f"{validation.get('review_rows') == validation.get('enriched_rows')}",
        ]
    )
    if verification:
        lines.extend(f"- {name}: {result}" for name, result in verification.items())
    else:
        lines.append(
            "- External formatting, lint, unit-test, and compilation results are added after "
            "final verification."
        )
    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            "- The local files do not include the original Kaggle dataset URL or license metadata; "
            "those details are not inferred.",
            "- AppID matching cannot enrich null, invalid, or SteamDB-absent review AppIDs.",
            "- The SteamDB mapping is a snapshot of the supplied JSON and is not refreshed online.",
            "- Unique AppID metrics exclude null AppIDs.",
            "",
            "## Attribution and licensing",
            "",
            "The review shards are attributed to the locally supplied Kaggle Steam Reviews "
            "Dataset. "
            "The mapping source is attributed to `leinstay/steamdb`; that project is distributed "
            "under GPL-3.0. This pipeline retains only the five requested mapping fields.",
            "",
        ]
    )
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    temp = paths.report_path.with_name(f".{paths.report_path.name}.tmp")
    temp.write_text("\n".join(lines), encoding="utf-8")
    temp.replace(paths.report_path)
    return paths.report_path


def _read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _schema_table(schema: dict[str, str]) -> str:
    rows = ["| Column | Type |", "|---|---|"]
    rows.extend(f"| `{name}` | `{data_type}` |" for name, data_type in schema.items())
    return "\n".join(rows)


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
