from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    source_dir: Path
    steamdb_json: Path
    output_root: Path
    audit_dir: Path
    reviews_fact_dir: Path
    mapping_parquet: Path
    mapping_csv: Path
    reviews_enriched_dir: Path
    unmatched_csv: Path
    unmatched_parquet: Path
    report_path: Path
    notebook_path: Path

    @classmethod
    def from_project_root(
        cls,
        root: str | Path,
        output_root: str | Path | None = None,
        *,
        dry_run: bool = False,
    ) -> PipelinePaths:
        project_root = Path(root).expanduser().resolve()
        source_dir = project_root / "data" / "raw" / "reviews"
        root_json = project_root / "data" / "external" / "steamdb.json"
        nested_json = source_dir / "steamdb.json"
        steamdb_json = root_json if root_json.exists() or not nested_json.exists() else nested_json
        if output_root is None:
            output_name = "kaggle_dry_run" if dry_run else "kaggle"
            resolved_output = project_root / "runs" / "data_preparation" / output_name
        else:
            resolved_output = Path(output_root).expanduser().resolve()
        report_path = (
            resolved_output / "reports" / "kaggle_steamdb_merge_audit.md"
            if dry_run or output_root is not None
            else project_root / "runs" / "data_preparation" / "reports" / "kaggle_steamdb_merge_audit.md"
        )
        notebook_path = (
            resolved_output / "notebooks" / "01_data_preparation_validation.ipynb"
            if dry_run or output_root is not None
            else project_root / "runs" / "data_preparation" / "notebooks" / "01_data_preparation_validation.ipynb"
        )
        return cls(
            project_root=project_root,
            source_dir=source_dir,
            steamdb_json=steamdb_json,
            output_root=resolved_output,
            audit_dir=resolved_output / "audit",
            reviews_fact_dir=resolved_output / "reviews_fact",
            mapping_parquet=resolved_output / "steam_games_mapping.parquet",
            mapping_csv=resolved_output / "steam_games_mapping.csv",
            reviews_enriched_dir=resolved_output / "reviews_enriched",
            unmatched_csv=resolved_output / "unmatched_appids.csv",
            unmatched_parquet=resolved_output / "unmatched_appids.parquet",
            report_path=report_path,
            notebook_path=notebook_path,
        )


def discover_review_files(source_dir: Path) -> list[Path]:
    """Return only original Kaggle review shards, in deterministic order."""
    return sorted(path for path in source_dir.glob("reviews-*.csv") if path.is_file())
