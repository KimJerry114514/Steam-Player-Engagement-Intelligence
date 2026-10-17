from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .audit import _atomic_json, run_audit
from .config import PipelinePaths
from .mapping import build_mapping
from .merge import MergeMetrics, merge_reviews
from .notebook import write_validation_notebook
from .report import render_report
from .reviews import combine_reviews
from .validation import analyze_review_duplicates, validate_outputs

STAGES = ["audit", "combine-reviews", "build-mapping", "merge", "validate", "all"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Kaggle Steam reviews with SteamDB metadata"
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in STAGES:
        subparser = subparsers.add_parser(stage)
        subparser.add_argument("--project-root", type=Path, default=Path.cwd())
        subparser.add_argument("--output-root", type=Path)
        subparser.add_argument("--dry-run", action="store_true")
        subparser.add_argument("--rows-per-file", type=int)
        subparser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = PipelinePaths.from_project_root(
        args.project_root,
        output_root=args.output_root,
        dry_run=args.dry_run,
    )
    row_limit = args.rows_per_file
    if args.dry_run and row_limit is None:
        row_limit = 1_000

    if args.stage == "all":
        metrics = _run_all(paths, row_limit=row_limit, force=args.force)
    elif args.stage == "audit":
        _prepare_target(paths.audit_dir, force=args.force)
        audit = run_audit(paths, rows_per_file=row_limit)
        audit.write(paths.audit_dir)
        metrics = _serializable(audit)
    elif args.stage == "combine-reviews":
        _prepare_target(paths.reviews_fact_dir, force=args.force)
        audit = run_audit(paths, rows_per_file=row_limit)
        if not paths.audit_dir.exists():
            audit.write(paths.audit_dir)
        result = combine_reviews(audit, paths.reviews_fact_dir, rows_per_file=row_limit)
        _write_metrics(paths.audit_dir / "review_metrics.json", result)
        metrics = result.to_dict()
    elif args.stage == "build-mapping":
        targets = [
            paths.mapping_parquet,
            paths.mapping_csv,
            paths.audit_dir / "duplicate_metadata_appids.csv",
        ]
        for target in targets:
            _prepare_target(target, force=args.force)
        result = build_mapping(paths.steamdb_json, *targets)
        _write_metrics(paths.audit_dir / "mapping_metrics.json", result)
        metrics = result.to_dict()
    elif args.stage == "merge":
        for target in [paths.reviews_enriched_dir, paths.unmatched_csv, paths.unmatched_parquet]:
            _prepare_target(target, force=args.force)
        result = merge_reviews(
            paths.reviews_fact_dir,
            paths.mapping_parquet,
            paths.reviews_enriched_dir,
            unmatched_csv=paths.unmatched_csv,
            unmatched_parquet=paths.unmatched_parquet,
        )
        _write_metrics(paths.audit_dir / "merge_metrics.json", result)
        metrics = result.to_dict()
    else:
        metrics = _run_validation(paths, force=args.force)

    print(json.dumps(metrics, indent=2, sort_keys=True, default=str))
    return 0


def _run_all(paths: PipelinePaths, *, row_limit: int | None, force: bool) -> dict[str, Any]:
    _validate_output_root_safety(paths)
    existing = [
        path
        for path in [paths.output_root, paths.report_path, paths.notebook_path]
        if path.exists()
    ]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite generated outputs: " + ", ".join(str(path) for path in existing)
        )
    if force:
        for path in existing:
            _remove_generated(path)

    audit = run_audit(paths, rows_per_file=row_limit)
    audit.write(paths.audit_dir)
    review = combine_reviews(audit, paths.reviews_fact_dir, rows_per_file=row_limit)
    _write_metrics(paths.audit_dir / "review_metrics.json", review)
    mapping = build_mapping(
        paths.steamdb_json,
        paths.mapping_parquet,
        paths.mapping_csv,
        paths.audit_dir / "duplicate_metadata_appids.csv",
    )
    _write_metrics(paths.audit_dir / "mapping_metrics.json", mapping)
    duplicates = analyze_review_duplicates(paths.reviews_fact_dir, paths.audit_dir)
    _write_metrics(paths.audit_dir / "duplicate_metrics.json", duplicates)
    merge = merge_reviews(
        paths.reviews_fact_dir,
        paths.mapping_parquet,
        paths.reviews_enriched_dir,
        unmatched_csv=paths.unmatched_csv,
        unmatched_parquet=paths.unmatched_parquet,
    )
    _write_metrics(paths.audit_dir / "merge_metrics.json", merge)
    validation = validate_outputs(paths, merge_metrics=merge)
    render_report(paths)
    write_validation_notebook(paths)
    return {
        "audit": _serializable(audit),
        "reviews": review.to_dict(),
        "mapping": mapping.to_dict(),
        "duplicates": duplicates.to_dict(),
        "merge": merge.to_dict(),
        "validation": validation.to_dict(),
        "report": str(paths.report_path),
        "notebook": str(paths.notebook_path),
    }


def _run_validation(paths: PipelinePaths, *, force: bool) -> dict[str, Any]:
    duplicate_metrics_path = paths.audit_dir / "duplicate_metrics.json"
    if force:
        for target in [
            paths.audit_dir / "exact_duplicate_groups.parquet",
            paths.audit_dir / "fallback_duplicate_groups.parquet",
            duplicate_metrics_path,
        ]:
            _remove_generated(target)
    if not duplicate_metrics_path.exists():
        duplicates = analyze_review_duplicates(paths.reviews_fact_dir, paths.audit_dir)
        _write_metrics(duplicate_metrics_path, duplicates)
    merge_metrics = _load_merge_metrics(paths.audit_dir / "merge_metrics.json")
    validation = validate_outputs(paths, merge_metrics=merge_metrics)
    render_report(paths)
    write_validation_notebook(paths)
    return validation.to_dict()


def _load_merge_metrics(path: Path) -> MergeMetrics | None:
    if not path.exists():
        return None
    return MergeMetrics(**json.loads(path.read_text(encoding="utf-8")))


def _write_metrics(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.to_dict() if hasattr(value, "to_dict") else _serializable(value)
    _atomic_json(path, payload)


def _serializable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        payload = asdict(value)
        if "review_files" in payload:
            payload["review_files"] = [str(path) for path in payload["review_files"]]
        return payload
    return value


def _prepare_target(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite generated output: {path}")
    if force:
        _remove_generated(path)


def _remove_generated(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _validate_output_root_safety(paths: PipelinePaths) -> None:
    output_root = paths.output_root.resolve()
    project_root = paths.project_root.resolve()
    source_dir = paths.source_dir.resolve()
    if (
        output_root == project_root
        or project_root.is_relative_to(output_root)
        or output_root.is_relative_to(source_dir)
    ):
        raise ValueError(
            f"Unsafe output root {output_root}: it overlaps the project or read-only source tree"
        )


if __name__ == "__main__":
    raise SystemExit(main())
