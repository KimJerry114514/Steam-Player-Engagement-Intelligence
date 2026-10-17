from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import nbformat

if TYPE_CHECKING:
    from .analysis_readiness import AnalysisPaths


def write_analysis_readiness_notebook(paths: AnalysisPaths) -> Path:
    notebook = nbformat.v4.new_notebook()
    base = str(paths.analysis_base_dir / "**" / "*.parquet")
    negative = str(paths.negative_reviews_dir / "**" / "*.parquet")
    tables = str(paths.tables_dir)
    figures_relative = "../data/processed/analysis/figures"
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Phase 2: Steam review analysis readiness\n"
            "This notebook presents the persisted audit outputs without modifying canonical "
            "data or training models. DuckDB queries are lazy and use compact audit tables "
            "where possible."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport duckdb\n\n"
            f"BASE = {base!r}\nNEGATIVE = {negative!r}\nTABLES = Path({tables!r})\n"
            "con = duckdb.connect()\ncon.execute(\"SET TimeZone = 'UTC'\")"
        ),
        *_section(
            "Provenance boundary",
            "No defensible review/playtime observation snapshot was identified. File "
            "modification and Kaggle update labels are not treated as observation dates. "
            "Exposure-adjusted measures are intentionally absent.",
        ),
        *_visual_section(
            "Temporal coverage",
            f"{figures_relative}/temporal_reviews_by_year.svg",
            "temporal_by_year",
        ),
        nbformat.v4.new_code_cell(
            'con.sql(f"SELECT min(created_at_utc), max(created_at_utc), '
            "min(updated_at_utc), max(updated_at_utc) "
            "FROM read_parquet('{BASE}', hive_partitioning=true)\")"
        ),
        nbformat.v4.new_code_cell(
            "con.sql(f\"SELECT * FROM read_parquet('"
            "{TABLES / 'temporal_by_month.parquet'}') ORDER BY review_month\")"
        ),
        *_visual_section(
            "Matching bias",
            f"{figures_relative}/match_rate_by_year.svg",
            "matched_unmatched_comparison",
        ),
        nbformat.v4.new_code_cell(
            "con.sql(f\"SELECT * FROM read_parquet('"
            "{TABLES / 'top_50_unmatched_appids.parquet'}') "
            'ORDER BY review_count DESC LIMIT 50")'
        ),
        *_visual_section(
            "Duplicate patterns",
            f"{figures_relative}/duplicate_patterns.svg",
            "duplicate_classification_summary",
        ),
        nbformat.v4.new_code_cell(
            'con.sql(f"SELECT candidate_key_duplicate_count, '
            "candidate_review_text_varies, candidate_update_timestamp_varies, count(*) "
            "FROM read_parquet('{BASE}', hive_partitioning=true) "
            'WHERE candidate_key_duplicate_flag GROUP BY ALL ORDER BY 1,2,3")'
        ),
        *_visual_section(
            "Playtime target behavior",
            f"{figures_relative}/playtime_positive_share.svg",
            "playtime_by_sentiment",
        ),
        nbformat.v4.new_code_cell(
            "con.sql(f\"SELECT * FROM read_parquet('"
            "{TABLES / 'playtime_by_year_sentiment.parquet'}') "
            'ORDER BY review_year, voted_up")'
        ),
        *_visual_section(
            "Helpfulness accumulation",
            f"{figures_relative}/helpfulness_zero_inflation.svg",
            "helpfulness_by_sentiment",
        ),
        nbformat.v4.new_code_cell(
            "con.sql(f\"SELECT * FROM read_parquet('"
            "{TABLES / 'helpfulness_by_year.parquet'}') ORDER BY review_year\")"
        ),
        *_section(
            "Chronological split candidates",
            "The following alternatives are recommendations for approval, not a silently "
            "selected modeling split.",
        ),
        nbformat.v4.new_code_cell(
            "con.sql(f\"SELECT * FROM read_parquet('"
            "{TABLES / 'chronological_split_summary.parquet'}') ORDER BY design, split\")"
        ),
        *_section(
            "Analysis-base validation",
            "The negative-review dataset is a filtered convenience view; the full base keeps "
            "unmatched rows, duplicates, negative playtime differences, and missing metadata.",
        ),
        nbformat.v4.new_code_cell(
            "con.sql(f\"SELECT (SELECT count(*) FROM read_parquet('{BASE}', "
            "hive_partitioning=true)) AS full_rows, "
            "(SELECT count(*) FROM read_parquet('{NEGATIVE}', "
            'hive_partitioning=true)) AS negative_rows")'
        ),
    ]
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    paths.notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, paths.notebook_path)
    return paths.notebook_path


def _section(title: str, body: str) -> list:
    return [nbformat.v4.new_markdown_cell(f"## {title}\n\n{body}")]


def _visual_section(title: str, figure: str, table: str) -> list:
    return [
        nbformat.v4.new_markdown_cell(f"## {title}\n\n![{title}]({figure})"),
        nbformat.v4.new_code_cell(
            f"con.sql(f\"SELECT * FROM read_parquet('{{TABLES / '{table}.parquet'}}')\")"
        ),
    ]
