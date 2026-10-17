from __future__ import annotations

from pathlib import Path

import nbformat

from .config import PipelinePaths


def write_validation_notebook(paths: PipelinePaths) -> Path:
    notebook = nbformat.v4.new_notebook()
    fact_glob = str(paths.reviews_fact_dir / "**" / "*.parquet")
    enriched_glob = str(paths.reviews_enriched_dir / "**" / "*.parquet")
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Steam review data-preparation validation\n"
            "This notebook validates and presents generated outputs. The ETL implementation lives "
            "in the `steam_data_prep` package. Queries aggregate or sample lazily with DuckDB."
        ),
        nbformat.v4.new_code_cell(
            "import json\nfrom pathlib import Path\nimport duckdb\n\n"
            f"FACT = {fact_glob!r}\nENRICHED = {enriched_glob!r}\n"
            f"AUDIT = Path({str(paths.audit_dir)!r})\ncon = duckdb.connect()\n"
            "con.execute(\"SET TimeZone = 'UTC'\")"
        ),
        *_section(
            "Input and output schemas",
            'fact_schema = con.sql(f"DESCRIBE SELECT * FROM read_parquet('
            "'{FACT}', hive_partitioning=true)\")\n"
            'enriched_schema = con.sql(f"DESCRIBE SELECT * FROM read_parquet('
            "'{ENRICHED}', hive_partitioning=true)\")\n"
            "fact_schema, enriched_schema",
        ),
        *_section(
            "Source and final row counts",
            'con.sql(f"SELECT (SELECT count(*) FROM read_parquet('
            "'{FACT}', hive_partitioning=true)) AS fact_rows, "
            "(SELECT count(*) FROM read_parquet("
            "'{ENRICHED}', hive_partitioning=true)) AS enriched_rows\")",
        ),
        *_section(
            "Unique AppID counts",
            'con.sql(f"SELECT count(DISTINCT appid) AS unique_appids '
            "FROM read_parquet('{ENRICHED}', hive_partitioning=true)\")",
        ),
        *_section(
            "Timestamp coverage",
            'con.sql(f"SELECT min(created_at_utc), max(created_at_utc), '
            "min(updated_at_utc), max(updated_at_utc) FROM read_parquet("
            "'{ENRICHED}', hive_partitioning=true)\")",
        ),
        *_section(
            "Missing-value rates",
            "validation = json.loads((AUDIT / 'validation_metrics.json').read_text())\n"
            "validation['missing_value_rates']",
        ),
        *_section(
            "Duplicate diagnostics",
            "json.loads((AUDIT / 'duplicate_metrics.json').read_text())",
        ),
        *_section(
            "Merge match rates",
            "json.loads((AUDIT / 'merge_metrics.json').read_text())",
        ),
        *_section(
            "Unmatched AppIDs",
            f'con.sql("SELECT * FROM read_parquet('
            f'{str(paths.unmatched_parquet)!r}) ORDER BY appid")',
        ),
        *_section(
            "Top genres",
            'con.sql(f"SELECT genre, count(*) AS reviews FROM read_parquet('
            "'{ENRICHED}', hive_partitioning=true), UNNEST(genres) t(genre) "
            'GROUP BY genre ORDER BY reviews DESC LIMIT 20")',
        ),
        *_section(
            "Top publishers",
            'con.sql(f"SELECT publisher, count(*) AS reviews FROM read_parquet('
            "'{ENRICHED}', hive_partitioning=true), UNNEST(publishers) t(publisher) "
            'GROUP BY publisher ORDER BY reviews DESC LIMIT 20")',
        ),
        *_section(
            "Sample enriched review records",
            'con.sql(f"SELECT * FROM read_parquet('
            "'{ENRICHED}', hive_partitioning=true) USING SAMPLE 20 ROWS LIMIT 20\")",
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


def _section(title: str, code: str) -> list:
    return [nbformat.v4.new_markdown_cell(f"## {title}"), nbformat.v4.new_code_cell(code)]
