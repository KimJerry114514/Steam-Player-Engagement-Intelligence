from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from .reviews import _sql_literal

if TYPE_CHECKING:
    from .analysis_readiness import AnalysisPaths


def render_figures(paths: AnalysisPaths) -> list[Path]:
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        _bar_from_table(
            paths,
            "temporal_by_year",
            "review_year",
            "review_count",
            "Reviews by creation year",
            "temporal_reviews_by_year.svg",
        ),
        _bar_from_table(
            paths,
            "match_by_year",
            "review_year",
            "match_rate",
            "Metadata match rate by review year",
            "match_rate_by_year.svg",
            percent=True,
        ),
        _bar_from_table(
            paths,
            "duplicate_classification_summary",
            "classification",
            "candidate_group_count",
            "Candidate-key duplicate groups",
            "duplicate_patterns.svg",
        ),
        _bar_from_table(
            paths,
            "playtime_by_sentiment",
            "voted_up",
            "positive_share",
            "Share with positive additional playtime",
            "playtime_positive_share.svg",
            percent=True,
            label_map={"False": "negative", "True": "positive"},
        ),
        _bar_from_table(
            paths,
            "helpfulness_by_sentiment",
            "voted_up",
            "votes_up_zero_share",
            "Share with zero helpful votes",
            "helpfulness_zero_inflation.svg",
            percent=True,
            label_map={"False": "negative", "True": "positive"},
        ),
    ]
    return figures


def render_analysis_reports(paths: AnalysisPaths, metrics: dict[str, Any]) -> None:
    paths.audit_report.parent.mkdir(parents=True, exist_ok=True)
    paths.target_report.parent.mkdir(parents=True, exist_ok=True)
    tables = _load_report_tables(paths)
    overview = metrics["summary"]["overview"]
    provenance = metrics["provenance"]
    unmatched = metrics["unmatched_catalog_evidence"]
    playtime = tables["playtime_by_sentiment"]
    helpfulness = tables["helpfulness_by_sentiment"]
    helpfulness_extremes = tables["helpfulness_extremes"]
    duplicate_summary = tables["duplicate_classification_summary"]
    candidate_group_count = sum(row["candidate_group_count"] for row in duplicate_summary)
    candidate_instance_count = metrics["analysis_base"]["candidate_duplicate_rows"]
    exact_instance_count = metrics["analysis_base"]["exact_duplicate_rows"]
    duplicate_connection = duckdb.connect()
    try:
        exact_group_count = duplicate_connection.execute(
            f"SELECT sum(exact_subgroup_count) FROM read_parquet("
            f"{_sql_literal(str(paths.duplicate_diagnostics))})"
        ).fetchone()[0]
    finally:
        duplicate_connection.close()
    metadata_timing = tables.get("metadata_catalog_timing", [])
    metadata_timing_text = (
        "Local current-catalog release-date evidence:\n\n"
        + _md_table(metadata_timing)
        + "\n\nThe future-dated maximum is itself a metadata anomaly; only the count of "
        "post-review-window release dates is used to demonstrate that later catalog growth "
        "can inflate mapping size."
        if metadata_timing
        else "The optional current-catalog timing table was unavailable."
    )
    verification_path = paths.audit_dir / "verification_metrics.json"
    verification = (
        json.loads(verification_path.read_text(encoding="utf-8"))
        if verification_path.exists()
        else {"status": "verification not yet recorded"}
    )

    audit = f"""# Steam Review Analysis Readiness Audit

Generated from the immutable Phase 1 canonical outputs. No raw or canonical rows were
modified, deduplicated, or discarded.

## Executive conclusion

The canonical data are suitable for descriptive negative-review analysis and for an
exploratory, time-ordered prediction of **additional play observed by collection**. They do
not support a clean causal or exposure-adjusted measure of post-review engagement because no
defensible common observation date was found. Raw helpfulness counts have the same unresolved
accumulation-time problem and should remain secondary/descriptive outcomes.

## Provenance and observation timing

- Dataset: [{provenance["dataset_url"]}]({provenance["dataset_url"]})
- Author: `{provenance["author"]}`
- Accessed: {provenance["access_date"]}
- License: **{provenance["license"]}**
- Publication date: {provenance["publication_date"]}
- Dataset update date: {provenance["dataset_update_date"]}
- Review collection/observation date: {provenance["review_collection_date"]}
- Common review/playtime snapshot established: **no**

The accessible Kaggle data card identifies the author and describes roughly 15 million
reviews across 8,183 games, but it does not establish when the review/playtime values were
observed. The 11 local CSVs have closely grouped filesystem modification timestamps, and the
Kaggle page exposes a relative update label; neither evidence links those dates to field
measurement. The shards overlap heavily in calendar coverage, which is consistent with a
packaged export, but is not proof of one common observation snapshot. Accordingly,
`review_age_days_at_snapshot`, exposure-day rates, and age-adjusted helpfulness measures were
**not created**.

## Temporal coverage

- Review creation: **{overview["earliest_created_utc"]}** through
  **{overview["latest_created_utc"]}** (UTC)
- Review updates: **{overview["earliest_updated_utc"]}** through
  **{overview["latest_updated_utc"]}** (UTC)
- Reviews: **{_int(overview["review_count"])}**
- Positive: **{_int(overview["positive_reviews"])}**
- Negative: **{_int(overview["negative_reviews"])}**
- Unique review AppIDs: **{_int(overview["unique_appids"])}**
- Updates after the latest review-creation timestamp:
  **{_int(overview["updates_after_latest_creation"])}**

One update timestamp is in 2042 and should be treated as an anomalous source value, not as
evidence of the extraction date. Creation coverage ends partway through April 2021, so 2021
is an incomplete year.

![Reviews by year](../data/processed/analysis/figures/temporal_reviews_by_year.svg)

{_md_table(tables["temporal_by_year"])}

Monthly counts are persisted in
`data/processed/analysis/audit/tables/temporal_by_month.csv`; all 127 observed months from
October 2010 through April 2021 are represented. Source-file time bounds show material
differences in when each shard begins, while nearly every nontrivial shard reaches April 2021:

{_md_table(tables["source_csv_coverage"])}

## Unmatched AppIDs

All **{_int(unmatched["unmatched_appids"])}** unmatched AppIDs were checked against every raw
catalog record using the semantic `steam_appid` field, without the Phase 1 kind filter.

- Present as `steam_appid` in any raw kind: **{_int(unmatched["present_as_steam_appid_any_kind"])}**
- Excluded only because `kind != "steam"`: **{_int(unmatched["excluded_by_kind_filter"])}**
- Coincide only with the unrelated raw internal `id`:
  **{_int(unmatched["internal_id_collisions"])}**

Thus the Phase 1 kind filter explains none of the misses. Internal-ID collisions are explicitly
flagged and are not joins. Plausible explanations are delisting/survivorship, non-game Steam
applications, region/catalog coverage differences, or gaps in the current SteamDB-derived
catalog; local evidence cannot distinguish these mechanisms. Later games can explain why the
current metadata universe is much larger than the 2021 review universe, but they cannot
explain historical reviewed AppIDs being absent now.

{metadata_timing_text}

![Match rate by year](../data/processed/analysis/figures/match_rate_by_year.svg)

Unmatched-row concentration:

{_md_table(tables["unmatched_concentration"])}

Match rate by review year:

{_md_table(tables["match_by_year"])}

Match rate by game review-volume decile:

{_md_table(tables["match_by_review_volume_decile"])}

Positive versus negative matching:

{_md_table(tables["match_by_sentiment"])}

Matched/unmatched distribution comparison:

{_md_table(tables["matched_unmatched_comparison"])}

The 835-AppID diagnostics are in `unmatched_appid_diagnostics.parquet`. The top 50 AppIDs by
review count are listed below and also saved as CSV/Parquet:

{_md_table(tables["top_50_unmatched_appids"], max_columns=10)}

## Duplicate records

The canonical rows remain intact. The analysis base adds full-payload exact-duplicate and
`appid + steamid + unix_timestamp_created` candidate-key counts, flags, and ranks, plus text
and update-variation flags.

- Exact payload duplicates: **{_int(exact_group_count)} groups**,
  **{_int(exact_instance_count)} instances**, and
  **{_int(exact_instance_count - exact_group_count)} redundant copies**.
- Candidate-key duplicates: **{_int(candidate_group_count)} groups**,
  **{_int(candidate_instance_count)} instances**, and
  **{_int(candidate_instance_count - candidate_group_count)} rows beyond one per key**.

![Duplicate patterns](../data/processed/analysis/figures/duplicate_patterns.svg)

{_md_table(tables["duplicate_classification_summary"])}

Classification is conservative: identical payloads are true duplicated records; differing
update timestamps are likely stored versions of one review; same-timestamp text variants and
other payload-only changes remain ambiguous. No same-key pattern provides strong evidence for
distinct legitimate reviews. The recommended analytical sensitivity rule is to retain
candidate rank 1 (latest update, deterministic tie-breakers), while retaining the unfiltered
base as the audit trail and comparing results with/without that selector.

## Post-review playtime

`additional_playtime_minutes = playtime_forever - playtime_at_review` uses the source units as
minutes. It is never clamped. Across the full data there are
**{_int(overview["additional_playtime_nulls"])}** null results,
**{_int(overview["additional_playtime_negative"])}** negative results,
**{_int(overview["additional_playtime_zero"])}** zero results, and
**{_int(overview["additional_playtime_positive"])}** positive results.

![Observed additional play](../data/processed/analysis/figures/playtime_positive_share.svg)

{_md_table(playtime)}

The continuous difference is highly right-skewed and older reviews have had unknown, generally
longer opportunity to accumulate play. It is defensible only as **lifetime playtime difference
observed by collection**, not as a fixed-horizon post-review response. A binary
`additional_playtime_positive` outcome is more robust to the tail but still exposure-biased.
Year/sentiment and game/sentiment tables are saved under the audit tables directory.

## Helpfulness outcomes

There are no nulls in the three helpfulness fields in the canonical data, but `votes_up`,
`votes_funny`, and `weighted_vote_score` are zero-inflated and heavy-tailed.

![Helpfulness zero inflation](../data/processed/analysis/figures/helpfulness_zero_inflation.svg)

{_md_table(helpfulness)}

Extreme-value checks:

{_md_table(helpfulness_extremes)}

The value 4,294,967,295 (`2^32 - 1`) occurs in `votes_funny` and is sentinel/overflow-like,
not a credible literal funny-vote count. It must be flagged as anomalous in any helpfulness
analysis rather than allowed to dominate means.

The strong relationship with review year is not separable from age-at-observation because the
collection snapshot is unknown. Raw counts are not a defensible primary cross-time target.
Safer secondary choices are within-month/quarter ranking, binary receipt of any helpful vote,
or weighted score within narrow creation cohorts, each labeled as descriptive rather than
fully age-corrected.

## Non-destructive analysis base

The full base and negative-review view preserve unmatched rows, duplicate rows, negative
playtime differences, and missing metadata. Added fields include metadata matching, duplicate
counts/ranks, UTC month, raw additional playtime, playtime validity flags, game age,
review-before-release, negative-review, and timestamp-anomaly flags. There are
**{_int(overview["reviews_before_release"])}** matched rows whose review date precedes the
current metadata release date; this is a metadata/version warning, not a deletion rule.

## Chronological split options

{_md_table(tables["chronological_split_summary"])}

- **Recommended for methodological approval:** Design B for a first predictive benchmark,
  limited to 2019-2020. It narrows era differences and avoids the partial 2021 tail, but it
  does not solve unknown review-age exposure.
- **Broader alternative:** Design A for maximum training history and a forward 2020-April 2021
  holdout.
- **Robustness alternative:** extend Design C into repeated annual expanding folds before a
  final untouched recent holdout.

No split is silently selected in the output data.

## Reproducible audit tables

Every table shown here is available as both Parquet and CSV under
`data/processed/analysis/audit/tables`. The notebook reads those compact tables and the
partitioned analysis base with DuckDB.

## Verification

{_md_table([verification])}
"""

    target = f"""# Target Framework Recommendation

## Decision

Use **any observed additional play** (`additional_playtime_positive`) as the initial primary
predictive outcome, and describe it exactly as: *whether lifetime playtime at collection
exceeded playtime recorded at review time*. Do not call it a fixed-window post-review
engagement rate, a causal effect, or exposure-adjusted retention.

The recommended initial population is negative reviews created during 2019-2020. For models
that require game metadata, use metadata-matched rows as the primary estimation population,
report the excluded unmatched share, and run a feature-light robustness analysis that retains
unmatched reviews. Use candidate-key duplicate rank 1 in the primary analytical slice and the
full canonical-preserving base as a sensitivity check.

## Why the continuous target is not primary

The full-data raw difference contains {_int(overview["additional_playtime_negative"])}
negative values and a very long right tail. More importantly, no defensible snapshot date was
found. A 2011 review and a 2021 review therefore have unequal, unknown exposure windows. The
continuous difference is valid as an observed source-field difference but not as a comparable
fixed-horizon behavioral response.

## Why the binary target is only conditionally defensible

Binary any-subsequent-play reduces sensitivity to extreme amounts but does not eliminate the
accumulation advantage of older reviews. It is acceptable for exploratory prediction when the
language, chronological split, cohort restriction, and sensitivity analyses are explicit.
It is not appropriate for causal claims.

## Helpfulness framework

Do not use raw helpful-vote counts as the primary target. Older reviews have had more unknown
time to collect votes, and zero inflation is substantial:

{_md_table(helpfulness)}

Defensible secondary analyses, in order of preference, are:

1. within-review-month or within-review-quarter helpfulness ranks;
2. weighted vote score within narrow creation cohorts;
3. binary receipt of any helpful vote within a narrow cohort;
4. raw counts only as descriptive outcomes with review-time stratification.

None is fully age-corrected without a verified observation date.

The `votes_funny` maximum is 4,294,967,295 (`2^32 - 1`) in 493 rows, a
sentinel/overflow-like anomaly that must be flagged or excluded under an approved rule before
using that field.

## Population and duplicate rules

- Preserve all rows in the analysis base.
- Primary engagement slice: negative, 2019-2020, valid nonnegative playtime inputs,
  candidate-key rank 1.
- Do not clamp negative differences; exclude or model them only through a documented
  sensitivity rule.
- Primary metadata model: matched rows, with unmatched diagnostics and a feature-light
  robustness population.
- Report full-base versus rank-1 results because some candidate groups are updated versions
  and others are ambiguous.

## Time-design recommendation

{_md_table(tables["chronological_split_summary"])}

Approve Design B for the first benchmark if the priority is comparability. Approve Design A
if the priority is historical breadth. In either case, add expanding annual folds as a
stability check and keep the latest chosen test interval untouched.

## Gate before modeling

Modeling should wait for methodological approval of: (1) the outcome wording, (2) treatment of
the {_int(overview["additional_playtime_negative"])} negative differences, (3) matched-only
versus feature-light population, (4) rank-1 duplicate sensitivity rule, and (5) the final
chronological design.
"""
    paths.audit_report.write_text(audit, encoding="utf-8")
    paths.target_report.write_text(target, encoding="utf-8")


def _load_report_tables(paths: AnalysisPaths) -> dict[str, list[dict[str, Any]]]:
    names = [
        "temporal_by_year",
        "source_csv_coverage",
        "match_by_year",
        "match_by_sentiment",
        "match_by_review_volume_decile",
        "matched_unmatched_comparison",
        "top_50_unmatched_appids",
        "unmatched_concentration",
        "duplicate_classification_summary",
        "playtime_by_sentiment",
        "helpfulness_by_sentiment",
        "helpfulness_extremes",
        "chronological_split_summary",
    ]
    optional_names = ["metadata_catalog_timing"]
    connection = duckdb.connect()
    try:
        result = {
            name: _query_dicts(
                connection,
                f"SELECT * FROM read_parquet("
                f"{_sql_literal(str(paths.tables_dir / f'{name}.parquet'))})",
            )
            for name in names
        }
        for name in optional_names:
            path = paths.tables_dir / f"{name}.parquet"
            if path.exists():
                result[name] = _query_dicts(
                    connection,
                    f"SELECT * FROM read_parquet({_sql_literal(str(path))})",
                )
        return result
    finally:
        connection.close()


def _bar_from_table(
    paths: AnalysisPaths,
    table: str,
    label_column: str,
    value_column: str,
    title: str,
    filename: str,
    *,
    percent: bool = False,
    label_map: dict[str, str] | None = None,
) -> Path:
    path = paths.tables_dir / f"{table}.parquet"
    connection = duckdb.connect()
    try:
        cursor = connection.execute(
            f"SELECT {label_column}, {value_column} FROM read_parquet("
            f"{_sql_literal(str(path))}) ORDER BY 1"
        )
        values = [(str(label), float(value or 0)) for label, value in cursor.fetchall()]
    finally:
        connection.close()
    if label_map:
        values = [(label_map.get(label, label), value) for label, value in values]
    output = paths.figures_dir / filename
    output.write_text(_bar_svg(values, title, percent=percent), encoding="utf-8")
    return output


def _bar_svg(values: list[tuple[str, float]], title: str, *, percent: bool) -> str:
    width, height = 960, 440
    left, right, top, bottom = 75, 30, 55, 105
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max((value for _, value in values), default=1) or 1
    count = max(len(values), 1)
    slot = plot_width / count
    bar_width = max(3.0, slot * 0.72)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" '
        'font-family="Arial" font-size="20" font-weight="600">'
        f"{html.escape(title)}</text>",
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#555"/>',
    ]
    for index, (label, value) in enumerate(values):
        bar_height = 0 if maximum == 0 else value / maximum * plot_height
        x = left + index * slot + (slot - bar_width) / 2
        y = top + plot_height - bar_height
        display = f"{value:.1%}" if percent else _compact_number(value)
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
            f'height="{bar_height:.2f}" fill="#3b82a0"><title>'
            f"{html.escape(label)}: {html.escape(display)}</title></rect>"
        )
        rotate = len(values) > 12 or max((len(item[0]) for item in values), default=0) > 10
        if rotate:
            parts.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{top + plot_height + 14}" '
                'text-anchor="end" transform="rotate(-55 '
                f'{x + bar_width / 2:.2f} {top + plot_height + 14})" '
                f'font-family="Arial" font-size="10">{html.escape(label)}</text>'
            )
        else:
            parts.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{top + plot_height + 18}" '
                f'text-anchor="middle" font-family="Arial" font-size="11">'
                f"{html.escape(label)}</text>"
            )
    for tick in range(6):
        fraction = tick / 5
        y = top + plot_height - fraction * plot_height
        tick_value = maximum * fraction
        display = f"{tick_value:.0%}" if percent else _compact_number(tick_value)
        parts.append(
            f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left + plot_width}" '
            f'y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="10">{html.escape(display)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _query_dicts(connection: duckdb.DuckDBPyConnection, query: str) -> list[dict[str, Any]]:
    cursor = connection.execute(query)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _md_table(rows: list[dict[str, Any]], *, max_columns: int | None = None) -> str:
    if not rows:
        return "_No rows._"
    columns = list(rows[0])
    if max_columns is not None:
        columns = columns[:max_columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [_format_value(row.get(column)) for column in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if 0 <= value <= 1:
            return f"{value:.3%}"
        return f"{value:,.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value).replace("|", "\\|")
    return str(value).replace("|", "\\|").replace("\n", " ")


def _int(value: Any) -> str:
    return f"{int(value):,}"


def _compact_number(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"
