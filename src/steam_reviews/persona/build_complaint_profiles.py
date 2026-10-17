"""Build full-history complaint profiles from extended P55 segments and frozen NLP flags."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import textwrap

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'team18-persona-matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'team18-persona-cache'))
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from steam_reviews.paths import ROOT
from .paths import PersonaPaths

SEGMENTS = ['Hardcore', 'Dropout', 'New Blood', 'Nobody']
SCOPES = ['all_history_2010_2023', 'history_2010_2021', 'validation_2022', 'test_2023']
SCOPE_LABELS = {'all_history_2010_2023': 'All historical negative reviews, 2010–2023', 'history_2010_2021': '2010–2021 history (includes the training cohort)', 'validation_2022': 'All 2022 negative reviews', 'test_2023': 'All 2023 negative reviews'}
COLORS = dict(zip(SEGMENTS, ['#2166AC', '#C64D33', '#28877D', '#7B61A8']))
DOMAIN_LABELS = {'technical': 'Technical', 'gameplay': 'Gameplay', 'progression': 'Progression', 'content': 'Content', 'social': 'Social', 'user experience': 'User experience', 'business': 'Business', 'service': 'Service'}


def sql_id(s):
    return '"' + s.replace('"', '""') + '"'


def sql_string(s):
    return "'" + str(s).replace("'", "''") + "'"


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def markdown_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines += ['| ' + ' | '.join(str(x).replace('|', '/') for x in row) + ' |' for row in rows]
    return '\n'.join(lines)


def aggregate(project_root, scratch, out, rebuild=False):
    from .build_historical_segments import ensure_segments
    segment_path, history_audit, history_sources = ensure_segments(project_root, out, rebuild=rebuild)
    final = project_root
    sources = {
        **history_sources,
        'segments': segment_path,
        'historical_audit': out.audit,
        'nlp_flags': final / 'data/processed/nlp/nlp_factor_scores_2023.parquet',
        'factor_dictionary': final / 'reports/tables/nlp/nlp_35_factors_dictionary.csv',
        'domain_mapping': final / 'reports/tables/nlp/nlp_domain_mapping.csv',
        'rules': final / 'results/segmentation/selected_segmentation_rules.json',
        'segment_summary': final / 'results/segmentation/segment_summary.json',
    }
    for p in sources.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    factors = pd.read_csv(sources['factor_dictionary'])
    domains = pd.read_csv(sources['domain_mapping'])
    assert len(factors) == factors.flag_column.nunique() == factors.factor_id.nunique() == 35
    assert len(domains) == 8 and domains.domain.nunique() == 8
    domain_flags = {row.domain: [x.strip() for x in row.flag_columns.split(';')] for row in domains.itertuples()}
    flattened = [f for flags in domain_flags.values() for f in flags]
    assert len(flattened) == len(set(flattened)) == 35
    assert set(flattened) == set(factors.flag_column)
    for row in factors.itertuples():
        assert row.flag_column in domain_flags[row.domain]
    rules = json.loads(sources['rules'].read_text())
    assert rules['prior_percentile'] == rules['score_percentile'] == 55
    assert rules['prior_cutoff_hours'] == 11.78
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute("SET memory_limit='2GB'")
    con.execute('SET threads=4')
    con.execute(f'SET temp_directory={sql_string(scratch)}')
    con.execute(f'CREATE TEMP TABLE segments AS SELECT * FROM read_parquet({sql_string(sources["segments"])})')
    n, unique_n, invalid_keys = con.execute('SELECT count(*), count(DISTINCT (appid, recommendationid)), count(*) FILTER (WHERE appid IS NULL OR recommendationid IS NULL) FROM segments').fetchone()
    assert n == unique_n == 16412862 and invalid_keys == 0
    definition_errors = con.execute('''SELECT count(*) FROM segments WHERE
        segment IS NULL OR split NOT IN ('train','val','test') OR split IS NULL OR
        hours_at_review IS NULL OR prob_raw IS NULL OR raw_probability_cutoff IS NULL OR
        prior_hours_cutoff IS NULL OR prior_hours_cutoff != 11.78 OR
        high_prior IS DISTINCT FROM (hours_at_review > 11.78) OR
        high_predicted_later IS DISTINCT FROM (prob_raw >= raw_probability_cutoff) OR
        segment IS DISTINCT FROM CASE WHEN hours_at_review > 11.78 AND prob_raw >= raw_probability_cutoff THEN 'Hardcore'
            WHEN hours_at_review > 11.78 THEN 'Dropout'
            WHEN prob_raw >= raw_probability_cutoff THEN 'New Blood' ELSE 'Nobody' END OR
        (split = 'train' AND NOT (review_created_utc >= TIMESTAMPTZ '2010-10-15' AND review_created_utc < TIMESTAMPTZ '2022-01-01')) OR
        (split = 'val' AND NOT (review_created_utc >= TIMESTAMPTZ '2022-01-01' AND review_created_utc < TIMESTAMPTZ '2023-01-01')) OR
        (split = 'test' AND review_created_utc < TIMESTAMPTZ '2023-01-01')''').fetchone()[0]
    assert definition_errors == 0
    projected = ', '.join('p.' + sql_id(c) for c in flattened + ['n_factors_flagged'])
    print(f'Joining all {n:,} historical records to the frozen NLP flags...', flush=True)
    con.execute(f'''CREATE TEMP TABLE joined AS SELECT s.appid, s.recommendationid, s.split,
        CAST(s.segment AS VARCHAR) AS segment, {projected}, p.appid IS NOT NULL AS has_nlp
        FROM segments s LEFT JOIN read_parquet({sql_string(sources['nlp_flags'])}) p
        ON s.appid = p.appid AND s.recommendationid = p.recommendationid''')
    joined_n, unmatched = con.execute('SELECT count(*), count(*) FILTER (WHERE NOT has_nlp) FROM joined').fetchone()
    assert joined_n == n and unmatched == 0, (joined_n, unmatched)
    invalid_expr = ' OR '.join(f'{sql_id(c)} IS NULL OR {sql_id(c)} NOT IN (0,1)' for c in flattened)
    sum_expr = ' + '.join(f'CAST({sql_id(c)} AS INTEGER)' for c in flattened)
    invalid_flags, inconsistent_counts = con.execute(f'''SELECT count(*) FILTER (WHERE {invalid_expr}),
        count(*) FILTER (WHERE n_factors_flagged IS NULL OR n_factors_flagged != ({sum_expr})) FROM joined''').fetchone()
    assert invalid_flags == inconsistent_counts == 0
    # Domain prevalence is the union of its factors, never a sum of factor rates.
    expressions = [f'SUM({sql_id(c)})::BIGINT AS {sql_id(c)}' for c in flattened]
    expressions += [f'SUM(CASE WHEN ' + ' OR '.join(f'{sql_id(c)} = 1' for c in flags) + f' THEN 1 ELSE 0 END)::BIGINT AS {sql_id("domain_" + d)}' for d, flags in domain_flags.items()]
    expressions += [f'SUM({sum_expr})::BIGINT AS factor_mentions', f'SUM(CASE WHEN ({sum_expr}) > 0 THEN 1 ELSE 0 END)::BIGINT AS any_complaint_reviews']
    grouped = con.execute('SELECT split, segment, count(*)::BIGINT AS total_reviews, ' + ', '.join(expressions) + ' FROM joined GROUP BY split, segment').df()
    actual_counts = grouped.groupby('split').total_reviews.sum().to_dict()
    expected_counts = {s: sum(r['rows'] for r in history_audit['period_and_model_cohort_counts'] if r['split']==s) for s in ['train','val','test']}
    assert actual_counts == expected_counts
    combined = grouped.drop(columns='split').groupby('segment', sort=False).sum(numeric_only=True).reset_index()
    combined.insert(0, 'split', 'combined')
    grouped = pd.concat([grouped, combined], ignore_index=True)
    grouped['scope'] = grouped['split'].map({'train': SCOPES[1], 'val': SCOPES[2], 'test': SCOPES[3], 'combined': SCOPES[0]})
    checks = {
        'source_segment_rows': n, 'unique_segment_join_keys': unique_n,
        'joined_rows': joined_n, 'unmatched_nlp_records': unmatched,
        'null_or_nonbinary_factor_records': invalid_flags,
        'inconsistent_stored_factor_counts': inconsistent_counts,
        'segment_definition_errors': definition_errors,
        'factor_count': 35, 'domain_count': 8,
        'domain_mapping_is_complete_and_disjoint': True,
        'segment_counts_match_full_history': True,
        'archived_4937916_assignments_preserved': history_audit['archived_assignment_mismatches']==0,
        'full_history_rows': history_audit['rows'],
        'records_without_binary_d30_included': sum(r['rows'] for r in history_audit['period_and_model_cohort_counts'] if r['model_cohort']=='unlabeled'),
        'date_range_utc': history_audit['date_range_utc'],
        'raw_score_cutoffs': history_audit['raw_score_cutoffs'],
        'period_and_model_cohort_counts': history_audit['period_and_model_cohort_counts'],
    }
    con.close()
    return grouped, factors, domains, domain_flags, sources, checks


def build_tables(grouped, factors, domains, domain_flags):
    rows, baseline_rows = [], []
    metadata = []
    for row in domains.itertuples():
        metadata.append(('domain', row.domain, row.domain, DOMAIN_LABELS[row.domain], 'domain_' + row.domain))
    for row in factors.itertuples():
        metadata.append(('factor', row.factor_id, row.domain, row.label_en, row.flag_column))
    for scope in SCOPES:
        g = grouped.loc[grouped.scope.eq(scope)].set_index('segment')
        assert set(g.index) == set(SEGMENTS)
        denominator = int(g.total_reviews.sum())
        for kind, key, domain, label, column in metadata:
            overall_count = int(g[column].sum())
            overall = overall_count / denominator
            baseline_rows.append(dict(scope=scope, metric_type=kind, metric_id=key, domain=domain, label=label, total_reviews=denominator, flagged_reviews=overall_count, prevalence=overall, prevalence_pct=100 * overall))
            for segment in SEGMENTS:
                total = int(g.loc[segment, 'total_reviews'])
                count = int(g.loc[segment, column])
                rate = count / total
                rows.append(dict(scope=scope, segment=segment, metric_type=kind, metric_id=key, domain=domain, label=label, total_reviews=total, flagged_reviews=count, prevalence=rate, prevalence_pct=100 * rate, overall_prevalence_pct=100 * overall, difference_vs_overall_pp=100 * (rate - overall), lift_vs_overall=rate / overall if overall else np.nan))
    all_rates = pd.DataFrame(rows)
    domain_rates = all_rates.loc[all_rates.metric_type.eq('domain')].drop(columns='metric_type').reset_index(drop=True)
    factor_rates = all_rates.loc[all_rates.metric_type.eq('factor')].drop(columns='metric_type').reset_index(drop=True)
    factor_rates = factor_rates.merge(factors[['factor_id', 'flag_column', 'test_precision', 'test_recall', 'test_f1', 'n_test_positives', 'caution']], left_on='metric_id', right_on='factor_id', validate='many_to_one').drop(columns='factor_id')
    ranked = factor_rates.assign(scope_order=factor_rates.scope.map(dict(zip(SCOPES, range(len(SCOPES))))), segment_order=factor_rates.segment.map(dict(zip(SEGMENTS, range(4))))).sort_values(['scope_order', 'segment_order', 'prevalence', 'metric_id'], ascending=[True, True, False, True]).drop(columns=['scope_order','segment_order']).copy()
    ranked['rank_within_segment'] = ranked.groupby(['scope', 'segment']).cumcount() + 1
    top5 = ranked.loc[ranked.rank_within_segment.le(5)].copy()
    summary = grouped[['scope', 'segment', 'total_reviews', 'factor_mentions', 'any_complaint_reviews']].copy()
    summary['scope_order'] = summary.scope.map(dict(zip(SCOPES, range(len(SCOPES)))))
    summary['segment_order'] = summary.segment.map(dict(zip(SEGMENTS, range(4))))
    summary = summary.sort_values(['scope_order','segment_order']).drop(columns=['scope_order','segment_order'])
    summary['share_within_scope'] = summary.total_reviews / summary.groupby('scope').total_reviews.transform('sum')
    summary['any_complaint_prevalence'] = summary.any_complaint_reviews / summary.total_reviews
    summary['mean_factors_flagged'] = summary.factor_mentions / summary.total_reviews
    return domain_rates, factor_rates, top5, summary, pd.DataFrame(baseline_rows)


def save_figure(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_heatmap(table, summary, domain_order, scope, kind, out, difference=False, vmax=None):
    data = table.loc[table.scope.eq(scope)].copy()
    if kind == 'domain':
        order = domain_order
        labels = [DOMAIN_LABELS[d] for d in order]
    else:
        baseline = table.loc[table.scope.eq(SCOPES[0])].drop_duplicates('metric_id').set_index('metric_id').overall_prevalence_pct
        order = []
        for domain in domain_order:
            ids = data.loc[data.domain.eq(domain), 'metric_id'].unique()
            order.extend(sorted(ids, key=lambda k: (-baseline[k], k)))
        meta = data.drop_duplicates('metric_id').set_index('metric_id')
        labels = [f'{DOMAIN_LABELS[meta.loc[k,"domain"]]} / {meta.loc[k,"label"]}' for k in order]
    value = 'difference_vs_overall_pp' if difference else 'prevalence_pct'
    matrix = data.pivot(index='metric_id', columns='segment', values=value).loc[order, SEGMENTS]
    counts = summary.loc[summary.scope.eq(scope)].set_index('segment').total_reviews
    fig, ax = plt.subplots(figsize=(11 if kind == 'domain' else 13, 6.4 if kind == 'domain' else 16))
    cbar_label = 'Difference from overall prevalence (percentage points)' if difference else 'Reviews with complaint (%)'
    args = dict(cmap='RdBu_r' if difference else 'YlGnBu', annot=True, fmt='+.1f' if difference else '.1f', linewidths=.6, linecolor='#F1F4F8', annot_kws={'size':10}, cbar_kws={'label':cbar_label, 'shrink':.7})
    if difference:
        lim = max(1, float(np.abs(matrix.to_numpy()).max()))
        args.update(vmin=-lim, vmax=lim, center=0)
    else:
        args.update(vmin=0, vmax=vmax)
    sns.heatmap(matrix, ax=ax, **args)
    ax.set_yticklabels(labels, rotation=0, fontsize=10)
    ax.set_xticklabels([f'{s}\nn = {counts[s]:,}' for s in SEGMENTS], rotation=0, fontsize=10)
    ax.set_xlabel('');ax.set_ylabel('')
    title = f'{"Eight complaint domains" if kind == "domain" else "35 complaint factors"} by segment'
    if difference:title += ': difference from overall'
    ax.set_title(title + '\n' + SCOPE_LABELS[scope], loc='left', fontsize=16, fontweight='bold', pad=20)
    if kind == 'factor':
        cumulative = 0
        for d in domain_order[:-1]:
            cumulative += int(data.loc[data.domain.eq(d), 'metric_id'].nunique())
            ax.axhline(cumulative, color='#475569', linewidth=1.2)
    fig.text(.02, .015, 'Each cell uses all records in its segment as the denominator. Domain = at least one factor flagged. Complaints can overlap.', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0,.04,1,1))
    save_figure(fig, out)


def plot_top5(top5, summary, scope, out, xmax):
    data = top5.loc[top5.scope.eq(scope)]
    counts = summary.loc[summary.scope.eq(scope)].set_index('segment').total_reviews
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5))
    for ax, segment in zip(axes.flat, SEGMENTS):
        part = data.loc[data.segment.eq(segment)].sort_values('rank_within_segment')
        y = np.arange(5)
        ax.barh(y, part.prevalence_pct, color=COLORS[segment], height=.62)
        ax.set_yticks(y, [textwrap.fill(label, 25) for label in part.label], fontsize=11)
        ax.invert_yaxis();ax.set_xlim(0,xmax)
        for i,row in enumerate(part.itertuples()):
            ax.text(row.prevalence_pct + .5, i, f'{row.prevalence_pct:.1f}%', va='center', fontsize=11)
        ax.set_title(f'{segment}  |  n = {counts[segment]:,}', loc='left', fontsize=14, fontweight='bold', color=COLORS[segment], pad=14)
        ax.set_xlabel('Reviews with complaint (%)', fontsize=10)
        ax.xaxis.grid(True, alpha=.2);ax.set_axisbelow(True)
        for edge in ['top','right','left']:ax.spines[edge].set_visible(False)
    fig.suptitle('Top 5 complaint factors in each segment\n' + SCOPE_LABELS[scope], x=.03, ha='left', fontsize=20, fontweight='bold')
    fig.text(.03,.015,'Ranked by within-segment prevalence of frozen binary NLP flags. Shared x-axis scale. Multiple complaints may occur in one review.',fontsize=10,color='#475569')
    fig.tight_layout(rect=(0,.04,1,.91),h_pad=3.5,w_pad=3)
    save_figure(fig, out)


def write_report(out, domain_rates, factor_rates, top5, summary, checks):
    scope = SCOPES[0]
    combined = summary.loc[summary.scope.eq(scope)].set_index('segment')
    report = ['# Complaint profiles of the four P55 review segments', '',
        f'The main analysis covers all {checks["full_history_rows"]:,} cleaned historical negative-review records, from {checks["date_range_utc"][0]} to {checks["date_range_utc"][1]}. No training-period records are excluded. The {checks["records_without_binary_d30_included"]:,} records without a binary D30 outcome are also included because segmentation requires model predictions, not observed outcome labels. These are review-level player-game records, not unique players. Separate 2010–2021, 2022, and 2023 summaries show period differences.', '',
        '## Segmentation and complaint definitions', '',
        'The frozen LightGBM model and the existing P55/P55 method are extended without retraining or optimizing a new rule. Higher prior engagement means hours at review > 11.78. Higher predicted later engagement means raw prediction >= the period-specific cutoff. The original 4,937,916 validation/test scores, cutoffs, and four-class assignments are preserved exactly. The remaining 11,474,946 records are scored with the same fitted model and its saved 96-feature preprocessing contract.', '',
        'The added 2010–2021 score cutoff is the 55th percentile of predictions for the original 10,763,783 eligible training records. The frozen 2022 and 2023 cutoffs retain their original eligible validation/test reference cohorts. Unlabeled records receive the cutoff for their review period; they are included in the complaint denominators but do not change these reference cohorts. These are relative score thresholds, so high-score shares in the expanded periods need not equal exactly 45%. Calibration is not refitted; raw probabilities suffice for the original ranking-based segmentation.', '',
        markdown_table(['Review period','Raw-score cutoff','Prior-hours cutoff'],[[{'train':'2010–2021','val':'2022','test':'2023'}[s],f'{checks["raw_score_cutoffs"][s]:.12f}','11.78'] for s in ['train','val','test']]), '',
        'Hardcore is high/high, Dropout is high/low, New Blood is low/high, and Nobody is low/low. These labels describe model-based profiles and do not establish actual churn or customer value. Training-cohort predictions are in sample, not out-of-fold or independent performance evidence. The all-history figures are retrospective descriptive profiles.', '',
        'The expanded segmentation table is joined one-to-one to the NLP export using appid and recommendationid. Existing binary flag_ columns are used as supplied. A domain is present if any of its mapped factors is flagged. Each prevalence denominator includes every record in the relevant segment, including records with zero flagged factors. Factors and domains overlap, so percentages do not sum to 100%. Domains contain unequal numbers of retained factors, from one to seven; domain prevalence measures the union of those factors rather than standardized complaint severity.', '',
        '## Coverage by period and original model eligibility', '',
        markdown_table(['Review period','Original model cohort','Records'],[[{'train':'2010–2021','val':'2022','test':'2023'}[r['split']],r['model_cohort'],f'{r["rows"]:,}'] for r in checks['period_and_model_cohort_counts']]), '',
        'The model_cohort field is train, val, or test only for records with a binary D30 label in the original time windows; otherwise it is unlabeled. Period-specific figure folders include both eligible and unlabeled records from that period. The split field identifies review period and does not itself establish model eligibility.', '',
        '## Cohort overview', '',
        markdown_table(['Segment','Records','Share','Any complaint flagged','Mean flagged factors'],[[s,f'{int(combined.loc[s,"total_reviews"]):,}',f'{100*combined.loc[s,"share_within_scope"]:.2f}%',f'{100*combined.loc[s,"any_complaint_prevalence"]:.2f}%',f'{combined.loc[s,"mean_factors_flagged"]:.2f}'] for s in SEGMENTS]), '',
        '## Eight complaint domains', '',
        '![Domain prevalence](../figures/domain_prevalence_heatmap.png)', '',
        '![Domain differences from overall](../figures/domain_difference_heatmap.png)', '',
        'Differences use the overall review-weighted prevalence in the same scope as the reference. A positive difference means the complaint is more common in that segment than in the overall cohort. It is a percentage-point difference, not a relative percentage change.', '',
        '## Top 5 factors by within-segment prevalence', '',
        '![Top five complaint factors](../figures/top5_complaints_by_segment.png)', '']
    for segment in SEGMENTS:
        part=top5.loc[top5.scope.eq(scope)&top5.segment.eq(segment)].sort_values('rank_within_segment')
        report += [f'### {segment}', '', markdown_table(['Rank','Complaint','Domain','Flagged records','Prevalence','Difference vs overall'],[[row.rank_within_segment,row.label,DOMAIN_LABELS[row.domain],f'{row.flagged_reviews:,}',f'{row.prevalence_pct:.2f}%',f'{row.difference_vs_overall_pp:+.2f} pp'] for row in part.itertuples()]), '']
        d=domain_rates.loc[domain_rates.scope.eq(scope)&domain_rates.segment.eq(segment)].sort_values(['prevalence','metric_id'],ascending=[False,True]).iloc[0]
        e=factor_rates.loc[factor_rates.scope.eq(scope)&factor_rates.segment.eq(segment)].sort_values(['difference_vs_overall_pp','metric_id'],ascending=[False,True]).iloc[0]
        report += [f'The most prevalent domain is {d["label"]} ({d.prevalence_pct:.2f}%). The largest positive factor difference from the overall cohort is {e["label"]} ({e.difference_vs_overall_pp:+.2f} percentage points). The latter is a descriptive contrast, not a causal effect or an additional Top 5 ranking.', '']
    report += ['## All 35 factors', '', '![Factor prevalence](../figures/factor_prevalence_heatmap.png)', '', '![Factor differences from overall](../figures/factor_difference_heatmap.png)', '',
        '## Historical period comparisons', '',
        'The main full-history profile weights every record equally. It is not an unweighted average of period rates. Period-specific tables use each period\'s own segment denominator and overall baseline. The prevalence heatmaps use a shared color scale across periods for each metric type. Factor row order is held constant using all-history prevalence within each domain. The 2010–2021 period aggregates twelve years, whereas 2022 and 2023 are individual years; interpret composition differences accordingly.', '',
        '![2010–2021 domain prevalence](../figures/history_2010_2021/domain_prevalence_heatmap.png)', '',
        '![2022 domain prevalence](../figures/validation_2022/domain_prevalence_heatmap.png)', '',
        '![2023 domain prevalence](../figures/test_2023/domain_prevalence_heatmap.png)', '',
        'Period-specific 35-factor heatmaps and Top 5 charts are in the matching figure folders. CSV tables contain all four scopes: all_history_2010_2023, history_2010_2021, validation_2022, and test_2023. Validation/test names identify the original project time windows; expanded period totals include previously unlabeled records and therefore exceed the archived eligible validation/test counts.', '',
        '## Interpretation and limitations', '',
        '- NLP factors are classifier flags rather than human-confirmed labels. Per-factor held-out precision, recall, F1, and positive counts accompany the factor table. Three retained factors have fewer than ten held-out positives: Nonpaid randomness, Intrusive selling, and Account security and recovery. Their accuracy estimates are imprecise.',
        '- NLP predictors contribute to the model used to define the segments. Complaint contrasts between model-defined groups partly reflect that construction and do not independently prove why players stop playing.',
        '- Comparisons are descriptive and review-weighted. They do not adjust for game, language, review date, editing, or other composition differences. No causal effects or significance tests are claimed.',
        '- The historical cohort includes edited reviews. Review edits may occur after the outcome period. The supplied still_playing_d30 field is an open-ended snapshot activity proxy, not a fixed-horizon retention measure. Scores for records missing this field have no observed outcome against which to evaluate them here.',
        '- The eligible 2010–2021 training cohort is profiled using predictions from the model fitted on that cohort. These training-period contrasts can reflect model fit and must not be presented as independent prediction validation. Pooling historical records is appropriate for this descriptive coverage objective, not for estimating held-out predictive performance.',
        '- Final P55 thresholds were selected retrospectively after prior exposure to test results. These profiles are post hoc descriptive analysis, not a new untouched-test validation.', '',
        '## Verification', '',
        f'The join covers all {checks["joined_rows"]:,} cleaned historical records with zero unmatched NLP records. The join keys are unique. All 35 flags are non-null and binary; their sum matches the stored factor count. Every saved label agrees with its numeric decision rules, and all original 4,937,916 validation/test scores and assignments are unchanged.', '',
        'Independent aggregate verification is recorded under results/. Input checksums and the output inventory are in the accompanying analysis_manifest.json. Canonical project data, models, and archived results are read only during reproduction.', '']
    relative_figures=os.path.relpath(out.figures,out.report.parent).replace(os.sep,'/')
    out.report.write_text('\n'.join(report).replace('../figures/',relative_figures+'/'))
    out.readme.write_text("""# User Persona reproduction outputs

These outputs cover all 16,412,862 cleaned historical negative-review records, including the training period and 711,163 records without a binary D30 outcome. The source model, original eligible validation/test assignments, NLP flags, and P55 rule are preserved.

- Read [Complaint profiles](reports/complaint_profiles.md).
- See `figures/` for eight-domain and 35-factor heatmaps, percentage-point differences, and four Top 5 charts. PNG and vector PDF versions are supplied.
- See `tables/` for exact counts, rates, group overviews, and coverage by original model eligibility.
- See `data/review_segments_all_history.parquet` for review-level assignments.
- See `results/` for segmentation provenance, full-cohort verification, and the analysis inventory.

The main scope is all_history_2010_2023. Separate history_2010_2021, validation_2022, and test_2023 scopes include all reviews from those periods, including records missing D30. Training-cohort predictions are in sample and support descriptive profiling rather than independent predictive evaluation. Complaint percentages use all records in each segment and can overlap.

From the project root, run `python run.py persona` to regenerate profiles using the verified archived historical cache, or `python run.py persona --rebuild-segments` to also regenerate historical predictions with the frozen model. Run `python run.py verify-persona --result-dir runs/user_persona` to independently recount these outputs. The project README and docs/user_persona.md describe the archived submission layout and the complete method.
""")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--rebuild-segments',action='store_true')
    args=parser.parse_args()
    root=args.project_root.resolve()
    out=PersonaPaths(root,(args.output_dir or root/'runs/user_persona').resolve())
    out.create()
    with tempfile.TemporaryDirectory(prefix='team18-persona-') as scratch:
        grouped,factors,domains,domain_flags,sources,checks=aggregate(root,scratch,out,rebuild=args.rebuild_segments)
    domain_rates,factor_rates,top5,summary,baseline=build_tables(grouped,factors,domains,domain_flags)
    for name,table in [('domain_prevalence',domain_rates),('factor_prevalence',factor_rates),('top5_complaints',top5),('segment_overview',summary),('overall_prevalence',baseline)]:
        table.to_csv(out.tables/f'{name}.csv',index=False,float_format='%.12g')
    factors.to_csv(out.tables/'factor_dictionary.csv',index=False)
    domains.to_csv(out.tables/'domain_mapping.csv',index=False)
    for scope in SCOPES:
        for kind,table in [('domain',domain_rates),('factor',factor_rates)]:
            part=table.loc[table.scope.eq(scope)]
            wide=part.pivot(index='metric_id',columns='segment',values='prevalence_pct').reindex(columns=SEGMENTS)
            meta=part.drop_duplicates('metric_id').set_index('metric_id')[['domain','label','overall_prevalence_pct']]
            meta.join(wide).to_csv(out.tables/f'{kind}_prevalence_matrix_{scope}.csv',float_format='%.12g')
    checks['table_rows']={'domain_prevalence':len(domain_rates),'factor_prevalence':len(factor_rates),'top5':len(top5)}
    pd.DataFrame(checks['period_and_model_cohort_counts']).to_csv(out.tables/'coverage_by_period_and_model_cohort.csv',index=False)
    for kind in ['domain','factor']:
        stale=out.tables/f'{kind}_prevalence_matrix_combined_2022_2023.csv'
        if stale.exists():
            stale.unlink()
    assert len(domain_rates)==128 and len(factor_rates)==560 and len(top5)==80
    sns.set_theme(style='white',font='DejaVu Sans')
    plt.rcParams.update({'pdf.fonttype':42,'axes.titlelocation':'left'})
    domain_order=domains.domain.tolist()
    dmax=np.ceil(domain_rates.prevalence_pct.max()/5)*5
    fmax=np.ceil(factor_rates.prevalence_pct.max()/5)*5
    xmax=np.ceil(top5.prevalence_pct.max()/10)*10+7
    for scope in SCOPES:
        dest=out.figures if scope==SCOPES[0] else out.figures/scope
        dest.mkdir(exist_ok=True)
        plot_heatmap(domain_rates,summary,domain_order,scope,'domain',dest/'domain_prevalence_heatmap',vmax=dmax)
        plot_heatmap(factor_rates,summary,domain_order,scope,'factor',dest/'factor_prevalence_heatmap',vmax=fmax)
        plot_top5(top5,summary,scope,dest/'top5_complaints_by_segment',xmax)
        if scope==SCOPES[0]:
            plot_heatmap(domain_rates,summary,domain_order,scope,'domain',dest/'domain_difference_heatmap',difference=True)
            plot_heatmap(factor_rates,summary,domain_order,scope,'factor',dest/'factor_difference_heatmap',difference=True)
    write_report(out,domain_rates,factor_rates,top5,summary,checks)
    from .verify_complaint_profiles import verify
    independent = verify(root, out.base)
    checks['independent_full_cohort_recount_passed'] = independent['status'] == 'passed'
    manifest={'analysis':'All-history complaint prevalence by extended P55 segment','date':'2026-10-10','primary_scope':SCOPES[0],
        'scopes':SCOPES,'unit':'negative-review player-game record','prevalence_denominator':'all segment records within scope',
        'domain_definition':'any mapped frozen binary factor flag equals one','top5_ranking':'descending prevalence; factor ID for ties',
        'path_base':'project root for inputs/source_code; output directory for outputs',
        'inputs':[{'path':os.path.relpath(p,root),'sha256':digest(p)} for p in dict.fromkeys(sources.values())],
        'source_code':[{'path':str(p.relative_to(root)),'sha256':digest(p)} for p in sorted(Path(__file__).parent.glob('*.py'))],
        'checks':checks,'outputs':[str(p.relative_to(out.base)) for p in sorted(out.base.rglob('*')) if p.is_file() and p.name not in ['analysis_manifest.json','.DS_Store'] and '__pycache__' not in p.parts]}
    out.manifest.write_text(json.dumps(manifest,indent=2)+'\n')
    print(top5.loc[top5.scope.eq(SCOPES[0]),['segment','rank_within_segment','label','prevalence_pct','difference_vs_overall_pp']].to_string(index=False),flush=True)
    print('Built full-history complaint profiles, tables, and 14 PNG/PDF figure pairs.',flush=True)


if __name__=='__main__':
    main()
