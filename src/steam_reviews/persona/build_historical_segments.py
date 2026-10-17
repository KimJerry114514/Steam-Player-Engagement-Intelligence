"""Extend the frozen P55 segmentation to every cleaned historical negative review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import shutil
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from steam_reviews.paths import ROOT
from .paths import PersonaPaths


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def ident(value):
    return '"' + value.replace('"', '""') + '"'


def feature_matrix(frame, meta):
    dates = pd.to_datetime(frame.review_created_utc, utc=True)
    frame['review_year'] = dates.dt.year.astype('float32')
    frame['review_month'] = dates.dt.month.astype('float32')
    frame['review_dayofweek'] = dates.dt.dayofweek.astype('float32')
    frame['review_quarter_of_year'] = dates.dt.quarter.astype('float32')
    frame['log_hours_at_review'] = np.log1p(pd.to_numeric(frame.hours_at_review).clip(lower=0)).astype('float32')
    codes = {s: i for i, s in enumerate(meta['language_categories'])}
    frame['language_code'] = frame.language.fillna(meta['language_fill_value']).map(codes).fillna(meta['unknown_language_code']).astype('float32')
    return np.column_stack([pd.to_numeric(frame['language_code' if f == 'language' else f], errors='coerce').to_numpy(dtype=np.float32, na_value=np.nan) for f in meta['transformed_features']])


def ensure_segments(root, out, rebuild=False):
    final = root
    sources = {
        'cleaned_reviews': final / 'data/processed/negative_reviews_cleaned.parquet',
        'nlp_scores': final / 'data/processed/nlp/nlp_factor_scores_2023.parquet',
        'archived_segments': final / 'results/segmentation/review_segments_p55.parquet',
        'model': final / 'models/lightgbm/training_full_model.txt',
        'model_metadata': final / 'models/lightgbm/training_full_model.json',
        'applied_thresholds': final / 'results/segmentation/applied_thresholds.json',
        'selected_rules': final / 'results/segmentation/selected_segmentation_rules.json',
    }
    dest = out.segments
    audit_path = out.audit
    fingerprints = {str(p.relative_to(root)): digest(p) for p in sources.values()}
    code_hash = digest(Path(__file__))
    archived = PersonaPaths(root)
    candidates = [(dest,audit_path),(archived.segments,archived.audit)]
    if not rebuild:
        for cached_data,cached_audit in candidates:
            if not cached_data.exists() or not cached_audit.exists():
                continue
            audit = json.loads(cached_audit.read_text())
            cached_code = audit.get('packaged_source_code_sha256',audit.get('source_code_sha256'))
            if audit.get('input_sha256') == fingerprints and cached_code == code_hash and audit.get('output_sha256') == digest(cached_data):
                if cached_data != dest:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    audit_path.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(cached_data,dest)
                    shutil.copy2(cached_audit,audit_path)
                print('Reusing the verified full-history segmentation cache.',flush=True)
                return dest,audit,sources
    audit_path.parent.mkdir(parents=True,exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(sources['model_metadata'].read_text())
    model = lgb.Booster(model_file=str(sources['model']))
    assert model.num_trees() == meta['rounds_trained'] == 1000
    assert model.feature_name() == [f.replace(' ', '_') for f in meta['transformed_features']]
    assert digest(sources['model']) == meta['model_txt_sha256']
    thresholds = {r['split']: r['raw_score_cutoff'] for r in json.loads(sources['applied_thresholds'].read_text())}
    rules = json.loads(sources['selected_rules'].read_text())
    assert rules['prior_cutoff_hours'] == 11.78 and rules['score_percentile'] == 55
    engineered = {'language', 'log_hours_at_review', 'review_year', 'review_month', 'review_dayofweek', 'review_quarter_of_year'}
    probabilities = [f for f in meta['transformed_features'] if f.startswith('p_')]
    plain = [f for f in meta['transformed_features'] if f not in engineered and f not in probabilities]
    base = ['appid', 'recommendationid', 'review_created_utc', 'hours_at_review', 'still_playing_d30', 'edited', 'language', *plain]
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='team18-history-') as scratch:
        con = duckdb.connect()
        con.execute("SET TimeZone='UTC'")
        con.execute("SET memory_limit='2GB'")
        con.execute('SET threads=4')
        con.execute(f'SET temp_directory={literal(scratch)}')
        for name, key in [('cleaned', 'cleaned_reviews'), ('nlp', 'nlp_scores'), ('archived', 'archived_segments')]:
            con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({literal(sources[key])})')
        n, unique_n, bad = con.execute('SELECT count(*), count(DISTINCT (appid,recommendationid)), count(*) FILTER(WHERE appid IS NULL OR recommendationid IS NULL OR review_created_utc IS NULL OR hours_at_review IS NULL OR NOT isfinite(hours_at_review) OR hours_at_review<0) FROM cleaned').fetchone()
        assert n == unique_n == 16412862 and bad == 0
        nlp_n, nlp_unique = con.execute('SELECT count(*),count(DISTINCT(appid,recommendationid)) FROM nlp').fetchone()
        assert nlp_n == nlp_unique == n
        assert con.execute('SELECT count(*) FROM cleaned c ANTI JOIN nlp p USING(appid,recommendationid)').fetchone()[0] == 0
        reader = con.execute('SELECT ' + ','.join('c.' + ident(c) for c in base) + ',' + ','.join('p.'+ident(c) for c in probabilities) + ',a.prob_raw AS archived_prob_raw FROM cleaned c JOIN nlp p USING(appid,recommendationid) LEFT JOIN archived a USING(appid,recommendationid)').to_arrow_reader(batch_size=100000)
        raw_path = Path(scratch) / 'raw_predictions.parquet'
        writer = None
        processed = newly_scored = reused = 0
        largest_sample_difference = 0.0
        for batch in reader:
            frame = batch.to_pandas()
            existing = frame.archived_prob_raw.notna().to_numpy()
            raw = frame.archived_prob_raw.to_numpy(dtype=np.float64, copy=True)
            if (~existing).any():
                raw[~existing] = model.predict(feature_matrix(frame.loc[~existing].copy(), meta), num_threads=4)
            if reused == 0 and existing.any():
                sample = frame.loc[existing].head(512).copy()
                delta = np.abs(model.predict(feature_matrix(sample.copy(), meta), num_threads=4) - sample.archived_prob_raw.to_numpy())
                largest_sample_difference = float(delta.max())
                assert largest_sample_difference < 1e-12, largest_sample_difference
            assert np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all()
            record = frame[['appid','recommendationid','review_created_utc','hours_at_review','still_playing_d30','edited']].copy()
            record['prob_raw'] = raw
            record['archived_label_available'] = existing
            table = pa.Table.from_pandas(record, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(raw_path, table.schema, compression='zstd')
            writer.write_table(table)
            processed += len(frame);newly_scored += int((~existing).sum());reused += int(existing.sum())
            if processed % 1000000 == 0 or processed == n:
                print(f'Historical predictions: {processed:,}/{n:,}; newly scored {newly_scored:,}; elapsed {time.monotonic()-t0:.0f}s', flush=True)
        writer.close()
        assert processed == n and reused == 4937916
        con.execute(f'CREATE VIEW raw_predictions AS SELECT *, CASE WHEN review_created_utc < TIMESTAMPTZ \'2022-01-01\' THEN \'train\' WHEN review_created_utc < TIMESTAMPTZ \'2023-01-01\' THEN \'val\' ELSE \'test\' END AS split FROM read_parquet({literal(raw_path)})')
        # The added historical cutoff uses the original eligible full-training cohort.
        # Only cohort membership is used; the actual binary outcome is not used.
        train_n, train_cut = con.execute("SELECT count(*),quantile_cont(prob_raw,0.55) FROM raw_predictions WHERE split='train' AND still_playing_d30 IN(0,1)").fetchone()
        assert train_n == meta['rows']
        thresholds['train'] = train_cut
        cutoff = f"(CASE split WHEN 'train' THEN {repr(train_cut)} WHEN 'val' THEN {repr(thresholds['val'])} ELSE {repr(thresholds['test'])} END)::DOUBLE"
        temporary = dest.with_suffix('.tmp.parquet')
        con.execute(f'''COPY (WITH axes AS (SELECT *, 11.78::DOUBLE AS prior_hours_cutoff,
            {cutoff} AS raw_probability_cutoff,
            CASE WHEN still_playing_d30 IN(0,1) THEN split ELSE 'unlabeled' END AS model_cohort
            FROM raw_predictions), flags AS (SELECT *, hours_at_review > prior_hours_cutoff AS high_prior,
            prob_raw >= raw_probability_cutoff AS high_predicted_later FROM axes)
            SELECT *, CASE WHEN high_prior AND high_predicted_later THEN 'Hardcore'
                WHEN high_prior THEN 'Dropout' WHEN high_predicted_later THEN 'New Blood' ELSE 'Nobody' END AS segment
            FROM flags) TO {literal(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD)''')
        con.execute(f'CREATE VIEW expanded AS SELECT * FROM read_parquet({literal(temporary)})')
        mismatch = con.execute('''SELECT count(*) FROM archived a FULL JOIN expanded e USING(appid,recommendationid)
            WHERE a.appid IS NOT NULL AND (e.appid IS NULL OR a.segment IS DISTINCT FROM e.segment
            OR a.prob_raw IS DISTINCT FROM e.prob_raw OR a.raw_probability_cutoff IS DISTINCT FROM e.raw_probability_cutoff
            OR a.prior_hours_cutoff IS DISTINCT FROM e.prior_hours_cutoff)''').fetchone()[0]
        assert mismatch == 0
        rows = con.execute('SELECT split,model_cohort,count(*) FROM expanded GROUP BY 1,2 ORDER BY 1,2').fetchall()
        extent = con.execute('SELECT min(review_created_utc)::VARCHAR,max(review_created_utc)::VARCHAR FROM expanded').fetchone()
        assert pq.ParquetFile(temporary).metadata.num_rows == n
        temporary.replace(dest)
        con.close()
    audit = {'status':'passed', 'rows':n, 'date_range_utc':extent, 'new_predictions':newly_scored,
        'archived_predictions_and_labels_preserved':reused, 'archived_assignment_mismatches':mismatch,
        'preprocessing_reproduction_sample_max_abs_error':largest_sample_difference,
        'training_cutoff_reference_rows':train_n, 'prior_hours_cutoff':11.78, 'score_percentile':55,
        'raw_score_cutoffs':thresholds, 'period_and_model_cohort_counts':[{'split':s,'model_cohort':m,'rows':r} for s,m,r in rows],
        'training_predictions':'Final fitted model, in sample for the eligible original training cohort; not out-of-fold predictions or independent evaluation.',
        'unlabeled_records':'Scored without requiring a binary D30 outcome; the original eligible-cohort cutoff for their period is applied.',
        'method':'Existing P55 rule extended without retraining or threshold optimization; new training-period cutoff uses the original eligible training cohort scores.',
        'input_sha256':fingerprints, 'source_code_sha256':code_hash, 'output_sha256':digest(dest), 'seconds':round(time.monotonic()-t0,1)}
    audit_path.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)
    return dest, audit, sources


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--rebuild-segments',action='store_true',help='rescore all additional historical records instead of reusing the verified cache')
    args = parser.parse_args()
    root=args.project_root.resolve()
    layout=PersonaPaths(root,(args.output_dir or root/'runs/user_persona').resolve())
    ensure_segments(root,layout,rebuild=args.rebuild_segments)
