"""Independently verify complaint aggregates with streamed Arrow/NumPy counts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from steam_reviews.paths import ROOT
from .paths import PersonaPaths

SEGMENTS = ['Hardcore','Dropout','New Blood','Nobody']
SCOPES = ['all_history_2010_2023','history_2010_2021','validation_2022','test_2023']


def verify(root, out=None, report_path=None):
    root=Path(root).resolve()
    layout=PersonaPaths(root,Path(out).resolve() if out is not None else None)
    source=root
    factors=pd.read_csv(source/'reports/tables/nlp/nlp_35_factors_dictionary.csv')
    domains=pd.read_csv(source/'reports/tables/nlp/nlp_domain_mapping.csv')
    flags=factors.flag_column.tolist()
    seg=pd.read_parquet(layout.segments,columns=['appid','recommendationid','split','model_cohort','segment','hours_at_review','prob_raw','prior_hours_cutoff','raw_probability_cutoff','high_prior','high_predicted_later'])
    keys=pd.MultiIndex.from_frame(seg[['appid','recommendationid']])
    assert keys.is_unique and len(keys)==16412862
    high_hours=seg.hours_at_review.to_numpy()>11.78
    high_score=seg.prob_raw.to_numpy()>=seg.raw_probability_cutoff.to_numpy()
    classes=pd.Categorical(seg.segment,categories=SEGMENTS).codes
    expected_classes=np.where(high_hours,np.where(high_score,0,1),np.where(high_score,2,3)).astype(np.int8)
    assert np.array_equal(expected_classes,classes)
    assert np.array_equal(high_hours,seg.high_prior.to_numpy()) and np.array_equal(high_score,seg.high_predicted_later.to_numpy())
    assert (seg.prior_hours_cutoff==11.78).all()
    del expected_classes,high_hours,high_score
    audit=json.loads((layout.audit).read_text())
    train_mask=seg.model_cohort.eq('train').to_numpy()
    assert int(train_mask.sum())==10763783
    assert abs(float(np.quantile(seg.prob_raw.to_numpy()[train_mask],.55,method='linear'))-audit['raw_score_cutoffs']['train'])<1e-12
    assert int(seg.model_cohort.eq('unlabeled').sum())==711163
    raw_scores=seg.prob_raw.to_numpy()
    saved_cutoffs=seg.raw_probability_cutoff.to_numpy()
    original_checked=0
    for batch in pq.ParquetFile(source/'results/segmentation/review_segments_p55.parquet').iter_batches(batch_size=250000,columns=['appid','recommendationid','segment','prob_raw','raw_probability_cutoff']):
        frame=batch.to_pandas()
        positions=keys.get_indexer(pd.MultiIndex.from_frame(frame[['appid','recommendationid']]))
        assert (positions>=0).all()
        assert np.array_equal(classes[positions],pd.Categorical(frame.segment,categories=SEGMENTS).codes)
        assert np.array_equal(raw_scores[positions],frame.prob_raw.to_numpy())
        assert np.array_equal(saved_cutoffs[positions],frame.raw_probability_cutoff.to_numpy())
        original_checked+=len(frame)
    assert original_checked==4937916
    del raw_scores,saved_cutoffs,train_mask
    seg=seg[['appid','recommendationid','split','segment']]
    classes=pd.Categorical(seg.segment,categories=SEGMENTS).codes
    periods=seg.split.map({'train':0,'val':1,'test':2}).to_numpy(dtype=np.int8)
    assert (classes>=0).all() and np.isin(periods,[0,1,2]).all()
    group=(periods*4+classes).astype(np.int8)
    seen=np.zeros(len(seg),dtype=np.int8)
    counts=np.zeros(12,dtype=np.int64)
    factor_counts=np.zeros((12,35),dtype=np.int64)
    domain_counts=np.zeros((12,8),dtype=np.int64)
    any_counts=np.zeros(12,dtype=np.int64)
    domain_indices=[[flags.index(x.strip()) for x in row.flag_columns.split(';')] for row in domains.itertuples()]
    print('Independently recounting every joined review with Arrow/NumPy...',flush=True)
    nlp=pq.ParquetFile(source/'data/processed/nlp/nlp_factor_scores_2023.parquet')
    for batch in nlp.iter_batches(batch_size=250000,columns=['appid','recommendationid',*flags]):
        lookup=pd.MultiIndex.from_arrays([batch.column(0).to_numpy(),batch.column(1).to_numpy()])
        positions=keys.get_indexer(lookup)
        selected=positions>=0
        if not selected.any():continue
        positions=positions[selected]
        np.add.at(seen,positions,1)
        matrix=np.column_stack([batch.column(i+2).to_numpy()[selected] for i in range(35)])
        assert np.isin(matrix,[0,1]).all()
        row_group=group[positions]
        for g in range(12):
            piece=matrix[row_group==g]
            counts[g]+=len(piece)
            factor_counts[g]+=piece.sum(axis=0,dtype=np.int64)
            any_counts[g]+=np.count_nonzero(piece.any(axis=1))
            for d,indices in enumerate(domain_indices):
                domain_counts[g,d]+=np.count_nonzero(piece[:,indices].any(axis=1))
    assert (seen==1).all()
    assert counts.sum()==len(seg)
    expected_factor={};expected_domain={};expected_totals={};expected_any={}
    for scope in SCOPES:
        for i,segment in enumerate(SEGMENTS):
            groups=[i,i+4,i+8] if scope==SCOPES[0] else [i+4*(SCOPES.index(scope)-1)]
            key=(scope,segment)
            expected_totals[key]=int(counts[groups].sum())
            expected_any[key]=int(any_counts[groups].sum())
            expected_factor.update({(*key,row.factor_id):int(factor_counts[groups,j].sum()) for j,row in enumerate(factors.itertuples())})
            expected_domain.update({(*key,row.domain):int(domain_counts[groups,j].sum()) for j,row in enumerate(domains.itertuples())})
    checked=0
    for filename,expected,nmetric in [('factor_prevalence',expected_factor,35),('domain_prevalence',expected_domain,8)]:
        table=pd.read_csv(layout.tables/f'{filename}.csv')
        assert len(table)==len(SCOPES)*4*nmetric
        assert not table.duplicated(['scope','segment','metric_id']).any()
        for row in table.itertuples():
            total=expected_totals[(row.scope,row.segment)]
            numerator=expected[(row.scope,row.segment,row.metric_id)]
            overall_total=sum(expected_totals[(row.scope,s)] for s in SEGMENTS)
            overall_count=sum(expected[(row.scope,s,row.metric_id)] for s in SEGMENTS)
            overall=overall_count/overall_total
            assert row.total_reviews==total and row.flagged_reviews==numerator
            assert abs(row.prevalence-numerator/total)<1e-11
            assert abs(row.prevalence_pct-100*numerator/total)<1e-9
            assert abs(row.overall_prevalence_pct-100*overall)<1e-9
            assert abs(row.difference_vs_overall_pp-100*(numerator/total-overall))<1e-9
            assert abs(row.lift_vs_overall-(numerator/total)/overall)<1e-9
            checked+=1
    overview=pd.read_csv(layout.tables/'segment_overview.csv')
    for row in overview.itertuples():
        key=(row.scope,row.segment)
        assert row.total_reviews==expected_totals[key]
        assert row.any_complaint_reviews==expected_any[key]
        assert row.factor_mentions==sum(expected_factor[(*key,f)] for f in factors.factor_id)
    top5=pd.read_csv(layout.tables/'top5_complaints.csv')
    all_factors=pd.read_csv(layout.tables/'factor_prevalence.csv')
    for scope in SCOPES:
        for segment in SEGMENTS:
            rows=all_factors.loc[all_factors.scope.eq(scope)&all_factors.segment.eq(segment)].sort_values(['prevalence','metric_id'],ascending=[False,True]).head(5)
            top=top5.loc[top5.scope.eq(scope)&top5.segment.eq(segment)].sort_values('rank_within_segment')
            assert rows.metric_id.tolist()==top.metric_id.tolist()
            assert top.rank_within_segment.tolist()==[1,2,3,4,5]
    for p in [layout.report,layout.readme]:
        assert not re.search('[\u3400-\u9fff]',p.read_text()),p
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if not target.startswith(('http:','https:','#')):assert (p.parent/target).exists(),(p,target)
    report={'status':'passed','method':'Independent Arrow/NumPy full-cohort recount; no DuckDB aggregation reused',
        'records_checked':len(seg),'every_segment_key_matched_once':True,'aggregate_rows_checked':checked,
        'factor_counts_and_rates_match':True,'domain_union_counts_and_rates_match':True,
        'overall_baselines_and_percentage_point_differences_match':True,
        'all_sixteen_top5_lists_rank_correctly':True,'cohort_totals_and_intensity_match':True,
        'every_numeric_segmentation_assignment_matches':True,
        'original_scores_cutoffs_and_labels_preserved':original_checked,
        'new_training_percentile_independently_verified':True,
        'all_711163_unlabeled_records_included':True}
    destination=Path(report_path) if report_path is not None else (layout.verification if out is not None else root/'runs/user_persona_verification.json')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--report-path',type=Path)
    args=parser.parse_args()
    verify(args.project_root,args.output_dir,args.report_path)
