"""Independent all-row verification of the selected P55 delivery.
Does not import or call the assignment script.
"""
import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

def verify(bundle, result_dir, report_path=None):
    out=Path(result_dir)
    if Path(bundle).is_dir():
        source=pd.read_parquet(Path(bundle)/'predictions_val_test_calibrated.parquet')
    else:
        with ZipFile(bundle) as z:
            paths=[p for p in z.namelist() if p.rsplit('/',1)[-1]=='predictions_val_test_calibrated.parquet']
            assert len(paths)==1
            source=pd.read_parquet(io.BytesIO(z.read(paths[0])))
    saved=pd.read_parquet(out/'review_segments_p55.parquet')
    pd.testing.assert_frame_equal(source,saved[source.columns],check_exact=True)
    assert not saved.duplicated(['appid','recommendationid']).any()
    rules=json.loads((out/'selected_segmentation_rules.json').read_text())
    assert rules['prior_percentile']==rules['score_percentile']==55
    assert rules['prior_cutoff_hours']==11.78
    candidates=pd.read_json(out/'candidate_comparison.json')
    best=candidates.sort_values(['smallest_group_share','distance_from_medians','prior_percentile','score_percentile'],ascending=[False,True,True,True]).iloc[0]
    assert best.prior_percentile==best.score_percentile==55
    summaries=json.loads((out/'segment_summary.json').read_text())
    cutoffs=json.loads((out/'applied_thresholds.json').read_text())
    report={'rows_checked':len(source),'source_columns_unchanged':True,'selected_pair_is_grid_winner':True,'checks':[]}
    names=['Hardcore','Dropout','New Blood','Nobody']
    for split in ['val','test']:
        mask=source.split.eq(split);part=source.loc[mask];stored=saved.loc[mask]
        sc=float(np.quantile(part.prob_raw,.55,method='linear'))
        cc=float(np.quantile(part.prob_platt,.55,method='linear'))
        prior=part.hours_at_review.to_numpy()>11.78;later=part.prob_raw.to_numpy()>=sc
        expected=np.full(len(part),'Nobody',dtype=object)
        expected[prior&later]='Hardcore';expected[prior&~later]='Dropout';expected[~prior&later]='New Blood'
        np.testing.assert_array_equal(expected,stored.segment.astype(str))
        np.testing.assert_array_equal(prior,stored.high_prior)
        np.testing.assert_array_equal(later,stored.high_predicted_later)
        np.testing.assert_array_equal(later,part.prob_platt.to_numpy()>=cc)
        assert stored.prior_hours_cutoff.eq(11.78).all()
        assert stored.raw_probability_cutoff.eq(sc).all() and stored.platt_probability_cutoff.eq(cc).all()
        t=next(x for x in cutoffs if x['split']==split)
        assert t['raw_score_cutoff']==sc and t['calibrated_score_cutoff']==cc
        assert sum(x['rows'] for x in summaries if x['split']==split)==len(part)
        for name in names:
            chosen=expected==name;r=next(x for x in summaries if x['split']==split and x['segment']==name)
            assert r['rows']==int(chosen.sum())
            assert abs(r['share']-chosen.mean())<1e-12
            assert abs(r['observed_d30_proxy']-part.loc[chosen,'still_playing_d30'].mean())<1e-12
        report['checks'].append({'split':split,'rows':len(part),'every_assignment_correct':True,
            'raw_calibrated_assignments_equal':True,'all_summaries_correct':True})
    report['status']='passed'
    report_file = Path(report_path) if report_path else out/'independent_verification.json'
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True);parser.add_argument('--result-dir',required=True)
    parser.add_argument('--report-path')
    args=parser.parse_args();verify(args.bundle,args.result_dir,args.report_path)
