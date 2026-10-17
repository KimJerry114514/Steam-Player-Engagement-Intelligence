"""Select and apply percentile segmentation using validation maximin coverage.

python segment_p55.py --bundle SOURCE.zip --training-data CLEANED.parquet --output-dir results

No model is retrained. Actual outcome labels and test data never enter selection.
The supplied cohort selects P55/P55 under the stated coverage objective.
This is retrospective descriptive selection, not prospective business optimization.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = ['Hardcore', 'Dropout', 'New Blood', 'Nobody']
GRID = np.arange(25,76,5)

def labels(h, p, hc, pc):
    a, b = np.asarray(h)>hc, np.asarray(p)>=pc
    return np.select([a&b, a&~b, ~a&b], [0,1,2], default=3).astype('int8')

def select_candidate(hours, scores, prior_cuts):
    """Accept only validation axes and training thresholds; no outcomes/test."""
    score_cuts = np.quantile(scores, GRID/100, method='linear')
    rows=[]
    for hp in GRID:
        for sp, sc in zip(GRID,score_cuts):
            lab=labels(hours,scores,prior_cuts[int(hp)],sc)
            shares=np.bincount(lab,minlength=4)/len(lab)
            rows.append({'prior_percentile':int(hp),'score_percentile':int(sp),
                         'hours_cutoff':float(prior_cuts[int(hp)]),'raw_score_cutoff':float(sc),
                         'smallest_group_share':float(shares.min()),
                         'largest_group_share':float(shares.max()),
                         'distance_from_medians':int(abs(hp-50)+abs(sp-50)),
                         **{n+'_share':float(shares[i]) for i,n in enumerate(NAMES)}})
    candidates=pd.DataFrame(rows)
    winner=candidates.sort_values(['smallest_group_share','distance_from_medians','prior_percentile','score_percentile'],ascending=[False,True,True,True]).iloc[0].to_dict()
    return candidates, winner

def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--bundle',required=True); a.add_argument('--training-data',required=True)
    a.add_argument('--output-dir',required=True); args=a.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    bundle = Path(args.bundle)
    if bundle.is_dir():
        raw = (bundle / 'predictions_val_test_calibrated.parquet').read_bytes()
        data = pd.read_parquet(io.BytesIO(raw))
        score_report = json.loads((bundle / 'score_report.json').read_text())
    else:
        with ZipFile(bundle) as z:
            def read(name):
                paths=[p for p in z.namelist() if p.rsplit('/',1)[-1]==name]
                assert len(paths)==1
                return z.read(paths[0])
            raw=read('predictions_val_test_calibrated.parquet')
            data=pd.read_parquet(io.BytesIO(raw))
            score_report=json.loads(read('score_report.json'))
    assert not data.duplicated(['appid','recommendationid']).any()
    assert data[['hours_at_review','prob_raw','prob_platt','split']].notna().all().all()
    assert np.isfinite(data[['hours_at_review','prob_raw','prob_platt']].to_numpy()).all()
    assert data.hours_at_review.ge(0).all() and data.prob_raw.between(0,1).all()
    assert set(data.split.unique())=={'val','test'}
    dates=pd.to_datetime(data.review_created_utc,utc=True)
    assert ((dates.ge('2022-01-01')&dates.lt('2023-01-01'))==data.split.eq('val')).all()
    assert (dates.ge('2023-01-01')==data.split.eq('test')).all()
    for split in ['val','test']: assert data.split.eq(split).sum()==score_report['data_checks'][split+'_rows']
    pieces=[]
    for b in pq.ParquetFile(args.training_data).iter_batches(batch_size=300000,columns=['review_created_utc','still_playing_d30','hours_at_review']):
        f=b.to_pandas(); d=pd.to_datetime(f.review_created_utc,utc=True)
        m=d.ge('2010-10-15')&d.lt('2022-01-01')&f.still_playing_d30.isin([0,1])
        pieces.append(f.loc[m,'hours_at_review'].to_numpy())
    train=np.concatenate(pieces);del pieces
    assert len(train)==score_report['data_checks']['train_rows']
    assert np.isfinite(train).all() and (train>=0).all()
    prior_cuts=dict(zip(GRID.astype(int),np.quantile(train,GRID/100,method='linear')))
    val=data.loc[data.split.eq('val')]
    candidates, chosen=select_candidate(val.hours_at_review.to_numpy(),val.prob_raw.to_numpy(),prior_cuts)
    selection={'purpose':'descriptive coverage, not business or predictive optimality',
        'recommended':chosen,
        'selection_rule':'Choose the exact maximin-coverage grid winner',
        'outcome_labels_used_for_selection':False,'test_rows_used_for_selection':False,
        'retrospective_analysis_after_prior_test_exposure':True,
        'source_predictions_sha256':hashlib.sha256(raw).hexdigest()}
    save_json(out/'threshold_decision.json',selection)
    candidates.to_json(out/'candidate_comparison.json',orient='records',indent=2)
    print(json.dumps(selection,indent=2),flush=True)
    pct=np.arange(1,100)
    qtrain=np.quantile(train,pct/100);qval=np.quantile(val.prob_raw,pct/100)
    qcal=np.quantile(val.prob_platt,pct/100)
    save_json(out/'percentile_values.json',[{'percentile':int(q),'training_hours':float(h),'validation_raw_score':float(s),'validation_calibrated_score':float(c)} for q,h,s,c in zip(pct,qtrain,qval,qcal)])
    # Full-year validation cutoffs held fixed within each quarter for diagnostics.
    diagnostics=[]
    for title,r in [('selected',chosen)]:
        for quarter,part in val.groupby(val.review_created_utc.dt.quarter):
            lab=labels(part.hours_at_review,part.prob_raw,r['hours_cutoff'],r['raw_score_cutoff'])
            sh=np.bincount(lab,minlength=4)/len(part)
            diagnostics.append({'policy':title,'quarter':int(quarter),'records':len(part),
                                'smallest_share':float(sh.min()),**{n+'_share':float(sh[i]) for i,n in enumerate(NAMES)}})
    save_json(out/'quarterly_stability.json',diagnostics)
    summary=[];thresholds=[];recommend_codes=np.empty(len(data),dtype='int8')
    raw_cuts=np.empty(len(data)); cal_cuts=np.empty(len(data))
    for split in ['val','test']:
        mask=data.split.eq(split).to_numpy(); part=data.loc[mask];q=chosen['score_percentile']/100
        hc=chosen['hours_cutoff'];sc=float(np.quantile(part.prob_raw,q));cc=float(np.quantile(part.prob_platt,q))
        lab=labels(part.hours_at_review,part.prob_raw,hc,sc)
        assert np.array_equal(lab,labels(part.hours_at_review,part.prob_platt,hc,cc))
        recommend_codes[mask]=lab
        raw_cuts[mask]=sc;cal_cuts[mask]=cc
        thresholds.append({'split':split,'prior_cutoff_hours':hc,'prior_operator':'>','prior_percentile':chosen['prior_percentile'],
            'raw_score_cutoff':sc,'calibrated_score_cutoff':cc,'score_operator':'>=','score_percentile':chosen['score_percentile'],
            'score_reference':'same-period score distribution','score_cutoff_ties':int(part.prob_raw.eq(sc).sum())})
        for i,n in enumerate(NAMES):
            sub=part.loc[lab==i]
            summary.append({'split':split,'segment':n,'rows':len(sub),'share':len(sub)/len(part),
                            'observed_d30_proxy':float(sub.still_playing_d30.mean())})
    assigned=data.copy();assigned['segment']=pd.Categorical.from_codes(recommend_codes,categories=NAMES)
    assigned['high_prior']=assigned.hours_at_review.gt(chosen['hours_cutoff'])
    assigned['high_predicted_later']=assigned.prob_raw.ge(raw_cuts)
    assigned['prior_hours_cutoff']=chosen['hours_cutoff']
    assigned['raw_probability_cutoff']=raw_cuts;assigned['platt_probability_cutoff']=cal_cuts
    assigned.to_parquet(out/'review_segments_p55.parquet',index=False)
    save_json(out/'segment_summary.json',summary);save_json(out/'applied_thresholds.json',thresholds)
    save_json(out/'selected_segmentation_rules.json',{
        'version':'Final P55 2026-10-09','prior_percentile':int(chosen['prior_percentile']),
        'score_percentile':int(chosen['score_percentile']),'prior_cutoff_hours':chosen['hours_cutoff'],
        'prior_operator':'>','score_operator':'>=',
        'score_reference':'Same-period score percentile, separately for validation and test; approximately top 45 percent for P55',
        'selection_cohort':'2022 validation scores; prior cutoffs from eligible 2010-2021 training hours',
        'objective':'Maximize the smallest 2022 quadrant share over the P25 to P75 grid in steps of five',
        'tie_break':'Closest to the medians, then lower prior and score percentiles',
        'model_or_calibration_retrained':False,
        'test_used_in_selection':False,'actual_outcomes_used_in_selection':False,
        'retrospective_revision_after_prior_test_exposure':True,
        'labels':{'high_high':'Hardcore','high_low':'Dropout','low_high':'New Blood','low_low':'Nobody'}})
    # Figures use pre-test reference distributions; no visual search on test outcomes.
    plt.rcParams.update({'font.size':15,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    axes[0].plot(pct,qtrain,color='#215e83',lw=2)
    axes[0].set_yscale('function',functions=(np.log1p,np.expm1))
    axes[0].set(xlabel='Percentile in training cohort',ylabel='Hours at review (log1p scale)',title='Prior playtime percentiles',xticks=[10,25,40,50,60,75,90],yticks=[0,1,10,100,1000],ylim=(0,float(qtrain.max())*1.15))
    axes[1].plot(pct,qcal,color='#215e83',lw=2)
    axes[1].set(xlabel='Percentile in 2022 validation cohort',ylabel='Calibrated prediction score',title='Prediction score percentiles',xticks=[10,25,40,50,60,75,90])
    for ax,q,y in [(axes[0],chosen['prior_percentile'],chosen['hours_cutoff']),
                   (axes[1],chosen['score_percentile'],float(np.quantile(val.prob_platt,chosen['score_percentile']/100)))]:
        ax.axvline(q,color='#a13d3d',ls='--',lw=1.3);ax.scatter([q],[y],color='#a13d3d',zorder=5)
        ax.annotate(f'P{q:g} = {y:.4f}' if ax==axes[1] else f'P{q:g} = {y:g} hours',(q,y),xytext=(10,-26),textcoords='offset points',color='#a13d3d')
        ax.grid(alpha=.2)
    fig.suptitle('Selected percentile thresholds',fontsize=19)
    fig.savefig(out/'percentile_distributions.png',dpi=180);plt.close(fig)
    plt.rcParams.update({'font.size':11})
    pivot=candidates.pivot(index='prior_percentile',columns='score_percentile',values='smallest_group_share')*100
    fig,ax=plt.subplots(figsize=(8,7),layout='constrained')
    im=ax.imshow(pivot.to_numpy(),origin='lower',cmap='YlGnBu',vmin=0,vmax=pivot.to_numpy().max())
    ax.set(xticks=range(len(GRID)),yticks=range(len(GRID)),xticklabels=GRID,yticklabels=GRID,
           xlabel='2022 prediction score percentile',ylabel='Training playtime percentile',title='Smallest segment share across 121 candidate rules')
    for i in range(len(GRID)):
        for j in range(len(GRID)):
            v=pivot.iloc[i,j];ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=8.5,color='white' if v>5 else '#152333')
    wx=list(GRID).index(int(chosen['score_percentile']));wy=list(GRID).index(int(chosen['prior_percentile']))
    ax.plot(wx,wy,'o',mfc='none',mec='#f1a340',mew=2,markersize=31,label='Largest minimum group')
    fig.colorbar(im,ax=ax,label='Smallest group (% of 2022 records)')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=2,fontsize=9,markerscale=.5,frameon=False)
    fig.savefig(out/'threshold_coverage_heatmap.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    for name,style,color in [('selected','-','#215e83')]:
        sub=pd.DataFrame(diagnostics).query('policy==@name')
        ax.plot(sub.quarter,sub.smallest_share*100,style,marker='o',label=name.replace('_',' ').title(),color=color)
    ax.set(xticks=[1,2,3,4],xticklabels=['2022 Q1','2022 Q2','2022 Q3','2022 Q4'],ylabel='Smallest segment share (%)',title='Temporal coverage diagnostic with full-year score cutoffs held fixed')
    ax.grid(alpha=.2);ax.legend();fig.savefig(out/'quarterly_coverage.png',dpi=180);plt.close(fig)
    # Independent row checks use direct boolean assignment rather than labels().
    saved=pd.read_parquet(out/'review_segments_p55.parquet')
    pd.testing.assert_frame_equal(saved[data.columns],data)
    for t in thresholds:
        part=saved.loc[saved.split.eq(t['split'])]
        high_h=part.hours_at_review.gt(t['prior_cutoff_hours']);high_s=part.prob_raw.ge(t['raw_score_cutoff'])
        expected=np.full(len(part),'Nobody',dtype=object)
        expected[high_h&high_s]='Hardcore';expected[high_h&~high_s]='Dropout';expected[~high_h&high_s]='New Blood'
        assert np.array_equal(expected,part.segment.astype(str).to_numpy())
    save_json(out/'verification.json',{'status':'passed','rows_checked':len(saved),'original_prediction_columns_unchanged':True,
        'every_assignment_independently_verified':True,'training_rows':len(train),'candidate_rules':len(candidates),
        'outcome_labels_used_for_threshold_selection':False,'test_rows_used_for_threshold_selection':False})
    # Report figure retains the four canonical labels used in the team report.
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained')
    test_rows={r['segment']:r for r in summary if r['split']=='test'}
    for n,x,y,color in [('Hardcore',1,1,'#d7eee3'),('Dropout',1,0,'#f9dfd7'),('New Blood',0,1,'#dbe8f8'),('Nobody',0,0,'#eaeaf0')]:
        r=test_rows[n]
        ax.add_patch(Rectangle((x,y),1,1,facecolor=color,edgecolor='white',lw=4))
        ax.text(x+.5,y+.72,n,ha='center',va='center',weight='bold',fontsize=16)
        ax.text(x+.5,y+.44,f"{r['share']:.2%}",ha='center',va='center',fontsize=20)
        ax.text(x+.5,y+.17,f"{r['rows']:,} review records",ha='center',va='center',fontsize=13)
    ax.set(xlim=(0,2),ylim=(0,2),xticks=[.5,1.5],yticks=[.5,1.5],
           xticklabels=[f"Prior hours <= {chosen['hours_cutoff']:g}",f"Prior hours > {chosen['hours_cutoff']:g}"],
           yticklabels=[f"Below period P{chosen['score_percentile']:g}",f"At or above period P{chosen['score_percentile']:g}"])
    ax.tick_params(length=0,pad=8,labelsize=11)
    ax.set_ylabel('Predicted engagement',fontsize=11,labelpad=12)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.savefig(out/'segmentation_2023_p55.png',dpi=200);plt.close(fig)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print('Completed selected segmentation, figures and independent row verification.',flush=True)

if __name__=='__main__':main()
