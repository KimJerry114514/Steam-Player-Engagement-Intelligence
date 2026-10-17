# Steam Player Engagement Intelligence

Beyond Negative Reviews: Predicting Player Engagement and Identifying At-Risk Players on Steam

Duke Fuqua · Data Science Final Project · Team 18

## Overview

A five-person team project combining player behavior and NLP-derived complaint factors to predict observed post-review engagement and interpret negative Steam reviews.

[Final presentation](docs/steam-project.pptx) · [Model comparison](reports/tables/cross_validation/cv_summary.csv) · [2023 test results](results/calibration/test_evaluation.json)

## Business Question

Negative reviews do not necessarily mean players will stop playing. Which reviews signal disengagement risk, and which complaint patterns could help prioritize different product or support responses?

## Data

- Primary source: 113,883,717 Steam reviews across 8,183 games.
- Broader analysis pipeline: approximately 16.4 million negative reviews.
- SteamDB game metadata merged by AppID, matching approximately 89.8% of unique AppIDs.

The `still_playing_d30` target reflects observed post-review engagement from historical playtime snapshots and should not be interpreted as confirmed activity exactly 30 days after a review.

Raw reviews, processed datasets, row-level predictions, caches, and trained model weights are intentionally excluded from this public repository.

## NLP Complaint Analysis

LLM-assisted coding of 3,248 sampled reviews informed a taxonomy of 78 complaint factors. A multilingual classifier achieved support-weighted F1 = 0.66; 35 sufficiently reliable factors were retained for downstream analysis. This was a team contribution, not the work of one author alone.

## Predictive Modeling

Four specifications were compared using 10-fold cross-validation on 8,873,576 negative reviews:

| Model | Features |
| --- | --- |
| Logistic Regression | Structured |
| L2 Logistic Regression | Structured + NLP |
| LightGBM | Structured |
| LightGBM | Structured + NLP |

The best model was LightGBM with structured and NLP features: cross-validation AP = 0.8457 and ROC-AUC = 0.8412.

The final model trained on 10,763,783 labeled reviews from 2010–2021. Validation used 2022; the Jan–Sep 2023 test set contained 2,154,782 reviews.

## Player Segmentation

| Segment | Prior playtime | Predicted engagement | 2023 share |
| --- | --- | --- | --- |
| Hardcore | High | High | 36.06% |
| Dropout | High | Low | 6.59% |
| New Blood | Low | High | 8.94% |
| Nobody | Low | Low | 48.41% |

These are relative engagement profiles, not confirmed churn labels. The “Nobody” label does not imply low-value feedback.

## Key Results

- **LightGBM + NLP: test ROC-AUC = 0.824.**
- **Top predicted-engagement decile: 88.7% observed engagement and 2.38× lift.**
- Four profiles distinguish prior playtime from predicted engagement and support further investigation of different complaint patterns.

## Limitations

- The target is an engagement proxy derived from historical lifetime-playtime snapshots with unequal follow-up, not confirmed activity exactly 30 days after a review.
- Average calibrated prediction was **46.0% versus 37.3% observed engagement** on the 2023 test set, an 8.7 percentage-point overestimate. Periodic recalibration would be needed for deployment.
- Generalization differed between unseen games (AUC = 0.790) and seen games (AUC = 0.850).
- Complaint associations are observational. An A/B test would be needed to establish retention or ROI impact from targeted responses.
- The archived source tables and JSON files retain their original numerical precision; the presentation and headline results use rounded values.

## My Contribution

Ruize Jin:

- Non-NLP data collection, merging, and cleaning.
- EDA including K-Means.
- Model selection based on 10-fold CV and out-of-sample metrics.
- Final model training.
- Analysis of NLP factors across the four player personas.

## Team

Ruize Jin · Jinyu Liu · Lucy Lu · Yiheng Huang · Zihan Zhang

Duke Fuqua · Data Science Final Project · Team 18

## Data Sources

- [Steam Reviews Dataset](https://www.kaggle.com/datasets/kieranpoc/steam-reviews)
- [SteamDB](https://steamdb.info/)

Source terms and availability apply. This repository does not redistribute source review records.

## Reproduction / Repository Structure

```text
src/           Existing data-preparation, modeling, evaluation, and persona code
scripts/       Existing NLP labeling, classifier, SQL, and analysis scripts
notebooks/     Three analysis notebooks; outputs cleared for public release
requirements/ Existing optional NLP dependency lists
reports/       Saved aggregate tables, figures, and analytical sections
results/       Lightweight aggregate metrics and frozen rules
run.py         Existing pipeline entry point
pyproject.toml Existing package configuration
requirements.txt Recorded dependencies for the original Python 3.12 environment
docs/steam-project.pptx Original final presentation
```

Code and notebook inputs are unchanged. Notebook outputs were cleared to avoid publishing local paths and embedded review-level data; important findings remain in the saved aggregate tables, figures, metrics, and presentation. Numerical result files were copied without modification. The original local source directory was not changed.

Use 64-bit Python 3.12 and install the recorded dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python run.py --help
```

Activate `.venv` using the command appropriate to your operating system before installing. Optional NLP dependencies are recorded under `requirements/`.

**This is a source-and-results release, not a self-contained data bundle.** To rerun the pipeline, obtain the public source data and rebuild the local inputs expected by the notebooks and code. `TEAM18_PROJECT_ROOT` can select a local project root. The retained pipeline stages are `prepare`, `cv`, `train`, `score`, `calibrate`, `diagnostics`, `segment`, `verify`, `persona`, and `verify-persona`; they require the relevant local datasets and, for scoring, trained model artifacts. Full training is computationally substantial.

The original `check --hash` command expects the complete private submission manifest and excluded data/model bundle. It is not a public-release smoke test. NLP training or inference additionally requires the relevant annotations or trained assets, which are not included here; LLM-labeling scripts also require the external CLI and schemas referenced by that original workflow. No end-to-end rerun is claimed for this public package.
