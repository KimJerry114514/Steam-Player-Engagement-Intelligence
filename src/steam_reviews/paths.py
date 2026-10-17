"""Portable project paths shared by command-line stages."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("TEAM18_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models" / "lightgbm"
CALIBRATION = ROOT / "results" / "calibration"
SEGMENTATION = ROOT / "results" / "segmentation"
RUNS = ROOT / "runs"
CLEANED = DATA / "processed" / "negative_reviews_cleaned.parquet"
MERGED = DATA / "processed" / "negative_reviews_merged.parquet"
NLP = DATA / "processed" / "nlp" / "nlp_factor_scores_2023.parquet"
