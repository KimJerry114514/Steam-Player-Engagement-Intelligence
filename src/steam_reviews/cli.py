"""Portable submission entry point. New calculations are written under runs/."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from steam_reviews.paths import ROOT, CALIBRATION as CAL, SEGMENTATION as SEG, CLEANED


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check(hash_all=False):
    import pyarrow.parquet as pq
    import lightgbm as lgb
    manifest = json.loads((ROOT / 'docs/provenance/submission_manifest.json').read_text())
    for entry in manifest['files']:
        p = ROOT / entry['path']
        if not p.is_file() or p.stat().st_size != entry['bytes']:
            raise ValueError(f'Missing or incomplete submission file: {p}')
        if hash_all and sha256(p) != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {p}')
    for p in [* (ROOT / 'src').rglob('*.py'), * (ROOT / 'scripts').rglob('*.py'), ROOT / 'run.py']:
        if 'runs' not in p.relative_to(ROOT).parts:
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    for p in (ROOT / 'notebooks').glob('*.ipynb'):
        if 'runs' in p.relative_to(ROOT).parts:
            continue
        for c in json.loads(p.read_text())['cells']:
            if c['cell_type'] == 'code':
                ast.parse(''.join(c['source']), filename=str(p))
    meta = json.loads((ROOT / 'models/lightgbm/training_full_model.json').read_text())
    model = ROOT / 'models/lightgbm/training_full_model.txt'
    frozen = json.loads((CAL / 'frozen_rules.json').read_text())
    assert sha256(model) == meta['model_txt_sha256'] == frozen['model']['sha256']
    booster = lgb.Booster(model_file=str(model))
    assert booster.num_trees() == meta['rounds_trained'] == 1000
    assert booster.num_feature() == len(meta['transformed_features']) == 96
    n = pq.ParquetFile(CAL / 'predictions_val_test_calibrated.parquet').metadata.num_rows
    assert n == pq.ParquetFile(SEG / 'review_segments_p55.parquet').metadata.num_rows == 4937916
    assert pq.ParquetFile(CLEANED).metadata.num_rows == 16412862
    rules = json.loads((SEG / 'selected_segmentation_rules.json').read_text())
    assert rules['prior_percentile'] == rules['score_percentile'] == 55
    assert rules['prior_cutoff_hours'] == 11.78
    persona_data = ROOT / 'data/processed/segmentation/review_segments_all_history.parquet'
    persona_rows = pq.ParquetFile(persona_data).metadata.num_rows
    assert persona_rows == 16412862
    persona = json.loads((ROOT / 'results/user_persona/analysis_manifest.json').read_text())
    assert persona['primary_scope'] == 'all_history_2010_2023'
    assert persona['checks']['factor_count'] == 35 and persona['checks']['domain_count'] == 8
    assert persona['checks']['unmatched_nlp_records'] == 0
    verification = json.loads((ROOT / 'results/user_persona/verification.json').read_text())
    assert verification['status'] == 'passed' and verification['records_checked'] == persona_rows
    print(f'PASS: {len(manifest["files"])} files; model: 96 features / 1000 trees; '
          f'{n:,} validation/test segment records; {persona_rows:,} historical persona records; P55/P55.')
    print('All SHA-256 checks passed.' if hash_all else 'Use check --hash for every file checksum.')


def run_module(module, *args):
    subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=ROOT, check=True)


def notebook(path):
    output = ROOT / 'runs/notebooks'
    output.mkdir(parents=True, exist_ok=True)
    # Bind execution to this command's Python, independent of the author's kernels.
    jupyter_data = ROOT / 'runs/jupyter'
    kernel_dir = jupyter_data / 'kernels/steam-team18-runtime'
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Steam Team 18 runtime', 'language': 'python',
    }))
    env = os.environ.copy()
    env.setdefault('IPYTHONDIR', str(ROOT / 'runs/ipython'))
    env.setdefault('JUPYTER_RUNTIME_DIR', str(ROOT / 'runs/jupyter_runtime'))
    env['JUPYTER_PATH'] = str(jupyter_data) + os.pathsep + env.get('JUPYTER_PATH', '')
    subprocess.run([sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute',
                    '--ExecutePreprocessor.timeout=-1',
                    '--ExecutePreprocessor.kernel_name=steam-team18-runtime',
                    '--output-dir', str(output), str(ROOT / path)],
                   cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['check', 'prepare', 'cv', 'train', 'score',
                                          'calibrate', 'diagnostics', 'segment', 'verify',
                                          'persona', 'verify-persona'])
    parser.add_argument('--hash', action='store_true', help='check every packaged checksum')
    parser.add_argument('--bundle', type=Path, default=CAL, help='calibration directory or original ZIP')
    parser.add_argument('--result-dir', type=Path, help='segmentation/persona output, or result directory to verify')
    parser.add_argument('--rebuild-segments', action='store_true',
                        help='with persona, regenerate historical scores instead of using the verified cache')
    args = parser.parse_args()
    os.environ.setdefault('TEAM18_PROJECT_ROOT', str(ROOT))
    os.environ['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runs/matplotlib'))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    if args.step == 'check':
        check(args.hash)
    elif args.step in {'prepare', 'cv', 'train'}:
        notebook({'prepare': 'notebooks/01_data_preparation_and_eda.ipynb',
                  'cv': 'notebooks/02_model_comparison_cv.ipynb',
                  'train': 'notebooks/03_final_model_training.ipynb'}[args.step])
    elif args.step in {'score', 'calibrate', 'diagnostics'}:
        scripts = {'score': ['score_val_test.py'], 'calibrate': ['freeze_rules.py', 'evaluate_test.py'],
                   'diagnostics': ['supplement_diagnostics.py']}[args.step]
        for name in scripts:
            run_module('steam_reviews.evaluation.' + Path(name).stem)
    elif args.step == 'segment':
        dest = args.result_dir or ROOT / 'runs/segmentation'
        run_module('steam_reviews.segmentation.segment_p55', '--bundle', args.bundle,
                   '--training-data', CLEANED, '--output-dir', dest)
    elif args.step == 'verify':
        dest = args.result_dir or SEG
        run_module('steam_reviews.segmentation.verify_p55', '--bundle', args.bundle, '--result-dir', dest,
                   '--report-path', ROOT / 'runs/independent_verification.json')
    elif args.step == 'persona':
        dest = args.result_dir or ROOT / 'runs/user_persona'
        extra = ['--rebuild-segments'] if args.rebuild_segments else []
        run_module('steam_reviews.persona.build_complaint_profiles', '--output-dir', dest, *extra)
    elif args.step == 'verify-persona':
        extra = ['--output-dir', args.result_dir] if args.result_dir else []
        run_module('steam_reviews.persona.verify_complaint_profiles', *extra)


if __name__ == '__main__':
    main()
