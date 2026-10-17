"""Collect the classifier outputs (thresholds, metrics, per-review scores written by w20_finetune.py) and build a 0/1
factor matrix (rid + one int8 column per factor) with the frozen per-factor thresholds. Env MODEL_DIR selects the model
folder (ft = fine-tuned, model_pass1 = linear probe); env GPU_OUT points to the folder w20_finetune.py wrote to.
Usage: MODEL_DIR=ft python 21_factor_matrix.py <out.parquet> [sets: focus sample rest]"""
import sys, json, glob, os, shutil, duckdb
ROOT = "."; MD = os.environ.get("MODEL_DIR", "ft")
GPU_OUT = os.environ.get("GPU_OUT", f"{ROOT}/gpu_work/{MD}")
LOC = f"{ROOT}/results/scores_{MD}"; os.makedirs(LOC, exist_ok=True)
out = sys.argv[1]; sets = sys.argv[2:] or ["focus", "sample"]
for f in ("thresholds.json", "metrics.json", "test_predictions.parquet"):
    shutil.copy(f"{GPU_OUT}/{f}", f"{ROOT}/results/{MD}_{f}")
for s in sets:
    for p in glob.glob(f"{GPU_OUT}/scores/{s}_*.parquet"):
        shutil.copy(p, LOC + "/")
th = json.load(open(f"{ROOT}/results/{MD}_thresholds.json"))["thresholds"]
files = sorted(f for s in sets for f in glob.glob(f"{LOC}/{s}_*.parquet"))
cols = ", ".join(f'("{f}" >= {int(round(t * 255))})::INT8 AS "{f}"' for f, t in th.items())
con = duckdb.connect(); con.execute("SET threads=8")
con.execute(f"COPY (SELECT DISTINCT ON (rid) rid, {cols} FROM read_parquet({files})) TO '{out}' (FORMAT PARQUET)")
print(out, con.execute(f"SELECT count(*) FROM '{out}'").fetchone()[0], "rows,", len(th), "factors from", len(files), "files")
