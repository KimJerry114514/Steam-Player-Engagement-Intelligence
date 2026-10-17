"""Generic resumable LLM CLI batch runner.

jobs file: JSON list of {"id", "prompt", "schema", "model", "effort", "out"}.
Each job -> `llm exec` in a read-only sandbox with a strict output schema; result validated as JSON and
atomically moved into place. Done jobs (existing non-empty out) are skipped; failures retried.
Usage: python llm_run.py jobs.json --parallel 6 [--retries 2]
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

LLM_CLI = "llm-cli"
ROOT = "."
HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, "logs")
os.makedirs(LOGDIR, exist_ok=True)


def run_one(job, retries):
    out = job["out"]
    if os.path.exists(out) and os.path.getsize(out) > 2:
        return job["id"], "skip", 0
    tmp = out + ".tmp"
    for attempt in range(retries + 1):
        t0 = time.time()
        cmd = [LLM_CLI, "exec", "--ignore-user-config", "--skip-git-repo-check", "--ephemeral", "-s", "read-only",
               "-C", ROOT, "-m", job.get("model", "llm-model-a"),
               "-c", f'model_reasoning_effort="{job.get("effort", "medium")}"',
               "--output-schema", job["schema"], "-o", tmp, job["prompt"]]
        with open(os.path.join(LOGDIR, f"{job['id']}.log"), "w") as lf, open(os.devnull) as devnull:
            rc = subprocess.run(cmd, stdin=devnull, stdout=lf, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        try:
            json.load(open(tmp))
            os.replace(tmp, out)
            status = "ok"
        except Exception:
            status = f"bad(rc={rc})"
        with open(os.path.join(LOGDIR, "_runner.log"), "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {job['id']} {status} attempt={attempt} secs={dt:.0f}\n")
        if status == "ok":
            return job["id"], "ok", dt
    return job["id"], "failed", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    res = {"ok": 0, "skip": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        futs = [ex.submit(run_one, j, a.retries) for j in jobs]
        for f in as_completed(futs):
            jid, st, dt = f.result()
            res[st] += 1
            print(f"{jid} {st} {dt:.0f}s | ok={res['ok']} skip={res['skip']} failed={res['failed']} / {len(jobs)}", flush=True)
    print("DONE", res)
    sys.exit(1 if res["failed"] else 0)


if __name__ == "__main__":
    main()
