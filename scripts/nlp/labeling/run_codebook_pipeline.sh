#!/bin/bash
# Chain the LLM stages after open coding: synth -> arch -> judge -> critic -> revise. Resumable (done jobs are skipped).
cd ./llm_jobs
PY=./.venv/bin/python
set -o pipefail
for stage in synth arch judge critic revise; do
  par=6; [ "$stage" = "arch" ] && par=3; [ "$stage" = "critic" ] && par=3
  echo "$(date +%H:%M:%S) START $stage" >> logs/_pipeline.log
  $PY build_jobs.py $stage >> logs/_pipeline.log 2>&1 || { echo "build $stage failed" >> logs/_pipeline.log; exit 1; }
  $PY llm_run.py jobs_$stage.json --parallel $par > logs/stage_$stage.out 2>&1 || { echo "$(date +%H:%M:%S) FAIL $stage" >> logs/_pipeline.log; exit 1; }
  echo "$(date +%H:%M:%S) DONE $stage" >> logs/_pipeline.log
done
echo "$(date +%H:%M:%S) START label+label2" >> logs/_pipeline.log
$PY build_jobs.py label >> logs/_pipeline.log 2>&1 && $PY build_jobs.py label2 >> logs/_pipeline.log 2>&1 || { echo "build label failed" >> logs/_pipeline.log; exit 1; }
$PY llm_run.py jobs_label.json --parallel 6 > logs/stage_label.out 2>&1 &
$PY llm_run.py jobs_label2.json --parallel 2 > logs/stage_label2.out 2>&1 &
wait
echo "$(date +%H:%M:%S) PIPELINE DONE" >> logs/_pipeline.log
