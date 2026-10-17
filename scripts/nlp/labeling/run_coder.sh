#!/bin/bash
# run_coder.sh <stratum> <batch_path> <n>  -> out/<basename>.json ; uses LLM CLI (user's LLM quota), read-only sandbox
C=llm-cli
S="$1"; P="$2"; N="$3"; B=$(basename "$P" .json)
[ -s "out/$B.json" ] && exit 0
PROMPT=$(sed -e "s|__PATH__|$P|" -e "s|__N__|$N|" -e "s|__STRATUM__|$S|" prompt_coder.txt)
start=$(date +%s)
$C exec --ignore-user-config --skip-git-repo-check --ephemeral -s read-only -C . \
  -m llm-model-a -c model_reasoning_effort=\"medium\" \
  --output-schema schemas/coder.json -o "out/$B.tmp.json" "$PROMPT" < /dev/null > "logs/$B.log" 2>&1
rc=$?
if [ $rc -eq 0 ] && python3 -c "import json,sys;json.load(open('out/$B.tmp.json'))" 2>/dev/null; then mv "out/$B.tmp.json" "out/$B.json"; fi
echo "$B rc=$rc secs=$(( $(date +%s)-start ))" >> logs/_summary.log
