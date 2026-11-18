#!/usr/bin/env bash
# Run the LLM experiments of the paper on the 10-metric dataset, then print the model-to-model comparisons.
#
# Needs a running Ollama server with the models pulled, and OPENAI_API_KEY for GPT-4 Turbo.
# Each model is run one after the other (one process on the GPU at a time, as in the paper).
#
#   bash scripts/run_paper_experiments.sh            # all experiments
#   SKIP_OPENAI=1 bash scripts/run_paper_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${DATASET:-cloud_edge_kwollect_10}
OLLAMA_MODELS=${OLLAMA_MODELS:-"llama3.3:70b mistral gemma3:27b mixtral:8x7b deepseek-r1:70b"}
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4-1106-preview}
LOGDIR=${LOGDIR:-./logs}
mkdir -p "$LOGDIR"

run() {  # run <backend> <model> <extra args...>
  local backend=$1 model=$2; shift 2
  echo ">>> $backend $model $*"
  python run.py --dataset "$DATASET" --backend "$backend" --model "$model" --logdir "$LOGDIR" "$@" \
    > "$LOGDIR/last_run.log" 2>&1 || { echo "failed, see $LOGDIR/last_run.log"; exit 1; }
}

models=()
for m in $OLLAMA_MODELS; do models+=("ollama $m"); done
[[ -z "${SKIP_OPENAI:-}" ]] && models+=("openai $OPENAI_MODEL")

for entry in "${models[@]}"; do
  read -r backend model <<< "$entry"
  for relations in direct all; do                       # Table II (direct) and Table III (direct + indirect)
    run "$backend" "$model" --alg llm_pairwise --relations "$relations"
    run "$backend" "$model" --alg llm_batch --relations "$relations" --batch_size 4
  done
  for bs in 7 10; do                                    # batch-size study (Table VII)
    run "$backend" "$model" --alg llm_batch --relations all --batch_size "$bs"
  done
done

echo
echo "== Pairwise vs batch (batch size 4), direct and indirect relations"
python scripts/compare_graphs.py "$LOGDIR/$DATASET"/llm_pairwise_*_all.npy
python scripts/compare_graphs.py "$LOGDIR/$DATASET"/llm_batch_*_all_bs4.npy
