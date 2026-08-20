#!/usr/bin/env bash
# GRPO on the tool-loop gym: miles + SGLang rollouts, one GB10.
# Run INSIDE the miles container, from /root/miles, with this repo mounted at /ws:
#   docker run --rm --gpus all --ipc=host --shm-size 16g \
#     -v ~/miles:/root/miles -v ~/opensource/spark-inference-lab:/ws \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -w /root/miles radixark/miles:latest bash /ws/scripts/run_gym_grpo.sh
#
# SMOKE=1 (default) exits after 2 rollout steps — the "does the loop close"
# test. SMOKE=0 runs the real thing. LORA=0 falls back to full fine-tune on
# the FSDP backend (the Megatron/TE path may not carry sm_121a kernels).
set -euo pipefail

SMOKE="${SMOKE:-1}"
LORA="${LORA:-1}"
# trainer attention: 'flash' routes to flash-attn's CuTe-DSL fwd, which fails
# MLIR compilation on sm_121a (layout congruence, flash_fwd.py epilogue);
# 'fused' = TE/cuDNN, 'unfused' = plain softmax (slow, bulletproof)
ATTN_BACKEND="${ATTN_BACKEND:-fused}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA="${DATA:-/ws/data/gym_train.jsonl}"
OUT="${OUT:-/ws/runs/grpo_gym}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="/ws/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

python3 -m sparklab.p3.miles_env.build_dataset --out "$DATA" --rows 500

if [ "$LORA" = "1" ]; then
    # Megatron backend + LoRA. KNOWN BROKEN on GB10 as of miles:latest
    # (2026-08): the pre-rollout Megatron-bridge -> engine weight sync
    # corrupts weights (healthy engine, gibberish rollouts). Kept for
    # retesting; use LORA=0 until fixed upstream.
    MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py qwen2.5-1.5B)"
    read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
    # the preset emits no rope flag; Qwen2.5 uses rope_theta 1e6 and
    # miles' hf_validate_args rightly refuses Megatron's default 10000
    MODEL_ARGS+=(--rotary-base 1000000)
    BACKEND_ARGS=("${MODEL_ARGS[@]}"
                  --lora-rank 32 --lora-alpha 32 --lora-dropout 0.0
                  --target-modules all-linear --megatron-to-hf-mode bridge
                  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
                  --context-parallel-size 1
                  --attention-dropout 0.0 --hidden-dropout 0.0
                  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
                  --attention-backend "$ATTN_BACKEND")
else
    # FSDP backend: full fine-tune, torch-native, no Megatron/TE in the
    # trainer path; attn via classic flash_attention_2 (verified on GB10)
    BACKEND_ARGS=(--train-backend fsdp
                  --load "$MODEL" --ref-load "$MODEL"
                  --gradient-checkpointing
                  --attn-implementation flash_attention_2
                  --update-weight-buffer-size 536870912)
fi
SMOKE_ARGS=()
[ "$SMOKE" = "1" ] && SMOKE_ARGS=(--debug-exit-after-rollout 2)

ray stop --force 2>/dev/null; sleep 2
ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"env_vars":{"PYTHONPATH":"/root/Megatron-LM:/root/miles:/ws/src","CUDA_DEVICE_MAX_CONNECTIONS":"1"}}' \
  -- python3 train.py \
  --hf-checkpoint "$MODEL" \
  "${BACKEND_ARGS[@]}" \
  --save "$OUT/ckpt" --save-interval 20 \
  \
  --custom-generate-function-path sparklab.p3.miles_env.rollout.generate \
  --generate-tool-call-parser qwen25 \
  --generate-max-turns 8 \
  --log-multi-turn \
  \
  --prompt-data "$DATA" \
  --input-key prompt --label-key label --metadata-key metadata \
  --apply-chat-template --rollout-shuffle --rollout-seed 42 \
  \
  --num-rollout 50 \
  --rollout-batch-size 8 --n-samples-per-prompt 8 --global-batch-size 64 \
  --rollout-max-response-len 1024 --rollout-temperature 1.0 \
  --rollout-max-prompt-len 2048 \
  \
  --advantage-estimator grpo \
  --eps-clip 0.2 --eps-clip-high 0.28 \
  --kl-loss-coef 0.0 --kl-coef 0.0 --entropy-coef 0.0 \
  \
  --optimizer adam --lr 1e-5 --lr-decay-style constant \
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  \
  --use-dynamic-batch-size --max-tokens-per-gpu 8192 \
  \
  --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.35 \
  \
  --actor-num-nodes 1 --actor-num-gpus-per-node 1 --num-gpus-per-node 1 \
  --colocate \
  \
  --use-tensorboard --tb-project-name gym-grpo --tb-experiment-name "$(date +%m%d-%H%M)" \
  --dump-details "$OUT/dumps" \
  "${SMOKE_ARGS[@]}"
