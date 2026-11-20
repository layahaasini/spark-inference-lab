#!/usr/bin/env bash
# GRPO on the tool-loop gym: miles + SGLang rollouts, one GB10.
# Run INSIDE the miles container, from /root/miles, with this repo mounted at /ws:
#   docker run --rm --gpus all --ipc=host --shm-size 16g \
#     -v ~/miles:/root/miles -v ~/opensource/spark-inference-lab:/ws \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -w /root/miles radixark/miles:latest bash /ws/scripts/run_gym_grpo.sh
#
# SMOKE=1 (default) stops after 2 rollouts to check the loop works end to end.
# SMOKE=0 runs the full 50 rollouts. LORA=0 (default) is full fine-tuning on
# the FSDP backend, which is what the writeup's run used. LORA=1 uses the
# Megatron backend + LoRA, which was broken on GB10 at the time (see below).
set -euo pipefail

SMOKE="${SMOKE:-1}"
LORA="${LORA:-0}"
# Megatron trainer attention: 'flash' uses flash-attn's CuTe-DSL kernel, which
# fails to compile on sm_121a; 'fused' = TE/cuDNN, 'unfused' = plain (slow)
ATTN_BACKEND="${ATTN_BACKEND:-fused}"
# 1e-5 collapsed the policy within 10 rollouts (response length 215 -> 7,
# reward -> 0). miles' own full fine-tune example uses 1e-6.
LR="${LR:-1e-6}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA="${DATA:-/ws/data/gym_train.jsonl}"
OUT="${OUT:-/ws/runs/p3/grpo_gym}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="/ws/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

python3 -m sparklab.p3.miles_env.build_dataset --out "$DATA" --rows 500

if [ "$LORA" = "1" ]; then
    # Megatron backend + LoRA. Broken on GB10 with miles:latest (Aug 2026):
    # the Megatron-bridge -> SGLang weight sync corrupts the weights and
    # rollouts come out as gibberish. Kept so it can be retested.
    MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py qwen2.5-1.5B)"
    read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
    # the preset leaves out rope_theta; Qwen2.5 uses 1e6, not Megatron's 10000
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
    # FSDP backend: full fine-tune, plain PyTorch, flash_attention_2 (works on GB10)
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
  --input-key prompt --label-key label --metadata-key metadata --tool-key tools \
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
  --optimizer adam --lr "$LR" --lr-decay-style constant \
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
