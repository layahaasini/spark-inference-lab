"""miles custom generate function for the tool-loop gym.

Based on miles.rollout.generate_hub.multi_turn.generate. The difference: each
sample builds its own seeded ToolEnv from the row's metadata
({"task": name, "seed": n}), and the reward is set from the env's verifier when
the episode ends. miles skips the reward-model step when sample.reward is
already set.

The engine's per-request prompt/cached token counts are summed into
sample.metadata["radix"], which is how the training run's cache hit rate is
measured.
"""

import argparse
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import (
    create_tool_call_parser,
    execute_tool_calls,
    update_sample_with_tool_responses,
)
from miles.utils.http_utils import post

from ..tasks import TEMPLATES
from .tools import TOOL_SPECS


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = deepcopy(input.sample)
    tokenizer = input.state.tokenizer

    meta = sample.metadata or {}
    episode = TEMPLATES[meta["task"]](int(meta["seed"]))
    initial = episode.env.snapshot()

    async def execute_tool(name: str, params: dict) -> str:
        # runs against this sample's own env
        return episode.env.call(name, params)

    tool_call_parser = create_tool_call_parser(TOOL_SPECS, args.generate_tool_call_parser)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    sample.tokens = compute_prompt_ids_from_sample(input.state, sample, tools=TOOL_SPECS).copy()
    prompt_toks = cached_toks = 0

    for _turn in range(args.generate_max_turns):
        payload, halt_status = compute_request_payload(args, sample.tokens, input.sampling_params)
        if payload is None:
            sample.status = halt_status
            break
        output = await post(url, payload, headers=compute_routing_headers(args, sample))
        meta_info = output.get("meta_info", {})
        prompt_toks += meta_info.get("prompt_tokens", 0)
        cached_toks += meta_info.get("cached_tokens", 0)
        await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)
        if meta_info["finish_reason"]["type"] in ("abort", "length"):
            break
        try:
            _, tool_calls = tool_call_parser.parse_non_stream(output["text"])
        except Exception:
            # malformed <tool_call> markup made the parser raise once (rollout ~40).
            # Treat it as the end of the episode and score whatever state was reached.
            break
        if len(tool_calls) == 0:
            break  # no tool call = the model considers itself done
        tool_messages = await execute_tool_calls(tool_calls, execute_tool)
        update_sample_with_tool_responses(sample, tool_messages, tokenizer=tokenizer)

    sample.reward = float(episode.verify(initial))
    sample.metadata["radix"] = {"prompt_tokens": prompt_toks, "cached_tokens": cached_toks,
                                "hit_rate": cached_toks / max(prompt_toks, 1)}
    sample.metadata["env_steps"] = episode.env.steps
    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=8)
    parser.add_argument("--generate-tool-call-parser", type=str, default="qwen25")


generate.add_arguments = _add_arguments
