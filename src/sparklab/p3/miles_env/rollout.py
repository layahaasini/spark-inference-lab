"""miles custom generate function for the tool-loop gym.

Modeled on miles.rollout.generate_hub.multi_turn.generate, with the one change
that matters: each sample gets ITS OWN seeded ToolEnv, built from the dataset
row's metadata ({"task": template_name, "seed": n}), and the reward is set
inline from the environment's verifier when the episode ends — miles skips the
reward-model stage entirely when sample.reward is already set, so the verdict
travels with the trajectory and no cross-process state is needed.

Rollout economics: the per-request cached/prompt token counts from the engine's
responses are accumulated into sample.metadata["radix"] — P1's instrumentation
riding inside the training loop, which is the bridge number's official home.
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
        # bound to THIS episode's env — the reason this file exists
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
            # a response mangled enough to make the parser RAISE (seen once at
            # rollout ~40: malformed <tool_call> markup) is an episode outcome,
            # not an infrastructure error — same rule the env applies to bad
            # tool args; the verifier will score whatever state was reached
            break
        if len(tool_calls) == 0:
            break  # no tool call = the model considers itself done
        tool_messages = await execute_tool_calls(tool_calls, execute_tool)
        update_sample_with_tool_responses(sample, tool_messages, tokenizer=tokenizer)

    # verdict from the world, not the words — and it travels with the sample
    sample.reward = float(episode.verify(initial))
    sample.metadata["radix"] = {"prompt_tokens": prompt_toks, "cached_tokens": cached_toks,
                                "hit_rate": cached_toks / max(prompt_toks, 1)}
    sample.metadata["env_steps"] = episode.env.steps
    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=8)
    parser.add_argument("--generate-tool-call-parser", type=str, default="qwen25")


generate.add_arguments = _add_arguments
