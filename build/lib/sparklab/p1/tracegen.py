"""Synthetic agentic-trace generator.

Models how agent frameworks actually assemble context:

- one fixed system-prompt + tool-schema block -> a long prefix every session shares
- per-session history that only ever grows at the end (append-only rule: any edit
  above the tail kills the cached match, so real frameworks never do it)
- N sessions interleaved, competing for the same cache
- group mode (GRPO rollout shape): several samples share one full context and
  differ only in a tiny tail — the most cache-favorable workload there is
- Zipf mode: uneven session popularity. Even scheduling is the harder, honest
  test; heavy-tail is the realistic one. Reports should show both.

Token accounting caveat: "tokens" here are whitespace words, which only roughly
track tokenizer counts. That is fine because every published number (hit rate,
tokens avoided) comes from server-reported token counts at replay time; the
generator's counts only set approximate sizes.
"""

import argparse
import json
import pathlib
import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TraceConfig:
    seed: int = 0
    n_sessions: int = 4
    turns: int = 6                     # per session (average per session in zipf mode)
    sys_tokens: int = 800              # size of the shared system+tools prefix
    tool_tokens: tuple = (50, 400)     # per-turn tool-result size range
    group_size: int = 1                # >1 switches to GRPO group-sampling shape
    zipf_alpha: float = 0.0            # 0 = every session acts each round; >0 = heavy-tail scheduling
    # word budget per session (~5x that in tokens): sessions at the cap stop
    # receiving turns, the way real frameworks stop at the model's context limit.
    # None = unbounded; REQUIRED in zipf mode against a real server, where the
    # hot session otherwise outgrows context_len.
    max_session_words: int = None


def _words(rng, n, tag):
    return " ".join(f"{tag}{rng.randint(0, 9999)}" for _ in range(n))


def shared_prefix(cfg):
    """The fixed head every session starts with. Deterministic given (seed, sys_tokens):
    uses a fresh generator in the same draw order as gen_trace, so both agree."""
    rng = random.Random(cfg.seed)
    return "SYS: " + _words(rng, cfg.sys_tokens, "t")


def gen_trace(cfg=None, **kwargs):
    """Build one trace: a list of replayable events, in arrival order.

    Each event is {"session", "turn", "sample", "prompt"} where prompt is the
    full context the agent framework would send at that moment.

    Scheduling: even mode gives every session one turn per shuffled round; zipf
    mode draws the next session per-event with weight 1/rank^alpha. Sessions at
    the max_session_words budget stop receiving turns (their weight share is
    redistributed by the draw), so total event count is preserved unless every
    session is capped.
    """
    if cfg is None:
        cfg = TraceConfig(**kwargs)
    elif kwargs:
        raise TypeError("pass either a TraceConfig or keyword overrides, not both")
    rng = random.Random(cfg.seed)
    sysp = "SYS: " + _words(rng, cfg.sys_tokens, "t")
    n = cfg.n_sessions
    sessions = {s: sysp + f"\n[session {s}]" for s in range(n)}
    turn_of = {s: 0 for s in range(n)}
    words_of = {s: cfg.sys_tokens for s in range(n)}
    events = []

    def eligible(s):
        return cfg.max_session_words is None or words_of[s] < cfg.max_session_words

    def emit_turn(s):
        t = turn_of[s]
        turn_of[s] += 1
        n_tool = rng.randint(*cfg.tool_tokens)
        words_of[s] += n_tool + 4  # tool words plus the turn-header words
        # append-only: a turn adds a user step and a tool result at the end, never edits above
        sessions[s] += f"\nUSER: step {t}\nTOOL: " + _words(rng, n_tool, "r")
        for g in range(cfg.group_size):
            sfx = f"\nSAMPLE[{g}]" if cfg.group_size > 1 else ""
            events.append({"session": s, "turn": t, "sample": g, "prompt": sessions[s] + sfx})

    if cfg.zipf_alpha > 0:
        weights = [1 / (i + 1) ** cfg.zipf_alpha for i in range(n)]
        for _ in range(n * cfg.turns):
            live = [s for s in range(n) if eligible(s)]
            if not live:
                break
            emit_turn(rng.choices(live, weights=[weights[s] for s in live], k=1)[0])
    else:
        for _ in range(cfg.turns):
            for s in rng.sample(range(n), n):
                if eligible(s):
                    emit_turn(s)
    return events


def save_trace(events, path, cfg=None):
    path = pathlib.Path(path)
    path.write_text(json.dumps({"config": asdict(cfg) if cfg else None, "events": events}))
    return path


def load_trace(path):
    return json.loads(pathlib.Path(path).read_text())["events"]


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic agentic trace")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--sys-tokens", type=int, default=800)
    ap.add_argument("--tool-tokens", type=int, nargs=2, default=(50, 400), metavar=("MIN", "MAX"))
    ap.add_argument("--group-size", type=int, default=1)
    ap.add_argument("--zipf-alpha", type=float, default=0.0)
    ap.add_argument("--max-session-words", type=int, default=None)
    ap.add_argument("--out", default="trace.json")
    a = ap.parse_args()
    cfg = TraceConfig(seed=a.seed, n_sessions=a.sessions, turns=a.turns, sys_tokens=a.sys_tokens,
                      tool_tokens=tuple(a.tool_tokens), group_size=a.group_size, zipf_alpha=a.zipf_alpha,
                      max_session_words=a.max_session_words)
    events = gen_trace(cfg)
    save_trace(events, a.out, cfg)
    print(f"{len(events)} events -> {a.out}")


if __name__ == "__main__":
    main()
