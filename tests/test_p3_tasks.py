"""Pin the gym's promises. All pure-python — runs everywhere, gates every push.

The one that matters most: every template's own lucky-guess trajectory must
score 0. A reward function whose hacks aren't tested is a reward function
whose hacks will be found — by the optimizer, mid-training, expensively.
"""

import pytest

from sparklab.p3.env import ToolEnv, is_done, parse_tool_call
from sparklab.p3.tasks import TEMPLATES

ALL = sorted(TEMPLATES)


@pytest.mark.parametrize("name", ALL)
def test_seeded_determinism(name):
    a, b = TEMPLATES[name](7), TEMPLATES[name](7)
    assert a.goal == b.goal and a.env.snapshot() == b.env.snapshot()
    assert TEMPLATES[name](8).env.snapshot() != a.env.snapshot() or TEMPLATES[name](8).goal != a.goal


@pytest.mark.parametrize("name", ALL)
def test_lucky_guess_scores_zero(name):
    ep = TEMPLATES[name](3)
    initial = ep.env.snapshot()
    ep.run_lucky_guess()
    assert ep.verify(initial) is False, f"{name}: its own documented hack scored 1"


@pytest.mark.parametrize("name", ALL)
def test_honest_solution_scores_one(name):
    # solve each task by direct state manipulation — the verifier must accept
    # a genuinely correct outcome, not just reject hacks
    ep = TEMPLATES[name](5)
    init = ep.env.snapshot()
    if name == "write_constant":
        key = ep.goal.split("'")[1]
        val = int(ep.goal.rstrip(".").split()[-1])
        ep.env.call("write", {"key": key, "value": val})
    elif name == "copy_value":
        src, dst = ep.goal.split("'")[1], ep.goal.split("'")[3]
        ep.env.call("write", {"key": dst, "value": init[src]})
    elif name == "increment_counter":
        key = ep.goal.split("'")[1]
        by = int(ep.goal.rstrip(".").split()[-1])
        ep.env.call("incr", {"key": key, "by": by})
    elif name == "sum_two_keys":
        a, b, dst = ep.goal.split("'")[1], ep.goal.split("'")[3], ep.goal.split("'")[5]
        ep.env.call("write", {"key": dst, "value": init[a] + init[b]})
    elif name == "largest_key":
        dst = ep.goal.split("'")[3]  # [1] is the 'k' prefix mention; [3] is the target
        ep.env.call("write", {"key": dst, "value": max(init.values())})
    elif name == "cleanup_threshold":
        thr = int(ep.goal.split("greater than ")[1].split(".")[0])
        for k, v in init.items():
            if v > thr:
                ep.env.call("delete", {"key": k})
    assert ep.verify(init) is True, f"{name}: honest solution rejected"


def test_difficulty_ladder_has_every_rung():
    diffs = {TEMPLATES[n](0).difficulty for n in ALL}
    assert diffs == {1, 2, 3}, "gradient signal must exist at every skill level"


def test_env_never_raises_on_garbage():
    env = ToolEnv.seeded(0)
    for name, args in [("read", {}), ("write", {"key": "k"}), ("nope", {}),
                       ("incr", {"key": "ghost"}), ("write", {"key": "k0", "value": "str"})]:
        out = env.call(name, args)
        assert isinstance(out, str)  # errors are observations, not crashes


def test_tool_call_parsing():
    assert parse_tool_call('thinking...\nTOOL: {"name": "read", "args": {"key": "k1"}}') == ("read", {"key": "k1"})
    assert parse_tool_call("TOOL: {broken json") == ("__malformed__", {})
    assert parse_tool_call("no call here") is None
    assert is_done("all set\nDONE") and not is_done("DONE-ish")
