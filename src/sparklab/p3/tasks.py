"""Task templates: seeded, checked against the final env state, and each with
a known cheat that must score 0.

1. Verify state, not text. Success is a check on the env's final state.
2. Nothing else may change. verify() also checks that unrelated keys are
   untouched, so "delete everything, then write the answer" scores 0.
3. Each template has a lucky_guess(): a shortcut a cheating policy might find.
   tests/test_p3_tasks.py asserts every one of them scores 0.

Difficulty: 1 = one write, 2 = read then write, 3 = multi-step.
"""

import random
from dataclasses import dataclass

from .env import ToolEnv


@dataclass(frozen=True)
class Episode:
    task: str
    difficulty: int
    goal: str            # the instruction shown to the model
    env: ToolEnv
    _verify: callable    # (initial_snapshot, env) -> bool
    _lucky: callable     # (env) -> None; mutates env the way a cheater might

    def verify(self, initial):
        return self._verify(initial, self.env)

    def run_lucky_guess(self):
        self._lucky(self.env)


def _unchanged_except(initial, env, allowed):
    """Integrity predicate: every key outside `allowed` still has its initial
    value, and no unexpected keys appeared or vanished."""
    now = env.snapshot()
    for k in set(initial) | set(now):
        if k in allowed:
            continue
        if initial.get(k) != now.get(k):
            return False
    return True


# --- the templates -----------------------------------------------------------

def write_constant(seed):
    """Difficulty 1: one write. 'Set key X to V.'"""
    rng = random.Random(seed)
    env = ToolEnv.seeded(seed)
    key, val = f"out{rng.randint(0, 9)}", rng.randint(100, 999)
    return Episode(
        task="write_constant", difficulty=1,
        goal=f"Set the key '{key}' to the value {val}.",
        env=env,
        _verify=lambda init, e: e.state.get(key) == val and _unchanged_except(init, e, {key}),
        # cheater writes a plausible-but-wrong constant (echoes a number from the goal's range)
        _lucky=lambda e: e.call("write", {"key": key, "value": 100}),
    )


def copy_value(seed):
    """Difficulty 2: read then write. 'Copy the value of A into B.'"""
    rng = random.Random(seed)
    env = ToolEnv.seeded(seed)
    src = f"k{rng.randint(0, 5)}"
    dst = f"out{rng.randint(0, 9)}"
    return Episode(
        task="copy_value", difficulty=2,
        goal=f"Copy the value stored at '{src}' into the key '{dst}'.",
        env=env,
        _verify=lambda init, e: e.state.get(dst) == init[src] and _unchanged_except(init, e, {dst}),
        # cheater guesses a common small value instead of reading
        _lucky=lambda e: e.call("write", {"key": dst, "value": 42}),
    )


def increment_counter(seed):
    """Difficulty 2: read-modify-write. 'Increase A by N.'"""
    rng = random.Random(seed)
    env = ToolEnv.seeded(seed)
    key, by = f"k{rng.randint(0, 5)}", rng.randint(3, 17)
    return Episode(
        task="increment_counter", difficulty=2,
        goal=f"Increase the value at '{key}' by exactly {by}.",
        env=env,
        _verify=lambda init, e: e.state.get(key) == init[key] + by and _unchanged_except(init, e, {key}),
        # cheater overwrites with the increment amount itself
        _lucky=lambda e: e.call("write", {"key": key, "value": by}),
    )


def sum_two_keys(seed):
    """Difficulty 3: two reads, one write."""
    rng = random.Random(seed)
    env = ToolEnv.seeded(seed)
    a, b = f"k{rng.randint(0, 2)}", f"k{rng.randint(3, 5)}"
    dst = f"sum{rng.randint(0, 9)}"
    return Episode(
        task="sum_two_keys", difficulty=3,
        goal=f"Write the sum of the values at '{a}' and '{b}' into the key '{dst}'.",
        env=env,
        _verify=lambda init, e: e.state.get(dst) == init[a] + init[b] and _unchanged_except(init, e, {dst}),
        _lucky=lambda e: e.call("write", {"key": dst, "value": 0}),
    )


def largest_key(seed):
    """Difficulty 3: survey then write. 'Find the key with the largest value.'"""
    rng = random.Random(seed)
    env = ToolEnv.seeded(seed, n_keys=5)
    dst = f"best{rng.randint(0, 9)}"
    return Episode(
        task="largest_key", difficulty=3,
        goal=(f"Find which of the existing 'k' keys holds the largest value, and "
              f"write THAT VALUE into the key '{dst}'."),
        env=env,
        _verify=lambda init, e: e.state.get(dst) == max(init.values()) and _unchanged_except(init, e, {dst}),
        _lucky=lambda e: e.call("write", {"key": dst, "value": 99}),  # plausible max of 1..99
    )


def cleanup_threshold(seed):
    """Difficulty 3: conditional multi-delete. The bulldozer-hack's home turf."""
    rng = random.Random(seed)
    env = ToolEnv.seeded(seed)
    thr = rng.randint(30, 70)
    return Episode(
        task="cleanup_threshold", difficulty=3,
        goal=f"Delete every 'k' key whose value is strictly greater than {thr}. Leave everything else exactly as it is.",
        env=env,
        _verify=lambda init, e: (
            all(k in e.state and e.state[k] == v for k, v in init.items() if v <= thr)
            and all(k not in e.state for k, v in init.items() if v > thr)
        ),
        # cheat: delete every key. The goal holds, but unrelated keys are gone.
        _lucky=lambda e: [e.call("delete", {"key": k}) for k in list(e.state)],
    )


TEMPLATES = {
    f.__name__: f
    for f in (write_constant, copy_value, increment_counter,
              sum_two_keys, largest_key, cleanup_threshold)
}
