"""The tool environment: a tiny key-value store the model manipulates through
tool calls.

- state is a plain dict, seeded, so every episode can be replayed
- tools return strings (what the model sees); success is judged on the final
  state, never on what the model says
- transcripts are append-only, the same shape as the P1 traces, which is why
  rollouts get high prefix-cache hit rates
"""

import json
import random
from dataclasses import dataclass, field

TOOL_DOCS = """You can call tools by writing exactly one line:
TOOL: {"name": "<tool>", "args": {...}}

Available tools:
  read   {"key": k}            -> the value stored at k, or "MISSING"
  write  {"key": k, "value": v} -> stores v at k (v must be an integer), says "OK"
  incr   {"key": k, "by": n}    -> adds n to the integer at k, says the new value
  delete {"key": k}             -> removes k, says "OK"
  keys   {}                     -> the sorted list of all keys

Example: to store the number 5 at key "x", write exactly this line:
TOOL: {"name": "write", "args": {"key": "x", "value": 5}}

Call one tool per turn. When the task is complete, write exactly: DONE
"""


@dataclass
class ToolEnv:
    state: dict = field(default_factory=dict)
    steps: int = 0

    @classmethod
    def seeded(cls, seed, n_keys=6, prefix="k"):
        """Deterministic initial state: n_keys keys with small integer values."""
        rng = random.Random(seed)
        return cls(state={f"{prefix}{i}": rng.randint(1, 99) for i in range(n_keys)})

    def snapshot(self):
        return dict(self.state)

    def call(self, name, args):
        """Execute one tool call, return the observation string the model sees.
        Bad calls return error strings — they never raise: a confused model is
        an episode outcome, not a crash."""
        self.steps += 1
        try:
            if name == "read":
                return str(self.state.get(args["key"], "MISSING"))
            if name == "write":
                if not isinstance(args["value"], int):
                    return "ERROR: value must be an integer"
                self.state[args["key"]] = args["value"]
                return "OK"
            if name == "incr":
                if args["key"] not in self.state:
                    return "MISSING"
                self.state[args["key"]] += int(args.get("by", 1))
                return str(self.state[args["key"]])
            if name == "delete":
                self.state.pop(args["key"], None)
                return "OK"
            if name == "keys":
                return json.dumps(sorted(self.state))
            return f"ERROR: unknown tool {name!r}"
        except (KeyError, TypeError, ValueError) as e:
            return f"ERROR: bad args ({e})"


def parse_tool_call(text):
    """Find the first tool call in model output. Returns (name, args) or None.

    Accepts the 'TOOL: {...}' line, a bare JSON object with a "name" field, or
    'WRITE {...}' (tool name + JSON args). The 1.5B model often made correct
    calls in these near-miss formats, and rejecting them measured formatting
    instead of whether it could do the task. Reward still comes only from state."""
    known = {"read", "write", "incr", "delete", "keys"}
    for line in text.splitlines():
        line = line.strip().strip("`")
        if not line:
            continue
        if line.startswith("TOOL:"):
            try:
                obj = json.loads(line[len("TOOL:"):].strip())
                return obj["name"], obj.get("args", obj.get("arguments", {}))
            except (json.JSONDecodeError, KeyError, TypeError):
                return "__malformed__", {}
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "name" in obj:
                    return obj["name"], obj.get("args", obj.get("arguments", {}))
            except json.JSONDecodeError:
                pass
        head, _, rest = line.partition(" ")
        if head.lower() in known and rest.strip().startswith("{"):
            try:
                return head.lower(), json.loads(rest.strip())
            except json.JSONDecodeError:
                return "__malformed__", {}
    return None


def is_done(text):
    return any(line.strip() == "DONE" for line in text.splitlines())
