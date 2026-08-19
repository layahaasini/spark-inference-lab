"""OpenAI-format tool specs for the gym's five tools.

These are the specs handed to the model's chat template (native Qwen tool
calling) during miles rollouts. Execution does NOT go through a global
executor — the rollout function binds each episode's ToolEnv instance
instead, because the environments are stateful and per-sample (a global
executor would let group samples trample each other's state)."""

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "read",
        "description": "Read the value stored at a key. Returns the value, or MISSING.",
        "parameters": {"type": "object",
                       "properties": {"key": {"type": "string"}},
                       "required": ["key"]}}},
    {"type": "function", "function": {
        "name": "write",
        "description": "Store an integer value at a key. Returns OK.",
        "parameters": {"type": "object",
                       "properties": {"key": {"type": "string"},
                                      "value": {"type": "integer"}},
                       "required": ["key", "value"]}}},
    {"type": "function", "function": {
        "name": "incr",
        "description": "Add an integer to the value at a key. Returns the new value.",
        "parameters": {"type": "object",
                       "properties": {"key": {"type": "string"},
                                      "by": {"type": "integer"}},
                       "required": ["key", "by"]}}},
    {"type": "function", "function": {
        "name": "delete",
        "description": "Remove a key. Returns OK.",
        "parameters": {"type": "object",
                       "properties": {"key": {"type": "string"}},
                       "required": ["key"]}}},
    {"type": "function", "function": {
        "name": "keys",
        "description": "List all keys in the store.",
        "parameters": {"type": "object", "properties": {}}}},
]
