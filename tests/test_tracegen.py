"""Pin the trace generator's promises: an experiment built on a generator that
silently breaks determinism or append-only growth produces wrong hit-rate
numbers with no error anywhere."""

import pytest

from sparklab.p1.tracegen import TraceConfig, gen_trace, load_trace, save_trace, shared_prefix

# both scheduling modes must uphold every invariant
MODES = [TraceConfig(seed=1), TraceConfig(seed=1, zipf_alpha=1.1)]


def test_determinism():
    for cfg in MODES:
        assert gen_trace(cfg) == gen_trace(cfg)


def test_different_seeds_differ():
    assert gen_trace(seed=1) != gen_trace(seed=2)


@pytest.mark.parametrize("cfg", MODES)
def test_monotone_history(cfg):
    # append-only rule: each session's next prompt must begin with its previous one
    last = {}
    for ev in gen_trace(cfg):
        s = ev["session"]
        if s in last:
            assert ev["prompt"].startswith(last[s])
        last[s] = ev["prompt"]


@pytest.mark.parametrize("cfg", MODES)
def test_shared_prefix(cfg):
    # every event starts with the one fixed system+tools head
    head = shared_prefix(cfg)
    events = gen_trace(cfg)
    assert all(ev["prompt"].startswith(head) for ev in events)
    # and the head is a real prefix, not the whole prompt
    assert all(len(ev["prompt"]) > len(head) for ev in events)


def test_group_shares_full_context():
    # group samples must share everything except the sample tag (GRPO rollout shape)
    by_key = {}
    for ev in gen_trace(seed=3, group_size=4):
        by_key.setdefault((ev["session"], ev["turn"]), []).append(ev["prompt"])
    for prompts in by_key.values():
        assert len(prompts) == 4
        base = prompts[0].rsplit("\nSAMPLE[", 1)[0]
        assert all(p.startswith(base) for p in prompts)


def test_zipf_skews_turn_counts():
    counts = {}
    for ev in gen_trace(seed=5, n_sessions=8, turns=20, zipf_alpha=1.5):
        counts[ev["session"]] = counts.get(ev["session"], 0) + 1
    # heavy-tail scheduling: the hottest session gets a large share, total is conserved
    assert sum(counts.values()) == 8 * 20
    assert max(counts.values()) > 2 * (8 * 20) / 8


def test_session_word_cap():
    cap = 1500
    for alpha in (0.0, 1.5):
        cfg = TraceConfig(seed=8, n_sessions=6, turns=40, sys_tokens=200,
                          zipf_alpha=alpha, max_session_words=cap)
        counts = {}
        for ev in gen_trace(cfg):
            counts[ev["session"]] = ev["prompt"]
        # cap + one turn's worst case (max tool words + header) is the hard bound,
        # because eligibility is checked before the turn's size is drawn
        bound = cap + 400 + 4
        for prompt in counts.values():
            assert len(prompt.split()) - 2 <= bound + cfg.sys_tokens  # session tag/head overhead
        # determinism must survive the capped scheduling path
        assert gen_trace(cfg) == gen_trace(cfg)


def test_event_count_even_mode():
    cfg = TraceConfig(seed=4, n_sessions=3, turns=5, group_size=2)
    assert len(gen_trace(cfg)) == 3 * 5 * 2


def test_save_load_roundtrip(tmp_path):
    cfg = TraceConfig(seed=6)
    events = gen_trace(cfg)
    save_trace(events, tmp_path / "t.json", cfg)
    assert load_trace(tmp_path / "t.json") == events
