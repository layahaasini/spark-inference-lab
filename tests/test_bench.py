from sparklab.bench import collect_manifest, percentiles


def test_percentiles_tail_vs_mean():
    # 99 fast runs and one planted 1000ms outlier: the mean barely notices, p99 must not
    xs = [10.0] * 99 + [1000.0]
    s = percentiles(xs)
    assert s["p50"] == 10.0
    assert s["p99"] == 1000.0
    assert s["mean"] < 25.0
    assert s["n"] == 100


def test_percentiles_single_value():
    s = percentiles([5.0])
    assert s["p50"] == s["p99"] == 5.0
    assert s["stdev"] == 0.0


def test_manifest_degrades_gracefully():
    # must work on a machine with no GPU and no sglang — fields go None, no crash
    m = collect_manifest(extra={"note": "test"})
    assert m["python"] and m["host"]
    assert "sglang" in m["versions"]
    assert m["note"] == "test"
