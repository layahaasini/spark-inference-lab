# Monitoring

Prometheus (records the server's `/metrics` every 5s) + Grafana (draws the
history), provisioned with one dashboard: hit rate, live prefill
computed-vs-cached refund, TTFT p50/p99, KV pool occupancy, running/queued
requests, decode throughput, and a restart detector that goes red if the
server restarted within the hour — a nonzero value during a measurement
window invalidates the window.

```bash
# on the Spark, from this directory
docker compose up -d
docker compose --profile gpu up -d    # optional: GPU temp/clocks via DCGM (best-effort on GB10)

# from the dev machine (any free local port works on the left side;
# 3000 is often taken locally by dev servers/port-forwards)
ssh -L 3001:localhost:3000 cute-spark
# dashboard: http://localhost:3001/d/sparklab
# anonymous viewer access; admin/sparklab to edit
```

Nothing binds off-box: all four services (Prometheus :9090, Alertmanager
:9093, Grafana :3000, optional DCGM :9400) run host-networked but are pinned
to 127.0.0.1 — reached only through ssh tunnels. Alertmanager's UI (tunnel
9093) is where silences go during planned maintenance; its silences and
notification log persist in the `am-data` volume. Per-request experiment
records remain the source of truth for published numbers; this stack catches
what they can't — thermal throttling, memory creep, and restarts *between*
measurements.
