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

# from the dev machine
ssh -L 3000:localhost:3000 cute-spark
# open http://localhost:3000 — anonymous viewer access; admin/sparklab to edit
```

Nothing binds off-box: all three services use host networking on localhost
only, reached through the ssh tunnel. Per-request experiment records remain
the source of truth for published numbers; this catches what they can't —
thermal throttling, memory creep, and restarts *between* measurements.
