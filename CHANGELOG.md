# Changelog

## v0.2.0 — 2026-08-16

Operations hardening + interactive surface.

- Alerting: five Prometheus rules (server down/restarted, low hit rate,
  UI down, UI error ratio) via Alertmanager; persistent silences;
  Slack/email delivery slots.
- Inference web UI (FastAPI, systemd unit `sparklab-ui`, localhost:8080):
  prompt box with per-request TTFT + cache stats, own Prometheus metrics,
  three new dashboard panels.
- Review pass (10 verified findings fixed): monitoring stack pinned to
  127.0.0.1 (was LAN-exposed), broken cache-share panel PromQL, alert
  rules that could never fire at real traffic, latency-histogram buckets
  and failure-mixing, CI/deploy/image gaps for the ui extra, 45s
  interactive timeout.
- Failure drill verified: kill → auto-restart in 58s, restart alert
  fired, journald record.
- make deploy/status/logs cover both units; release builds cache layers.

## v0.1.0 — 2026-08-14

Project 1 measurement campaign complete.

- Trace generator: shared prefix, append-only growth, session interleaving,
  Zipf scheduling, GRPO group mode, per-session context budget.
- Measured client (server-reported cache ground truth), warmup discipline
  (past chunked_prefill_size), cold/warm sanity check, pressure sweep.
- Shared bench module: percentile timing, run manifests.
- Results: 84% prefill refund, knee predicted at 73 sessions / measured 72–80,
  tail-first degradation, Zipf holds 86.5% vs uniform 64.8% under identical
  pressure, group mode ~96%. Write-up draft + figures.
- Ops: CI (tests on 3.11/3.12), arm64 release image to GHCR on tags,
  systemd-managed serving on the Spark, make deploy/status/logs,
  Prometheus/Grafana monitoring with provisioned dashboard.
