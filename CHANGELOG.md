# Changelog

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
  systemd-managed serving on the Spark, make deploy/status/logs.
