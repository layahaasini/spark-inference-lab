# Operating the lab from the dev machine. `make help` lists targets.
SPARK_HOST ?= cute-spark
REMOTE_DIR ?= opensource/spark-inference-lab
# systemctl --user over ssh needs the user bus address, which non-login
# shells don't get for free
SYSCTL = XDG_RUNTIME_DIR=/run/user/$$(id -u) systemctl --user

RSYNC_EXCLUDES = --exclude .venv --exclude .pytest_cache --exclude __pycache__ \
                 --exclude "*.egg-info" --exclude runs --exclude .git

help: ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

test: ## Run the local test suite
	.venv/bin/python -m pytest -q

figs: ## Regenerate all figures from runs/*.json
	.venv/bin/python scripts/make_figs.py

deploy: ## Ship committed code to the Spark and restart both serving units
	@git diff --quiet && git diff --cached --quiet || { echo "uncommitted changes — commit first (deploys ship commits, not working trees)"; exit 1; }
	rsync -a $(RSYNC_EXCLUDES) ./ $(SPARK_HOST):$(REMOTE_DIR)/
	ssh $(SPARK_HOST) 'cd $(REMOTE_DIR) && .venv/bin/pip install -qe ".[dev,ui]" && $(SYSCTL) restart sparklab-server sparklab-ui'
	ssh $(SPARK_HOST) 'until curl -sf http://localhost:30000/health >/dev/null; do sleep 2; done; until curl -sf http://localhost:8080/ >/dev/null; do sleep 2; done; echo "deploy OK: server + ui healthy"'

status: ## Service + server health on the Spark
	ssh $(SPARK_HOST) '$(SYSCTL) status --no-pager sparklab-server sparklab-ui | grep -E "●|Active:"; curl -s -o /dev/null -w "server health: %{http_code}\n" --max-time 3 http://localhost:30000/health; curl -s -o /dev/null -w "ui health: %{http_code}\n" --max-time 3 http://localhost:8080/'

logs: ## Tail both serving units' journals on the Spark
	ssh $(SPARK_HOST) 'XDG_RUNTIME_DIR=/run/user/$$(id -u) journalctl --user -u sparklab-server -u sparklab-ui -n 50 --no-pager'

.PHONY: help test figs deploy status logs
