# Experiment image: NGC SGLang base + this repo, so a version tag names one
# immutable artifact containing both the serving environment and the
# instruments that measured it. Built arm64-only (the Spark).
FROM nvcr.io/nvidia/sglang:26.02-py3

WORKDIR /workspace/spark-inference-lab
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .
