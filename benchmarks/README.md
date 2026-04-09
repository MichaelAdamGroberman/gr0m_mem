# Gr0m_Mem benchmarks

Every claim about Gr0m_Mem's behavior in the top-level README must cite a
file in [`results/`](results/). These runners produce those files.

## loop_prevention

**What it measures:** how often a two-session agent scenario hits a question
it could have answered from the wakeup store. This is the thing Gr0m_Mem
exists to solve, so it's the benchmark we run on every commit.

Needs no external datasets. Fully reproducible locally.

```bash
cd ~/Gr0m_Mem
.venv/bin/python -m benchmarks.loop_prevention.run \
    --out benchmarks/results/$(date -u +%Y-%m-%d)-loop-prevention.json
```

Reports: scenarios passed, scenarios failed, average tokens per wakeup
snapshot, per-scenario trace, and the aggregate "avoided-re-question
rate" (the fraction of prior decisions that were recalled rather than
re-asked). **Higher is better.** The goal is 100%.

## longmemeval

**What it measures:** R@5 on the LongMemEval benchmark (Hu et al., 2024).

Dataset is too large to commit. The runner expects the dataset at
`benchmarks/datasets/longmemeval-s.json` — download it with:

```bash
mkdir -p benchmarks/datasets
# TODO: published URL
```

Run:

```bash
.venv/bin/python -m benchmarks.longmemeval.run \
    --dataset benchmarks/datasets/longmemeval-s.json \
    --out benchmarks/results/$(date -u +%Y-%m-%d)-longmemeval.json
```

## locomo

**What it measures:** LoCoMo session-level R@10 (Maharana et al., 2024).

Same structure as longmemeval — dataset downloaded separately, runner
committed.

```bash
.venv/bin/python -m benchmarks.locomo.run \
    --dataset benchmarks/datasets/locomo.json \
    --out benchmarks/results/$(date -u +%Y-%m-%d)-locomo.json
```

## Honesty pledge

Every results file committed here is produced by a script in this
directory and is reproducible on the hardware listed in the
`metadata.environment` field of the JSON. If you find a results file
whose number cannot be reproduced by running the committed script,
please open an issue — that's a bug we will treat as high priority.
