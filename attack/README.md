# MED-AI prompt injection benchmark

This directory contains the small attack-only benchmark described in
[`docs/implementation-detail.md`](docs/implementation-detail.md).

The initial version provides:

- all five paper attacks: Naive, Escape Characters, Context Ignoring, Fake
  Completion, and Combined;
- five remote/cached classification tasks: SST-2 sentiment, UCI SMS Spam, MRPC
  duplicate detection, HSOL hate detection, and RTE natural-language inference;
- deterministic pair sampling;
- PNA-T, PNA-I, Attack Success Value, and Matching Rate;
- selectable MED-AI YAML configuration with reasoning-only `external_source`.

Run the V0 smoke benchmark:

```bash
.venv/bin/python -m attack.cli \
  --target-config attack/configs/v0-attack.yaml \
  --attacks naive escape_characters context_ignoring fake_completion combined \
  --tasks sentiment spam duplicate hate nli \
  --target-limit 8 \
  --injected-limit 8 \
  --sample-size 8 \
  --seed 42 \
  --output-dir attack/output/smoke-v0
```

This makes live model calls. Hugging Face `datasets` downloads the datasets on
the first run and reuses its local cache on later runs. Sentiment uses
`stanfordnlp/sst2` validation data; spam uses `ucirvine/sms_spam` train data;
duplicate detection uses the MRPC validation split; hate detection uses the
HSOL train split; and NLI uses the RTE train split.

`--injected-limit` selects the PNA-I examples for each injected task.
`--sample-size` separately controls how many target/injected pairs are attacked.

The run writes `clean_targets.jsonl`, `clean_injected.jsonl`, and `cases.jsonl`
incrementally. It writes `metrics.json` after completion and records dataset and
run settings in `run_config.json`. Individual model-call failures are recorded
and counted instead of terminating the whole benchmark.

Run unit tests without an LLM endpoint:

```bash
.venv/bin/python -m unittest discover -s attack/tests -v
```
