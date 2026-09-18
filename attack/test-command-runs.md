```bash
.venv/bin/python -m attack.cli \
  --target-config attack/configs/v0-attack.yaml \
  --attacks naive escape_characters context_ignoring fake_completion combined \
  --tasks sentiment spam duplicate hate nli \
  --target-limit 100 \
  --injected-limit 100 \
  --sample-size 100 \
  --seed 42 \
  --output-dir attack/output/v0

.venv/bin/python -m attack.cli \
  --target-config attack/configs/v1-qr-attack.yaml \
  --attacks naive escape_characters context_ignoring fake_completion combined \
  --tasks sentiment spam duplicate hate nli \
  --target-limit 100 \
  --injected-limit 100 \
  --sample-size 100 \
  --seed 42 \
  --output-dir attack/output/v1

.venv/bin/python -m attack.cli \
  --target-config attack/configs/v2-qr-attack.yaml \
  --attacks naive escape_characters context_ignoring fake_completion combined \
  --tasks sentiment spam duplicate hate nli \
  --target-limit 100 \
  --injected-limit 100 \
  --sample-size 100 \
  --seed 42 \
  --output-dir attack/output/v2

.venv/bin/python -m attack.cli \
  --target-config attack/configs/v3-qr-attack.yaml \
  --attacks naive escape_characters context_ignoring fake_completion combined \
  --tasks sentiment spam duplicate hate nli \
  --target-limit 100 \
  --injected-limit 100 \
  --sample-size 100 \
  --seed 42 \
  --output-dir attack/output/v3
```
