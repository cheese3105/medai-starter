```bash
.venv/bin/python -m attack.cli \
  --target-config attack/configs/v0-attack.yaml \
  --attacks naive combined \
  --tasks sentiment \
  --target-limit 100 \
  --injected-limit 1 \
  --sample-size 1 \
  --seed 42 \
  --output-dir attack/output/pna-i-v0

.venv/bin/python -m attack.cli \
  --target-config attack/configs/v1-qr-attack.yaml \
  --attacks naive combined \
  --tasks sentiment \
  --target-limit 10 \
  --injected-limit 1 \
  --sample-size 1 \
  --seed 42 \
  --output-dir attack/output/pna-i-v1-qr

.venv/bin/python -m attack.cli \
  --target-config attack/configs/v0-attack.yaml \
  --attacks naive escape_characters context_ignoring fake_completion combined \
  --tasks sentiment spam duplicate hate nli \
  --target-limit 100 \
  --injected-limit 100 \
  --sample-size 100 \
  --seed 42 \
  --output-dir attack/output/v0
```
