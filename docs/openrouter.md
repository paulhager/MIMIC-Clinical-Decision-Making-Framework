# OpenRouter setup

This path is for hosted-model evaluation without downloading model weights.

## 1. Store the key locally

From the repository root:

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-your-key
```

The repository ignores `.env`. Do not put the key in a Hydra YAML file, shell
history, a result pickle, or a bug report.

## 2. Create the small API-only environment

```bash
uv venv --python 3.11 .venv-openrouter
uv pip install --python .venv-openrouter/bin/python \
  -r requirements-openrouter.txt
```

## 3. Run the synthetic smoke test

```bash
.venv-openrouter/bin/python scripts/smoke_openrouter.py
```

The test sends a fictional appendicitis vignette and passes only if Gemma 4
returns the known diagnosis. It does not load or send MIMIC data.

Gemma 4 is pinned to OpenRouter's `deepinfra` endpoint with provider fallback
disabled. This costs marginally more than the current cheapest endpoint but
prevents a leaderboard run from silently switching inference implementations.

## 4. Select Gemma 4 in the framework

The new Hydra model configuration is selected with:

```bash
python run.py model=Gemma4_26B_OpenRouter
```

The framework's full dependency environment and local dataset paths are still
required for a real benchmark run.

## Data governance

The model config requests an OpenRouter endpoint with `data_collection: deny`.
That routing preference is a technical safeguard, not a determination that
MIMIC-derived prompts may be sent to a third party. Confirm the applicable
PhysioNet credentialed-data agreement and your institution's approvals before
running real patient records through any hosted API.
