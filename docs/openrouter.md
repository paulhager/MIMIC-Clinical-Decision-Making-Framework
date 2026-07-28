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

That tests Gemma 4 only. Test every configured OpenRouter model with:

```bash
.venv-openrouter/bin/python scripts/smoke_openrouter.py --all
```

The tests send a fictional appendicitis vignette and pass only if each model
returns the known diagnosis. They do not load or send MIMIC data.

The currently configured hosted models are:

| Hydra config | OpenRouter model | Pinned provider |
| --- | --- | --- |
| `Gemma4_26B_OpenRouter` | `google/gemma-4-26b-a4b-it` | DeepInfra |
| `GPTOSS120B_OpenRouter` | `openai/gpt-oss-120b` | DeepInfra |
| `Qwen3.5_397B_OpenRouter` | `qwen/qwen3.5-397b-a17b` | Alibaba |
| `DeepSeekV4Flash_OpenRouter` | `deepseek/deepseek-v4-flash` | DeepInfra |
| `Llama4Maverick_OpenRouter` | `meta-llama/llama-4-maverick` | DeepInfra |
| `MistralSmall4_OpenRouter` | `mistralai/mistral-small-2603` | Mistral |

Provider fallback is disabled so a leaderboard run cannot silently switch
inference implementations. Every config also requires supported request
parameters and denies endpoints marked as collecting data.

## 4. Select a hosted model in the framework

Select a configuration by its filename stem:

```bash
python run.py model=Gemma4_26B_OpenRouter
python run.py model=GPTOSS120B_OpenRouter
python run.py model=Qwen3.5_397B_OpenRouter
```

The framework's full dependency environment and local dataset paths are still
required for a real benchmark run.

## 5. Prepare the released CSV dataset

Hosted inference has a CPU-only environment that excludes all CUDA and
local-weight packages:

```bash
uv venv --python 3.11 .venv-hosted
uv pip install --python .venv-hosted/bin/python \
  -r requirements-hosted.txt
```

Convert the credentialed PhysioNet release into the framework's runtime format:

```bash
.venv-hosted/bin/python scripts/prepare_physionet_data.py \
  /path/to/MIMIC-CDM-PhysionetDataset
```

This creates ignored files under `runtime_data/` and writes the ignored local
Hydra paths config at `configs/paths/local.yaml`. It also creates one-patient
smoke lists and deterministic five-patient pilot lists for inexpensive
end-to-end tests.

## 6. Keep the campaign under budget

Add these optional settings to `.env`:

```dotenv
OPENROUTER_USAGE_LOG=runtime_data/openrouter_usage.jsonl
OPENROUTER_BUDGET_USD=80
OPENROUTER_BUDGET_RESERVE_USD=1
```

Every hosted response then appends model, provider, token, and cost metadata to
the ignored JSONL ledger. Prompts and responses are never written to that file.
Before every request, the client totals the ledger and stops at $79, leaving the
configured $1 reserve. An $80 campaign limit intentionally leaves headroom below
a $100 account or card ceiling.

Inspect cumulative spend at any time with:

```bash
.venv-hosted/bin/python scripts/openrouter_usage.py
```

Use a five-case CDM pilot for each pathology before a full run:

```bash
.venv-hosted/bin/python run.py \
  paths=local \
  model=Gemma4_26B_OpenRouter \
  pathology=appendicitis \
  patient_list_path=runtime_data/appendicitis_pilot_patient.pkl \
  run_descr=_PILOT
```

After confirming that sending credentialed MIMIC-derived text to the configured
provider is permitted, a one-case Full Information test can be run with:

```bash
.venv-hosted/bin/python run_full_info.py \
  paths=local \
  model=Gemma4_26B_OpenRouter \
  pathology=appendicitis \
  patient_list_path=runtime_data/appendicitis_smoke_patient.pkl \
  run_descr=_SMOKE
```

## Data governance

The model configs request OpenRouter endpoints with `data_collection: deny`.
That routing preference is a technical safeguard, not authorization to send
credentialed health data to a third party. Confirm the applicable PhysioNet
credentialed-data agreement, provider terms, and institutional approvals before
running real patient records through any hosted API.
