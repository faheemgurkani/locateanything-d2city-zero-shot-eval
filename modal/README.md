# LocateAnything on Modal

Serverless inference for D2-City zero-shot eval. Based on [locateanything-modal](https://github.com/rohit4242/locateanything-modal).

## Prerequisites

- [Modal](https://modal.com/) account + CLI authenticated (`modal token set …`)
- Modal secret **`huggingface-secret`** with `HF_TOKEN` ([create secrets](https://modal.com/docs/guide/secrets))
- Accept the [NVIDIA LocateAnything-3B license](https://huggingface.co/nvidia/LocateAnything-3B) on Hugging Face

## Volume

The volume **`locateanything-weights`** is created automatically:

```python
volume = modal.Volume.from_name("locateanything-weights", create_if_missing=True)
```

Mounted at `/models/nvidia/LocateAnything-3B` inside containers. Populated once via `download.py`; persisted with `volume.commit()`.

## Commands

From project root (`locateanything-d2city-zero-shot-eval/`):

```bash
source .venv/bin/activate   # or parent repo .venv

# 1. One-time: download ~8 GB weights into Volume
python -m modal run modal/download.py::download_model

# 2. Dev server (ephemeral URL)
python -m modal serve modal/app.py

# 3. Production deploy (persistent URL)
python -m modal deploy modal/app.py
```

Copy the `*.modal.run` URL into `.env` as `MODAL_API_URL`.

## Smoke test

```bash
python scripts/test_modal_client.py \
  --url "$MODAL_API_URL" \
  --from-jsonl
```

Or with a specific image:

```bash
python scripts/test_modal_client.py \
  --url "$MODAL_API_URL" \
  --image path/to/frame.jpg \
  --categories car bus truck person bicycle motorcycle
```

## Batch D2-City eval

After `prepare_d2city_jsonl.py`:

```bash
python scripts/run_modal_eval.py --config config/d2city_eval.yaml

python eagle/Embodied/evaluation/metrics/other_metric.py \
  --data_path "$(python scripts/paths.py modal-jsonl)" \
  --output_path results/D2City_val/modal/eval_results.json
```

## Using another provider

Modal is one option. Any host that can run the FastAPI app in `modal/app.py` (or equivalent) and expose `POST /detect` works with `scripts/run_modal_eval.py` — point `--url` at your endpoint.

See also:

- [LocateAnything-3B on Hugging Face](https://huggingface.co/nvidia/LocateAnything-3B)
- [NVlabs/Eagle Embodied eval](https://github.com/NVlabs/Eagle/tree/main/Embodied/evaluation)
