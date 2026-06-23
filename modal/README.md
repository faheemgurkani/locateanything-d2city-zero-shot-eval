# LocateAnything on Modal

Serverless inference for D²-City zero-shot eval. Based on [locateanything-modal](https://github.com/rohit4242/locateanything-modal).

## One-time setup checklist

- [ ] [Modal](https://modal.com/) account created
- [ ] CLI installed: `pip install modal` (included in `requirements.txt`)
- [ ] Authenticated: `modal token set …`
- [ ] [Hugging Face token](https://huggingface.co/settings/tokens) created
- [ ] [LocateAnything-3B license](https://huggingface.co/nvidia/LocateAnything-3B) accepted on HF
- [ ] Modal secret **`huggingface-secret`** created with `HF_TOKEN` ([secrets guide](https://modal.com/docs/guide/secrets))
- [ ] Weights downloaded to Volume (step 1 below)
- [ ] API deployed (step 3 below)
- [ ] `MODAL_API_URL` saved in `.env`

## Volume

```python
volume = modal.Volume.from_name("locateanything-weights", create_if_missing=True)
```

Mounted at `/models/nvidia/LocateAnything-3B` inside containers. Populated once via `download.py`.

## Commands

From project root:

```bash
source .venv/bin/activate

# 1. One-time: download ~8 GB weights into Volume
python -m modal run modal/download.py::download_model

# 2. Dev server (ephemeral URL, for debugging)
python -m modal serve modal/app.py

# 3. Production deploy (persistent URL)
python -m modal deploy modal/app.py
```

Copy the `*.modal.run` URL:

```bash
echo 'MODAL_API_URL=https://YOUR-WORKSPACE--locateanything-3b-....modal.run' >> .env
export MODAL_API_URL=...
```

## Smoke test

```bash
python scripts/test_modal_client.py --url "$MODAL_API_URL" --from-jsonl
```

## Batch eval + metrics

```bash
python scripts/run_modal_eval.py --url "$MODAL_API_URL"

python eagle/Embodied/evaluation/metrics/other_metric.py \
  --data_path "$(python scripts/paths.py modal-jsonl)" \
  --output_path results/D2City_val/modal/eval_results.json

bash scripts/render_all_figures.sh
```

Or run the full pipeline:

```bash
bash scripts/reproduce_results.sh
```

## Expected performance (L40S, hybrid)

| Metric | Value |
|--------|-------|
| Frames | 500 submitted, 499 scored |
| Wall time | ~34 min |
| Mean latency | ~668 ms/frame (~1.5 FPS) |
| F1 @ IoU 0.5 | 0.719 |

See [README.md § Expected results](../README.md#expected-results-500-frame-subset).

## Using another provider

Any host exposing `POST /detect` compatible with `modal/app.py` works with `scripts/run_modal_eval.py` — pass `--url` to your endpoint. Modal, RunPod, AWS, GCP, etc.

## References

- [LocateAnything-3B on Hugging Face](https://huggingface.co/nvidia/LocateAnything-3B)
- [NVlabs/Eagle Embodied eval](https://github.com/NVlabs/Eagle/tree/main/Embodied/evaluation)
- [Modal docs](https://modal.com/docs)
