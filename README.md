# locateanything-d2city-zero-shot-eval

Zero-shot evaluation of **pretrained [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)** on **[D2-City](https://www.d2-city.org/)** dashcam validation data — no fine-tuning.

**Research question:** Does LocateAnything work for driver assistance out of the box?

This pipeline prepares D2-City frames, runs open-vocabulary detection via Modal (or local GPU), and computes box-level F1/recall metrics compatible with the official [NVlabs/Eagle](https://github.com/NVlabs/Eagle/tree/main/Embodied) evaluation scripts.

**License:** This repository is released under the [MIT License](LICENSE).

---

## References

| Resource | Link |
|----------|------|
| LocateAnything paper & project | [research.nvidia.com/labs/lpr/locate-anything/](https://research.nvidia.com/labs/lpr/locate-anything/) |
| Model weights (HF) | [huggingface.co/nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) |
| Official code (Embodied) | [github.com/NVlabs/Eagle/tree/main/Embodied](https://github.com/NVlabs/Eagle/tree/main/Embodied) |
| Hugging Face demo Space | [huggingface.co/spaces/nvidia/LocateAnything](https://huggingface.co/spaces/nvidia/LocateAnything) |
| Modal deployment pattern | [github.com/rohit4242/locateanything-modal](https://github.com/rohit4242/locateanything-modal) |
| Modal docs (Volumes, Secrets) | [modal.com/docs](https://modal.com/docs) |
| D2-City dataset (SciDB download) | [scidb.cn — D²-City](https://www.scidb.cn/en/detail?dataSetId=804399692560465920) |
| D2-City project page | [d2-city.org](https://www.d2-city.org/) |

> **Note:** NVIDIA does not host a public REST API for LocateAnything. Inference is self-hosted (Modal, local GPU, RunPod, etc.) using the HF weights and Eagle eval code.

---

## Directory layout

```
locateanything-d2city-zero-shot-eval/
├── README.md
├── LICENSE                    # MIT — see License section below
├── .gitignore
├── .env.example
├── config/
│   └── d2city_eval.yaml       # paths toggle, classes, sampling, Modal URL
├── data/                      # gitignored — see data/README.md
├── scripts/
│   ├── paths.py               # resolve data/output paths from config
│   ├── setup_env.sh           # local GPU env + Eagle install
│   ├── setup_modal.sh         # Modal client deps
│   ├── extract_d2city.sh      # unzip D2-City archives
│   ├── prepare_d2city_jsonl.py
│   ├── run_modal_eval.py      # batch inference via Modal API
│   ├── run_zero_shot_eval.sh  # local GPU inference + metrics
│   └── test_modal_client.py   # smoke test
├── modal/
│   ├── app.py                 # FastAPI /detect endpoint (L40S)
│   ├── download.py            # one-time HF weight download → Volume
│   └── README.md
├── eagle/                     # gitignored — clone NVlabs/Eagle here
├── models/                    # gitignored — hf download target
└── results/                   # gitignored — eval outputs
```

---

## Data path toggle

All scripts read `config/d2city_eval.yaml` → `paths` section.

| Mode | Config | Data location |
|------|--------|---------------|
| **Monorepo** (default) | `data_root_mode: monorepo` | `<parent-repo>/data/d2_city/` |
| **Standalone** | `data_root_mode: local` | `./data/d2_city/` at project root |
| **Custom** | `data_root: /path/to/data` | Explicit override |

```yaml
paths:
  data_root_mode: monorepo   # monorepo | local
  data_root: null            # optional override
  repo_root: "../../.."      # only for monorepo mode
```

Verify resolved paths:

```bash
python scripts/paths.py all
# or individually:
python scripts/paths.py data-root
python scripts/paths.py jsonl
```

See [data/README.md](data/README.md) for D2-City download and zip layout.

---

## Quick start

### 1. Clone dependencies

```bash
git clone https://github.com/YOUR_USER/locateanything-d2city-zero-shot-eval.git
cd locateanything-d2city-zero-shot-eval

# LocateAnything eval code (required for metrics + local GPU path)
git clone https://github.com/NVlabs/Eagle.git eagle
```

### 2. Python environment

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

For Modal-only workflow:

```bash
bash scripts/setup_modal.sh   # installs modal, requests, pycocotools, …
cp .env.example .env          # add MODAL_API_URL after deploy
```

### 3. D2-City data

**Standalone:** set `paths.data_root_mode: local`, download zips from [SciDB (D²-City)](https://www.scidb.cn/en/detail?dataSetId=804399692560465920) per [data/README.md](data/README.md), place under `data/d2_city/`.

**Monorepo:** keep default config; data lives in parent repo `data/d2_city/`.

```bash
bash scripts/extract_d2city.sh val
python scripts/prepare_d2city_jsonl.py
```

Default sampling (`frame_stride: 30`, `max_frames_per_video: 5`) yields **~500 eval frames** from 100 validation clips.

### 4a. Inference via Modal (no local GPU)

Prerequisites:

1. [Modal](https://modal.com/) account + CLI: `pip install modal && modal token set …`
2. Modal secret **`huggingface-secret`** with `HF_TOKEN` ([HF token](https://huggingface.co/settings/tokens); accept [NVIDIA model license](https://huggingface.co/nvidia/LocateAnything-3B) first)

```bash
# One-time: download ~8 GB weights into Modal Volume
python -m modal run modal/download.py::download_model

# Deploy persistent API → copy *.modal.run URL
python -m modal deploy modal/app.py
export MODAL_API_URL=https://YOUR-WORKSPACE--locateanything-3b-....modal.run

# Smoke test (uses first JSONL frame)
python scripts/test_modal_client.py --url "$MODAL_API_URL" --from-jsonl

# Batch eval (~33 min for 500 frames on L40S)
python scripts/run_modal_eval.py
```

Details: [modal/README.md](modal/README.md).

**Other cloud providers:** The client only needs an HTTP POST endpoint matching `/detect` (see `modal/app.py`). You can adapt the same FastAPI app for RunPod, AWS, GCP, etc.

### 4b. Inference via local GPU

```bash
# Accept NVIDIA license on Hugging Face, then:
hf download nvidia/LocateAnything-3B --local-dir models/LocateAnything-3B

bash scripts/run_zero_shot_eval.sh
```

Requires CUDA + Flash Attention 2 (recommended). See [Eagle/Embodied README](https://github.com/NVlabs/Eagle/tree/main/Embodied).

### 5. Metrics

```bash
python eagle/Embodied/evaluation/metrics/other_metric.py \
  --data_path "$(python scripts/paths.py modal-jsonl)" \
  --output_path results/D2City_val/modal/eval_results.json
```

Install deps if needed: `pip install pycocotools shapely`

---

## Eval configuration

Key settings in `config/d2city_eval.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `eval.query_classes` | 6 ADAS classes | Open-vocab queries sent to LocateAnything |
| `eval.frame_stride` | 30 | Sample every Nth frame (~1 fps) |
| `eval.max_frames_per_video` | 5 | Cap per clip (~500 total) |
| `model.generation_mode` | hybrid | hybrid \| fast \| slow |
| `modal.timeout_sec` | 600 | Per-frame HTTP timeout |

**Full val eval:** set `max_frames_per_video: null` (~2,473 frames).

---

## Pipeline overview

```mermaid
flowchart LR
  A[D2-City zips] --> B[extract_d2city.sh]
  B --> C[prepare_d2city_jsonl.py]
  C --> D{Inference}
  D -->|Modal| E[run_modal_eval.py]
  D -->|Local GPU| F[run_zero_shot_eval.sh]
  E --> G[other_metric.py]
  F --> G
  G --> H[results/*.json]
```

---

## Benchmark results (reproduced subset)

The following numbers were obtained on the **recommended 500-frame validation subset** defined in `config/d2city_eval.yaml` (`frame_stride: 30`, `max_frames_per_video: 5`). Inference used **Modal** (deployed LocateAnything-3B API); metrics used the official [`other_metric.py`](eagle/Embodied/evaluation/metrics/other_metric.py) script.

### Subset statistics

| Stat | Value |
|------|-------|
| D2-City split | Validation (`0008`) |
| Source clips (XML + MP4 pairs) | 100 |
| Eval frames (JPEG) | 500 (5 per clip) |
| Frame sampling | Every 30th frame (~1 fps from 30 fps source) |
| Query classes | `car`, `bus`, `truck`, `person`, `bicycle`, `motorcycle` |
| Ground-truth boxes (total) | **3,974** |
| GT boxes by class | car 2,980 · person 330 · truck 228 · bus 205 · bicycle 155 · motorcycle 76 |
| Extracted frames on disk | ~139 MB (`val/frames/0008/`) |
| Eval JSONL | `D2City_val.jsonl` (500 entries) |

Prepared with:

```bash
bash scripts/extract_d2city.sh val
python scripts/prepare_d2city_jsonl.py
# → 100 annotation files, 5 frames each, 500 total samples
```

### Inference run (Modal)

| Stat | Value |
|------|-------|
| Provider | [Modal](https://modal.com/) — L40S GPU endpoint |
| Model | [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) |
| Generation mode | `hybrid` |
| Frames submitted | 500 |
| Predictions written | **499** (1 HTTP 408 timeout) |
| Failed frame | `val/frames/0008/9475a1f61b820da2d2b03f3dc1da4ee6/000030.jpg` |
| Wall time | ~34 min (~4.1 s/frame avg) |
| Smoke-test latency (1 frame) | ~1.2 s |
| Predictions JSONL | `D2City_val_modal_answer.jsonl` |

Batch command:

```bash
python scripts/run_modal_eval.py
```

Smoke test (auto-picks first JSONL frame — avoid shell `<hash>` placeholders in zsh):

```bash
python scripts/test_modal_client.py --url "$MODAL_API_URL" --from-jsonl
```

### Detection metrics (`other_metric.py`, 499 scored samples)

| IoU threshold | Precision | Recall | F1 |
|---------------|-----------|--------|-----|
| **0.50** | **0.669** | **0.778** | **0.719** |
| 0.90 | 0.077 | 0.088 | 0.082 |
| **mIoU** (avg over 0.50–0.95) | **0.477** | **0.555** | **0.513** |

Additional reported metrics at IoU=0.5:

| Metric | Value |
|--------|-------|
| Instance follow rate | 0.9965 |
| Wrong rejection rate | 0.0000 |

Metrics command:

```bash
pip install pycocotools shapely   # required once

python eagle/Embodied/evaluation/metrics/other_metric.py \
  --data_path "$(python scripts/paths.py modal-jsonl)" \
  --output_path results/D2City_val/modal/eval_results.json
```

Full per-threshold breakdown is saved in [`results/D2City_val/modal/eval_results.json`](results/D2City_val/modal/eval_results.json).

> **Note:** One timed-out frame can be retried by re-running `run_modal_eval.py` with `--limit 1` on the missing entry, or by increasing `modal.timeout_sec` in the config.

---

## Comparison with RT-DETR

This repo produces a fixed JSONL (`D2City_val.jsonl`) with GT boxes. Fine-tuned RT-DETR from the parent [driver-assistance-system-using-RT-DETR](https://github.com/) project can be evaluated on the **same frames** for a fair zero-shot vs fine-tuned comparison.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Annotations not found` | Run `extract_d2city.sh val`; check `python scripts/paths.py data-root` |
| `eagle/ not found` | `git clone https://github.com/NVlabs/Eagle.git eagle` |
| Modal 408 timeout | Increase `modal.timeout_sec` or retry failed frames |
| `pycocotools` missing | `pip install pycocotools shapely` |
| HF model access denied | Accept license at [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) |
| Wrong data path | Toggle `paths.data_root_mode` or set `paths.data_root` |

---

## License

This repository (**locateanything-d2city-zero-shot-eval**) is licensed under the **MIT License**. See [LICENSE](LICENSE) for the full text.

Copyright (c) 2026 Muhammad Faheem

**Third-party components** (separate terms apply):

- LocateAnything model & Eagle code: [NVIDIA license](https://huggingface.co/nvidia/LocateAnything-3B)
- D2-City dataset: [SciDB download](https://www.scidb.cn/en/detail?dataSetId=804399692560465920) · [dataset terms](https://www.d2-city.org/)
