# MultiBanFakeDetect

Multimodal, ternary (Real / Human-Fake / LLM-Fake), explainable Bangla fake news detection.
**BanglaBERT-Large + ViT-B/16 + Cross-Modal Attention Fusion (CMAF) + Integrated Gradients.**

Target venue: EMNLP 2026 Findings | Course: CSE-4877 | Institution: IIUC
Team: Maruf Khan, Minhazul Alam, Saptarshi Barua | Supervisor: Nurul Absar

---

## Current Status

| Phase | Status | Result |
|-------|--------|--------|
| 1. Environment + dataset setup | ✅ Done | 9,600 samples, all paths verified |
| 2. LLM-fake generation | 🔄 In progress | Gemini 2.5 Flash batch 1 running |
| 3. Train binary baseline | ✅ Done | macro-F1 = 0.8792 (epoch 3, frozen ViT) |
| 3. Train ternary model | ⏳ After Phase 2 | target >0.90 |
| 4. Explainability (IG) | ⏳ After Phase 3 | — |
| 5. Evaluation + ablations | ⏳ After Phase 4 | — |
| 6. Paper writing | ⏳ Last | EMNLP 2026 Findings |

---

## Confirmed Kaggle Environment

| Setting | Value |
|---------|-------|
| GPU | T4 (14.56 GB VRAM) |
| Dataset path | `/kaggle/input/datasets/mukaffimoin/multibanfakedetect-multimodal-bangla-fake-news` |
| Working dir | `/kaggle/working/ML-Research` |
| BATCH_SIZE | 4 (OOM at 16) |
| GRAD_ACCUM_STEPS | 8 (effective batch = 32) |
| FREEZE_TEXT_LAYERS | 12 (of 24) |
| FREEZE_IMAGE | True |
| Trainable params | 159M / 429M total |
| Free VRAM after load | 9.39 GB |

---

## Generator Models (verified July 30, 2026)

| Key | Model ID | Target | Est. cost |
|-----|----------|--------|-----------|
| gemini-2.5-flash | `google/gemini-2.5-flash` | 1,200 | ~$1.38 |
| gpt-4o-mini | `openai/gpt-4o-mini` | 1,000 | ~$0.32 |
| llama-3.3-70b | `meta-llama/llama-3.3-70b-instruct` | 800 | ~$0.21 |
| claude-haiku | `anthropic/claude-3-haiku` | 800 | ~$0.60 |
| **Total** | | **3,800** | **~$2.51** |

---

## Bug Fixes (v3, July 30 2026)

| File | Bug | Fix |
|------|-----|-----|
| `model.py` | Gate residual only used text_pooled, dropped image when gate=0 | Residual now averages (text_pooled + image_pooled) / 2 |
| `model.py` | `_freeze_text_layers` assumed BERT structure, silent fail on Electra | Wrapped in try/except, prints clear warning |
| `model.py` | `FUSION_HIDDEN_DIM` missing from config, crash at import | Added back to config.py |
| `train.py` | Last partial grad-accum batch never flushed | Added flush after epoch loop |
| `train.py` | Only macro-F1 printed, can't monitor per-class during training | Per-class P/R/F1 now printed every epoch |
| `config.py` | Old broken OpenRouter model IDs | Correct IDs verified live July 30, 2026 |
| `config.py` | `USE_FALLBACK`, `FALLBACK_TEXT_MODEL_NAME` removed but still referenced | Added back for backward compatibility |
| `dataset.py` | Class weight for missing class exploded to 2,560 | Only weights classes present in split |

---

## Kaggle Notebook — Copy-Paste Cells In Order

### Cell 1 — Pull latest code
```python
import os
if os.path.exists("/kaggle/working/ML-Research"):
    os.chdir("/kaggle/working/ML-Research")
    !git pull
else:
    !git clone https://github.com/maruf99khan/ML-Research.git
    os.chdir("/kaggle/working/ML-Research")
```

### Cell 2 — Install dependencies
```python
!pip install -r requirements.txt -q
```

### Cell 3 — Setup environment
```python
import os, sys
os.environ["MBFD_BASE_DIR"]    = "/kaggle/working/ML-Research"
os.environ["MBFD_DATASET_DIR"] = "/kaggle/input/datasets/mukaffimoin/multibanfakedetect-multimodal-bangla-fake-news"
sys.path.append("/kaggle/working/ML-Research")

from kaggle_secrets import UserSecretsClient
os.environ["OPENROUTER_API_KEY"] = UserSecretsClient().get_secret("OPENROUTER_API_KEY")

from configs import config as cfg
print("Ready. Base dir:", cfg.BASE_DIR)
print("Generator models:", list(cfg.GENERATOR_MODELS.keys()))
```

### Cell 4 — Build manifest (run once, re-run after adding LLM-fake data)
```python
from src.build_manifest import build_manifest
build_manifest()
```

### Cell 5 — Generate LLM-fake samples (Phase 2)
```python
# Run 100 samples at a time. After ALL models done, re-run Cell 4.
from src.generate_llm_fake import generate_batch_from_manifest

generate_batch_from_manifest(
    strategy="rewrite",           # rewrite / extend / summarize_extend
    generator_key="gemini-2.5-flash",  # see GENERATOR_MODELS in config
    n=100,
    manifest_path=cfg.COMBINED_MANIFEST,
    output_dir=cfg.LLM_FAKE_DIR,
)
```

### Cell 6 — QC sample (Phase 2, after generation)
```python
from src.qc_sample import draw_sample
draw_sample(n=200)
# Have 2 annotators fill the CSV, then:
# from src.qc_sample import score_agreement
# score_agreement(path=f"{cfg.QC_DIR}/qc_sample_annotated.csv")
```

### Cell 7 — Train ternary model (Phase 3)
```python
from src.train import train
train(run_name="cmaf_ternary", mode="full", num_epochs=10)
```

### Cell 8 — Ablations (Phase 3, run after main model)
```python
train(run_name="text_only",  mode="text_only",  num_epochs=10)
train(run_name="image_only", mode="image_only", num_epochs=10)
```

### Cell 9 — Evaluate (Phase 5)
```python
from src.evaluate import evaluate_full
evaluate_full(
    checkpoint_path=f"{cfg.CHECKPOINT_DIR}/cmaf_ternary_best.pt",
    output_dir=cfg.METRICS_DIR,
)
```

### Cell 10 — Explainability (Phase 4)
```python
from src.explain import run_explanations
run_explanations(
    checkpoint_path=f"{cfg.CHECKPOINT_DIR}/cmaf_ternary_best.pt",
    split="test",
    output_dir=cfg.EXPLAIN_DIR,
)
```

---

## File Structure

```
configs/config.py           ← ALL settings. Only file to edit per environment.
src/build_manifest.py       ← builds combined 3-class manifest CSV
src/generate_llm_fake.py    ← OpenRouter generation (v2, quality-filtered)
src/qc_sample.py            ← manual QC + inter-annotator kappa
src/dataset.py              ← PyTorch Dataset (confirmed working)
src/model.py                ← BanglaBERT+ViT+CMAF (v3, gate bug fixed)
src/train.py                ← training loop (v2, grad-accum flush fixed)
src/evaluate.py             ← test metrics, confusion matrix, per-generator
src/explain.py              ← Integrated Gradients + plausibility check
requirements.txt            ← pip install -r requirements.txt
```

---

## If Something Breaks

Paste:
1. Full error traceback
2. Which cell you ran
3. What you expected vs what happened

---

## Outstanding TODOs Before Submission

- [ ] Locate "Explainable Fake News Detection in Bengali via LLM-Guided Hybrid Representations" — verify it doesn't conflict with novelty claim 3
- [ ] Complete Phase 2 generation (~3,800 samples across 4 models × 3 strategies)
- [ ] Manual QC (200 samples, 2 annotators, report Cohen's kappa)
- [ ] Rebuild manifest with 3 classes after generation
- [ ] Train ternary model (target macro-F1 > 0.90)
- [ ] Run text-only and image-only ablations
- [ ] Run IG explainability + human plausibility check (40 samples, 2 annotators)
- [ ] Write paper
- [ ] Re-run literature search 1-2 weeks before submission (fast-moving area)
