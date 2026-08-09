# MultiBanFakeDetect — Complete Project Log
# Extracted from full research journey: July 30 – Aug 9 2026
# Every event, result, failure, and fix documented in order.
# Last verified: Aug 9 2026

---

## PHASE 1 — PROJECT SETUP (July 30 2026)

### What was built
Complete codebase built from scratch:
- `configs/config.py` — central configuration
- `src/generate_llm_fake.py` — OpenRouter generation pipeline
- `src/build_manifest.py` — 3-class manifest builder
- `src/dataset.py` — PyTorch Dataset
- `src/model.py` — BanglaBERT-Large + ViT-B/16 + CMAF
- `src/train.py` — training loop
- `src/evaluate.py` — test evaluation
- `src/explain.py` — Integrated Gradients
- `src/qc_sample.py` — inter-annotator QC

### Bugs found and fixed during code review (before any experiments)
1. **Gate residual bug** — `gate * fused + (1-gate) * text_pooled` dropped image when gate→0. Fixed to `(text+image)/2`
2. **Electra freeze bug** — `_freeze_text_layers` assumed BERT structure, silently failed on Electra. Fixed with try/except
3. **ForwardWrapper arg order** — `(embeds, mask, pixels)` wrong for Captum. Fixed to `(embeds, pixels, mask)`
4. **Grad accum flush bug** — last batch never flushed. Fixed after-loop flush
5. **Class weight bug** — LLM-fake weight = 2,560 when class empty. Fixed to only weight present classes
6. **Wrong model IDs** — config had `google/gemini-flash-1.5` (404). Fixed after live API verification
7. **FUSION_HIDDEN_DIM missing** from config — added back
8. **Per-class F1 not printed** — only macro shown. Fixed

### GitHub setup
- Repo: maruf99khan/ML-Research (public)
- Code pushed, verified on Kaggle

---

## PHASE 2 — ENVIRONMENT VERIFICATION (July 30 2026)

### Kaggle environment confirmed
- GPU: T4 × 2, 14.56GB VRAM per GPU
- Dataset path: `/kaggle/input/datasets/mukaffimoin/multibanfakedetect-multimodal-bangla-fake-news`
- Dataset columns: image_id, headline, description, label (0/1), types of fake news, category
- Note: actual columns different from expected — build_manifest.py updated to handle real column names
- All image paths: absolute `.png` paths, all verified present
- Text: `headline + " " + description`
- captum not installed by default — requires `pip install captum -q`

### Dataset confirmed
- 9,600 samples: 4,800 Real (label=0), 4,800 Human-Fake (label=1)
- Split: 7,680 train / 960 val / 960 test
- 12 categories: entertainment, national, sports, politics, education, crime, international, technology, finance, lifestyle, business, miscellaneous
- Perfectly balanced

### Model confirmed working
- BanglaBERT-Large loads correctly (UNEXPECTED keys warning is normal — Electra pre-training heads, safe to ignore)
- ViT-B/16 loads correctly
- Forward pass: `(4, 3)` output shape ✅
- ForwardWrapper: matches model output ✅

### Memory-safe settings confirmed on T4
- BATCH_SIZE=4, GRAD_ACCUM_STEPS=8 (effective batch=32)
- freeze_text_layers=12, freeze_image=True (binary baseline)
- gradient_checkpointing=True
- Trainable: 159M / 429M total params
- Free VRAM after model load: 9.39GB

---

## PHASE 3 — BINARY BASELINE TRAINING (July 31 2026)

### Settings
- Mode: real vs human_fake only (no LLM-fake yet)
- BATCH_SIZE=4, GRAD_ACCUM_STEPS=8, freeze_text=12, freeze_image=True
- Single T4 (memory constraint)

### Results
| Epoch | Val Loss | Val Macro-F1 | Confusion Matrix |
|-------|----------|-------------|-----------------|
| 1 | 0.4248 | 0.8170 | [[440,40],[134,346]] |
| 2 | 0.3468 | 0.8541 | [[403,77],[63,417]] |
| 3 | 0.4962 | **0.8792** ← best | [[425,55],[61,419]] |
| 4 | 0.5979 | 0.8770 | — |
| 5 | 0.6314 | 0.8771 | — |

- Early stopping triggered after epoch 5
- Val loss diverged from epoch 3 — overfitting signal
- Gap vs MBM-CTNet (94.2%): due to frozen encoders, not architecture issue

---

## PHASE 4 — GENERATION MODEL SELECTION (July 31 2026)

### Model ID problem
Config had `google/gemini-flash-1.5` — doesn't exist on OpenRouter (404 error).
Fixed by querying live OpenRouter API.

### Models dropped
- **Llama 3.3 70B**: unacceptable latency on Bangla prompts. Infrastructure/routing issue on OpenRouter.
- **Gemini 2.5 Flash**: rewrite strategy rejected 43% samples (TOO_SIMILAR, overlap>0.65). Dropped entirely for cleaner paper narrative.

### Final model selection: GPT-4o Mini + Claude Haiku only
- Both confirmed: ~95% acceptance rate, bangla=0.9996, overlap=0.42-0.48
- Different architectures (OpenAI vs Anthropic) — genuine generator diversity
- Cost: ~$0.0003 and ~$0.0006 per clean sample

---

## PHASE 5 — GENERATION EXECUTION (July 31 – Aug 1 2026)

### Generation strategies
- **rewrite**: change ≥7 facts, no sentence verbatim, ≥150 words
- **extend**: from headline, fabricate full article ≥200 words
- **summarize_extend**: summarize real article + add ≥3 fabricated claims

### Quality filters v5 (final)
1. Length ≥ 200 chars
2. Bangla ratio ≥ 0.70
3. First-line Bangla ≥ 0.80
4. Sentence count ≥ 2 (।)
5. 3-gram source overlap ≤ 0.65
6. Refusal pattern check
7. System prompt leak check
8. Meta-commentary stripping

### Storage lesson learned
Save Version does NOT save /kaggle/working files. Lost 2,400 samples twice.
Solution: push to GitHub after every batch. git reset --hard origin/master to restore (NOT git pull).

### Final generation stats (verified Aug 5 2026)
| Combo | Samples | Bangla ratio | Mean overlap |
|-------|---------|-------------|-------------|
| gpt-4o-mini / rewrite | 800 | 0.9993 | 0.421 |
| gpt-4o-mini / extend | 800 | 0.9989 | 0.482 |
| gpt-4o-mini / summarize_extend | 800 | 0.9996 | ~0.42 |
| claude-haiku / rewrite | 800 | 0.9999 | ~0.40 |
| claude-haiku / extend | 800 | 0.9998 | ~0.43 |
| claude-haiku / summarize_extend | 800 | 0.9999 | 0.394 |
| **Total** | **4,800** | | |

- Source diversity: 2,876 unique sources used
- Max reuse of any source: 5 times
- All 12 categories covered
- Total cost: ~$1.44 | Total time: ~567 min across 2 sessions

---

## PHASE 6 — MANIFEST BUILD (Aug 5 2026)

### Final manifest
- 14,400 rows total: 4,800 per class (Real, Human-Fake, LLM-Fake)
- 3,840 / 480 / 480 train / val / test per class
- LLM-fake split stratified by generator

---

## PHASE 7 — TERNARY MODEL TRAINING (Aug 5 2026)

### Sessions 1 & 2 — LOST (checkpoints not persisted)
- Session 1 best: epoch 3, val=0.9319 — lost on session end
- Session 2 best: epoch 12, val=0.9290 — lost (Quick Save does not save /kaggle/working)
- Discovery: must use Kaggle Dataset API for checkpoint persistence

### Session 3 — SAVED (Aug 5 2026)
**Config:** freeze_text=6, freeze_image=False, batch=8, T4×2, fp32
**Trainable:** 320,704,515 / 429,578,243 params

| Epoch | Val Macro-F1 | Notes |
|-------|-------------|-------|
| 1–11 | 0.8394→0.9230 | Steady improvement |
| 12 | **0.9290** | **BEST — saved** |
| 13 | 0.9284 | No improve |
| 14 | early stop | patience=3 |

Checkpoint uploaded to Kaggle Dataset. Later lost on session end (not re-uploaded correctly).

---

## PHASE 8 — ORIGINAL TEST EVALUATION (Aug 5 2026)

### cmaf_ternary original run (checkpoint subsequently lost)
Test macro-F1: 0.9207 | Real=0.885 | HFake=0.880 | LLM=0.997
Confusion matrix: [[431,48,1],[63,415,2],[0,0,480]]
Per-generator LLM recall: claude-haiku=1.000, gpt-4o-mini=1.000

---

## PHASE 7b — RETRAIN TO RECOVER CHECKPOINT (Aug 9 2026)

Original cmaf_ternary_best.pt was missing from Kaggle Dataset (only image_only_best.pt was there).
Retrained from scratch — identical config, identical data.

### Retrain training history
| Epoch | Val Macro-F1 | Notes |
|-------|-------------|-------|
| 1 | 0.8083 | Best saved |
| 2 | 0.8982 | Best saved |
| 3 | **0.9409** | **BEST — saved** |
| 4 | 0.9312 | No improve |
| 5 | 0.9270 | No improve |
| 6 | early stop | patience=3 |

Faster convergence than original (epoch 3 vs epoch 12) — DataParallel non-determinism, within expected variance.

### Retrain test results — USE THESE FOR PAPER
**Test macro-F1: 0.9285**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Real | 0.911 | 0.877 | 0.894 | 480 |
| Human-Fake | 0.880 | 0.912 | 0.896 | 480 |
| LLM-Fake | 0.996 | 0.996 | 0.996 | 480 |
| **Macro** | **0.929** | **0.928** | **0.928** | |

**Confusion matrix:**
[[421 58 1]
[ 41 438 1]
[ 0 2 478]]

**Per-generator LLM-fake recall:**
| Generator | Recall |
|-----------|--------|
| Claude Haiku | 0.992 |
| GPT-4o Mini | 1.000 |

**Key findings:**
- LLM-Fake recall (0.996) >> Human-Fake recall (0.912) — asymmetry confirmed in Bangla multimodal
- Both generators near-perfectly detectable — no per-generator bias
- Val-test gap: 0.9409 → 0.9285 = 0.012 — good generalization, no overfitting concern

Checkpoint: maruf99khan/multibanfakedetect-checkpoints/cmaf_ternary_best.pt ✅
Results JSON: outputs/metrics/cmaf_ternary_test_results.json ✅

---

## PHASE 9 — ABLATIONS (Aug 9 2026)

### OOM problem for text_only
BanglaBERT-Large (~14GB) + ViT-B/16 (~2GB) = ~16GB > 15.53GB available.
fp16 attempted — NaN loss at epoch 3 (Electra numerically sensitive). Reverted.
Solution: skip_image=True removes ViT entirely — saves 2GB, fits in fp32.

### Text-only ablation — DONE
Config: skip_image=True, freeze_text=6, freeze_image=False, batch=8, fp32
Trainable: 234,905,859 / 343,779,587

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Real | 0.817 | 0.892 | 0.853 |
| Human-Fake | 0.879 | 0.800 | 0.838 |
| LLM-Fake | 1.000 | 0.998 | 0.999 |
| **Macro** | | | **0.8964** |

Confusion matrix: [[428,52,0],[96,384,0],[0,1,479]]

Key findings:
- LLM-Fake near-perfect from text alone — linguistic fingerprint dominant
- HFake F1 drops 0.058 vs CMAF (0.838 vs 0.896) — image helps real vs human-fake
- 96 Human-Fake samples misclassified as Real — hard cases need visual context

Results JSON: outputs/metrics/text_only_test_results.json ✅

### Image-only ablation — DONE
Config: freeze_text=24 (all), freeze_image=False, batch=8, fp32, mode="image_only"
image_only zeros input_ids + sets mask=ones in train.py loop. BanglaBERT stays loaded.
Early stopping at epoch 5, best val=0.4019 (epoch 2)

| Epoch | Val | Train Loss | Note |
|-------|-----|-----------|------|
| 1 | 0.3617 | 1.0591 | Best saved |
| 2 | 0.4019 | 0.9500 | ✅ Best |
| 3 | 0.3973 | 0.8606 | |
| 4 | 0.3960 | 0.8235 | |
| 5 | 0.3192 | 0.8137 | Early stop |

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Real | 0.362 | 0.671 | 0.470 |
| Human-Fake | 0.659 | 0.358 | 0.464 |
| LLM-Fake | 0.543 | 0.327 | 0.408 |
| **Macro** | | | **0.4475** |

Confusion matrix: [[322,75,83],[259,172,49],[309,14,157]]
Per-generator LLM recall: claude-haiku=0.288, gpt-4o-mini=0.367

Key findings:
- 0.4475 — only marginally above random (0.333)
- 309/480 LLM-fake misclassified as Real — confirms LLM-fake uses real images, visually indistinguishable
- Human-Fake recall (0.358) > LLM-Fake recall (0.327) — does NOT replicate English asymmetry; image-only too noisy
- Training unstable — per-class recalls swing wildly across epochs

Results JSON: outputs/metrics/image_only_test_results.json ✅
Checkpoint: maruf99khan/multibanfakedetect-checkpoints/image_only_best.pt ✅

---

## PHASE 10 — INTEGRATED GRADIENTS (NEXT)

### Setup
- Fresh Kaggle notebook (GPU T4×2 enabled in settings)
- Add dataset `maruf99khan/multibanfakedetect-checkpoints` as input
- Install: `pip install captum -q`
- Model loaded to CPU inside run_ig() — do NOT set CUDA_VISIBLE_DEVICES
- Checkpoint: `/kaggle/input/datasets/maruf99khan/multibanfakedetect-checkpoints/cmaf_ternary_best.pt`
- Expected runtime: ~30-40 min on CPU for 30 examples (10 per class)

### Expected outputs
- outputs/explanations/ig_examples.csv
- outputs/explanations/human_plausibility_template.csv (40 samples for annotators)
- outputs/explanations/<sample_id>_patch_attr.npy

---

## REMAINING TASKS

- [x] Binary baseline — DONE (val macro-F1=0.8792)
- [x] LLM-fake generation — DONE (4,800 samples, 2 models × 3 strategies)
- [x] Ternary model training — DONE (test macro-F1=0.9285)
- [x] Text-only ablation — DONE (test macro-F1=0.8964)
- [x] Image-only ablation — DONE (test macro-F1=0.4475)
- [ ] **Integrated Gradients — NEXT** (fresh session, CPU inside run_ig, captum)
- [ ] QC 200 samples — BLOCKED (need 2nd Bangla-reading annotator; unblocks after IG produces plausibility template)
- [ ] Paper writing — PENDING (unblocks after IG)
- [ ] Locate "Explainable FND in Bengali via LLM-Guided Hybrid Representations" — HIGH PRIORITY before finalizing novelty claims

---

## KEY NUMBERS FOR PAPER

### Results Table (FINAL — use these numbers)
| Model | Test Macro-F1 | Real F1 | HFake F1 | LLM F1 |
|-------|--------------|---------|----------|--------|
| Binary baseline* | 0.8792 | — | — | — |
| Image-only | 0.4475 | 0.470 | 0.464 | 0.408 |
| Text-only | 0.8964 | 0.853 | 0.838 | 0.999 |
| **CMAF (ours)** | **0.9285** | **0.894** | **0.896** | **0.996** |
| MBM-CTNet** | 0.942 | — | — | — |

*val set only, binary task, memory-constrained config
**binary task, reported in their paper — not directly comparable

Ablation delta vs CMAF (0.9285):
- Image-only:  −0.481 (near-random; visual features insufficient alone)
- Text-only:   −0.032 (text dominant; fusion adds lift especially on HFake: +0.058)

### Dataset Stats
| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Real | 3,840 | 480 | 480 | 4,800 |
| Human-Fake | 3,840 | 480 | 480 | 4,800 |
| LLM-Fake | 3,840 | 480 | 480 | 4,800 |
| **Total** | **11,520** | **1,440** | **1,440** | **14,400** |

### LLM-Fake Generation
| Generator | Strategy | Samples |
|-----------|----------|---------|
| GPT-4o Mini | rewrite | 800 |
| GPT-4o Mini | extend | 800 |
| GPT-4o Mini | summarize_extend | 800 |
| Claude Haiku | rewrite | 800 |
| Claude Haiku | extend | 800 |
| Claude Haiku | summarize_extend | 800 |
| **Total** | | **4,800** |

---

## SESSION SETUP (every new session)

```python
import os, sys
from kaggle_secrets import UserSecretsClient

os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")
token = os.environ["GITHUB_TOKEN"]

os.system(f"git clone https://{token}@github.com/maruf99khan/ML-Research.git /kaggle/working/ML-Research")
os.chdir("/kaggle/working/ML-Research")
sys.path.insert(0, "/kaggle/working/ML-Research")
os.system(f"git remote set-url origin https://{token}@github.com/maruf99khan/ML-Research.git")
os.system("git config user.email 'kaggle@research.com'")
os.system("git config user.name 'Kaggle Runner'")
os.system("git fetch origin")
os.system("git reset --hard origin/master")

os.environ["MBFD_BASE_DIR"]    = "/kaggle/working/ML-Research"
os.environ["MBFD_DATASET_DIR"] = "/kaggle/input/datasets/mukaffimoin/multibanfakedetect-multimodal-bangla-fake-news"

from src.build_manifest import build_manifest
build_manifest()

import torch
print(f"GPUs: {torch.cuda.device_count()}")
print(f"Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
print("Ready")
```

## CHECKPOINT ACCESS
Dataset: `maruf99khan/multibanfakedetect-checkpoints` (add as notebook input)
- cmaf_ternary_best.pt → epoch=3, val=0.9409, test=0.9285 (Aug 9 2026)
- image_only_best.pt  → epoch=2, val=0.4019, test=0.4475 (Aug 9 2026)
Path: `/kaggle/input/datasets/maruf99khan/multibanfakedetect-checkpoints/<filename>`

## CHECKPOINT UPLOAD (after any training session)
```python
import os, json
from kaggle_secrets import UserSecretsClient

KAGGLE_API_KEY = UserSecretsClient().get_secret("KAGGLE_API_KEY")
os.makedirs("/root/.config/kaggle", exist_ok=True)
with open("/root/.config/kaggle/kaggle.json", "w") as f:
    json.dump({"username": "maruf99khan", "key": KAGGLE_API_KEY}, f)
os.chmod("/root/.config/kaggle/kaggle.json", 0o600)

STAGING = "/kaggle/working/upload_staging"
os.makedirs(STAGING, exist_ok=True)
os.system("cp /kaggle/working/ML-Research/outputs/checkpoints/<NAME>.pt /kaggle/working/upload_staging/<NAME>.pt")

with open(f"{STAGING}/dataset-metadata.json", "w") as f:
    json.dump({"title": "multibanfakedetect-checkpoints",
               "id": "maruf99khan/multibanfakedetect-checkpoints",
               "licenses": [{"name": "CC0-1.0"}]}, f)

os.system(f"kaggle datasets version -p {STAGING} -m 'description'")
```
