# MultiBanFakeDetect — Complete Project Log
# Extracted from full research journey: July 30 – Aug 6 2026
# Every event, result, failure, and fix documented in order.

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
- Repo: maruf99khan/ML-Research (renamed from MultiBanFakeDetect)
- Made public
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
- captum not installed by default — requires `pip install captum --break-system-packages`

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

### Initial plan (v1 — all 4 models)
- Gemini 2.5 Flash Lite, GPT-4o Mini, Llama 3.3 70B, Claude Haiku
- Total target: 3,800 samples

### Model ID problem
Config had `google/gemini-flash-1.5` — doesn't exist on OpenRouter (404 error).
Fixed by querying live OpenRouter API:
- `google/gemini-2.5-flash` ✅
- `openai/gpt-4o-mini` ✅
- `meta-llama/llama-3.3-70b-instruct` ✅
- `anthropic/claude-3-haiku` ✅

### Model dropped — Llama 3.3 70B
- Tested: unacceptable latency on Bangla prompts
- Infrastructure/routing issue on OpenRouter
- Permanently dropped

### Model dropped — Gemini 2.5 Flash (rewrite strategy)
**Batch 1 results (100 samples, Gemini 2.5 Flash, rewrite):**
- Accepted: 100
- Rejected: 94 (mostly too_similar_to_source)
- Attempts: 194
- Acceptance rate: 51.5%

**Problems found in batch 1:**
1. 2 outright refusals — model returned prompt instructions instead of fake news
2. "সিন্থেটিক/ভুয়া সংবাদ:" and similar prefixes not caught by strip list
3. 38/100 samples had overlap 0.70-0.79 — model barely changed source
4. Sample [30] contained "MultiBanFakeDetect" verbatim — system prompt leaked project name

**Root causes:**
- System prompt explicitly named "MultiBanFakeDetect" — fixed in v4 (removed all project-identifying language)
- Strip prefix list incomplete — expanded in v4
- Overlap threshold lowered 0.70→0.65
- Rewrite prompt strengthened: "change at least 7 specific facts, no sentence verbatim"
- Min length raised 200→250 chars

**Decision:** Drop Gemini rewrite (proven bad). Keep Gemini extend (quality was excellent — Pakistan-Afghanistan article was the best sample seen). Eventually dropped Gemini entirely for cleaner paper narrative (2 models sufficient).

### Final model selection
**GPT-4o Mini + Claude Haiku only**
- Both confirmed: ~95% acceptance rate, bangla=0.9996, overlap=0.42-0.48
- Different architectures (OpenAI vs Anthropic) — genuine generator diversity
- Cost: ~$0.0003 and ~$0.0006 per clean sample

---

## PHASE 5 — GENERATION EXECUTION (July 31 – Aug 1 2026)

### Generation strategy design
- **rewrite**: change ≥7 facts, no sentence verbatim, ≥150 words. Most realistic misinformation.
- **extend**: from headline, fabricate full article ≥200 words. Clearly LLM style.
- **summarize_extend**: summarize real article + add ≥3 fabricated claims. Middle ground.
- All 3 needed: without rewrite, LLM-fake would be trivially detectable (model learns "long new article = LLM")

### Quality filters v5 (final)
1. Length ≥ 200 chars
2. Bangla ratio ≥ 0.70
3. First-line Bangla ≥ 0.80 (catches English headline before Bangla body)
4. Sentence count ≥ 2 (।)
5. 3-gram source overlap ≤ 0.65 (tightened from 0.70 after batch 1)
6. Refusal pattern check
7. System prompt leak check (added after batch 1 "MultiBanFakeDetect" leak)
8. Meta-commentary stripping (strips prefixes like "বিভ্রান্তিকর সংস্করণ:")

### Storage problem — data lost twice
**Session wipe 1:** Lost ~400 samples. Forgot Save Version.
**Session wipe 2:** Lost 2,400 samples. Save Version (Quick Save) does NOT save /kaggle/working files — only saves notebook output cells. Critical discovery.

**Solution: GitHub auto-push after every batch**
- Round-trip test: generate 10 → push → wipe → git reset --hard origin/master → restore: 10 ✅
- Manifest excluded via .gitignore (combined_manifest.csv generated dynamically)
- git reset --hard origin/master is correct restore command (NOT git pull)

### Session 1 results (July 31 2026)
- 400 samples per combo × 6 combos = 2,400 total
- Time: 277 minutes
- Cost: ~$0.72
- All on GitHub

### Session 2 results (Aug 1 2026)
- 400 more per combo → 800 total per combo
- Time: 290 minutes
- Cost: ~$0.72
- **Total: 4,800 samples, perfectly balanced**

### Final generation stats (Aug 5 2026 verification)
| Combo | Samples | Bangla ratio | Mean overlap |
|-------|---------|-------------|-------------|
| gpt-4o-mini/rewrite | 800 | 0.9993 | 0.421 |
| gpt-4o-mini/extend | 800 | 0.9989 | 0.482 |
| gpt-4o-mini/summarize_extend | 800 | 0.9996 | ~0.42 |
| claude-haiku/rewrite | 800 | 0.9999 | ~0.40 |
| claude-haiku/extend | 800 | 0.9998 | ~0.43 |
| claude-haiku/summarize_extend | 800 | 0.9999 | 0.394 |

- Source diversity: 2,876 unique sources used out of 3,840 real train articles
- Max reuse of any source: 5 times
- All 12 categories covered (lifestyle=434, education=416, politics=414, ...)
- Zero meta leaks, zero English contamination

---

## PHASE 6 — MANIFEST BUILD (Aug 5 2026)

### Final manifest
- 14,400 rows total
- 4,800 per class (Real, Human-Fake, LLM-Fake)
- 3,840/480/480 train/val/test per class
- LLM-fake split stratified by generator (each generator appears in all splits)

---

## PHASE 7 — TERNARY MODEL TRAINING (Aug 5 2026)

### Session 1 — LOST
- Best: epoch 3, val macro-F1 = 0.9319
- Per-class: Real=0.896, Human-Fake=0.901, LLM-Fake=0.999
- Checkpoint LOST — session ended before saving
- Root cause: relied on session staying alive overnight

### Session 2 — LOST
- Best: epoch 12, val macro-F1 = 0.9290
- Backed up to /kaggle/working/checkpoints_backup/ before Quick Save
- LOST again — Quick Save does not save /kaggle/working files reliably
- Discovery: need Kaggle Dataset API for large file persistence

### Session 3 — SAVED (Aug 5 2026)
**Config:** freeze_text=6, freeze_image=False, batch=8, T4×2, fp32
**Trainable:** 320,704,515 / 429,578,243 params

| Epoch | Val Macro-F1 | Notes |
|-------|-------------|-------|
| 1 | 0.8394 | |
| 2 | 0.8990 | |
| 3 | 0.9076 | |
| 4 | 0.9173 | |
| 5 | 0.9146 | no improve |
| 6 | 0.9182 | new best |
| 7 | 0.9110 | no improve |
| 8 | 0.9110 | no improve |
| 9 | 0.9201 | new best |
| 10 | 0.9230 | new best |
| 12 | **0.9290** | **BEST — saved** |
| 13 | 0.9284 | no improve |
| 14 | early stop | patience=3 |

**Why different from Session 1 (0.9319 vs 0.9290):**
DataParallel non-deterministic CUDA ops across 2 GPUs. Even with set_seed(42), multi-GPU training has inherent stochasticity. Both within expected variance.

**Checkpoint saved to Kaggle Dataset:**
- Dataset: maruf99khan/multibanfakedetect-checkpoints
- Path: /kaggle/input/datasets/maruf99khan/multibanfakedetect-checkpoints/cmaf_ternary_best.pt
- Verified: epoch=12, val_macro_f1=0.9290, 563 param tensors ✅

---

## PHASE 8 — TEST EVALUATION (Aug 5 2026)

### Main model — cmaf_ternary
**Test macro-F1: 0.9207**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Real | 0.872 | 0.898 | 0.885 | 480 |
| Human-Fake | 0.896 | 0.865 | 0.880 | 480 |
| LLM-Fake | 0.994 | 1.000 | 0.997 | 480 |
| **Macro** | **0.921** | **0.921** | **0.921** | |

**Confusion matrix:**
```
[[431  48   1]
 [ 63 415   2]
 [  0   0 480]]
```

**Per-generator LLM-fake recall:**
| Generator | Recall |
|-----------|--------|
| Claude Haiku | 1.000 |
| GPT-4o Mini | 1.000 |

**KEY FINDING:**
LLM-Fake recall (1.000) >> Human-Fake recall (0.865)
Both generators equally detectable — no per-generator bias.
This detectability asymmetry was previously documented only in English literature.
We are first to confirm it in Bangla multimodal setting.

**Why val F1 (0.9290) > test F1 (0.9207):**
Normal generalization gap. Val used for checkpoint selection, test completely unseen.
Gap of 0.0083 is small — good generalization, no overfitting concern.

Results JSON pushed to GitHub: outputs/metrics/cmaf_ternary_test_results.json

---

## PHASE 9 — ABLATIONS (Aug 5-6 2026)

### Problem: OOM for text_only ablation
BanglaBERT-Large (~14GB) + ViT-B/16 (~2GB) = ~16GB > 15.53GB available.
Even with batch=2 and all layers frozen, OOM occurs.

### Failed attempt: fp16
Applied fp16 (torch.amp.autocast + GradScaler).
Result: NaN loss at epoch 3.
Root cause: BanglaBERT-Large has numerically sensitive layers — fp16 underflow in gradient computation.
Decision: revert to fp32, use different approach.

### Solution: skip_image=True for text_only
Remove ViT completely from text_only model — saves ~2GB VRAM, fits in fp32.
Added `skip_image=False` parameter to `MultiBanFakeDetectModel.__init__`.
When `skip_image=True`: ViT not loaded, `_image_tokens()` returns zeros.
Paper statement: "For text-only ablation, image encoder excluded to isolate text encoder performance."

### Text-only ablation status (Aug 6 2026 — IN PROGRESS)
Config: skip_image=True, freeze_text=6, batch=8, fp32
Trainable: 234,905,859 / 343,779,587 (ViT excluded)
Epoch 3: val macro-F1 = 0.8916 (Real=0.846, HFake=0.830, LLM=0.998)
Training running...

### Image-only ablation
Status: PENDING (next session)
Note: image_only doesn't need skip_image — just zeros text tokens in forward pass

---

## PHASE 10 — INTEGRATED GRADIENTS (pending)

### Problem: OOM
Model takes full 14+GB, leaves insufficient VRAM for gradient computation.
Must run in fresh session with no other models loaded.
Patch: run on CPU (slow but guaranteed memory).

---

## REMAINING TASKS

- [ ] Text-only ablation: finish training, evaluate, push results
- [ ] Image-only ablation: next session
- [ ] Integrated Gradients: fresh session, CPU mode
- [ ] QC 200 samples: need 2nd Bangla-reading annotator (BLOCKER)
- [ ] Paper writing: after all results
- [ ] Find "Explainable FND in Bengali via LLM-Guided Hybrid Representations" paper

---

## KEY NUMBERS FOR PAPER

### Results Table
| Model | Macro-F1 | Real F1 | HFake F1 | LLM F1 |
|-------|----------|---------|----------|--------|
| Binary baseline | 0.8792* | — | — | — |
| Text-only | TBD | TBD | TBD | TBD |
| Image-only | TBD | TBD | TBD | TBD |
| **CMAF (ours)** | **0.9207** | **0.885** | **0.880** | **0.997** |
| MBM-CTNet (reported) | 0.942** | — | — | — |

*val set, memory-constrained (freeze_text=12, freeze_image=True)
**binary task, reported in their paper

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
os.environ["OPENROUTER_API_KEY"] = UserSecretsClient().get_secret("OPENROUTER_API_KEY")
token = os.environ["GITHUB_TOKEN"]

os.system(f"git clone https://{token}@github.com/maruf99khan/ML-Research.git /kaggle/working/ML-Research")
os.chdir("/kaggle/working/ML-Research")
sys.path.insert(0, "/kaggle/working/ML-Research")
os.system(f"git remote set-url origin https://{token}@github.com/maruf99khan/ML-Research.git")
os.system("git config user.email 'kaggle@research.com'")
os.system("git config user.name 'Kaggle Runner'")
os.system("git fetch origin")
os.system("git reset --hard origin/master")  # NOT git pull

os.environ["MBFD_BASE_DIR"]    = "/kaggle/working/ML-Research"
os.environ["MBFD_DATASET_DIR"] = "/kaggle/input/datasets/mukaffimoin/multibanfakedetect-multimodal-bangla-fake-news"

from src.build_manifest import build_manifest
build_manifest()

import torch
print(f"Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
print("Ready")
```

## CHECKPOINT ACCESS (every new session)
Checkpoint is in Kaggle Dataset — add `maruf99khan/multibanfakedetect-checkpoints` as input.
Path: `/kaggle/input/datasets/maruf99khan/multibanfakedetect-checkpoints/cmaf_ternary_best.pt`
