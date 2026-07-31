# MultiBanFakeDetect

**Multimodal Ternary Bangla Fake News Detection with Integrated Gradients Explainability**

BanglaBERT-Large + ViT-B/16 + Cross-Modal Attention Fusion (CMAF)  
3-class: Real / Human-Fake / LLM-Fake  
Target: EMNLP 2026 Findings | Course: CSE-4877 | Institution: IIUC  
Team: Maruf Khan, Minhazul Alam, Saptarshi Barua | Supervisor: Nurul Absar

---

## Current Status

| Phase | Status | Result |
|-------|--------|--------|
| 1. Environment + dataset | ✅ Done | 9,600 samples verified |
| 2. LLM-fake generation | ⏳ Day 1-3 | Target: 3,000 samples |
| 3. QC + manifest | ⏳ Day 3 | After generation |
| 4. Train ternary model | ⏳ Day 4 | Target macro-F1 > 0.80 |
| 5. Ablations | ⏳ Day 4 | text-only, image-only |
| 6. Evaluate | ⏳ Day 5 | All 3 checkpoints |
| 7. IG explainability | ⏳ Day 5 | 10 examples per class |
| 8. QC annotation | ⏳ Day 5 | 200 samples, 2 annotators |
| 9. Paper writing | ⏳ Day 6 | Complete draft |
| 10. Review + polish | ⏳ Day 7 | Submission ready |

---

## ⚠️ SAVE VERSION RULE

**Click Save Version in Kaggle after EVERY step that produces data.**
- After every generation batch (150 samples)
- After QC + manifest build
- After training completes
- After evaluation
- After IG

Failing to Save Version loses all data when the session ends. This has happened twice.

---

## Confirmed Kaggle Setup

| Setting | Value |
|---------|-------|
| GPU | T4 × 2 (15 GB each) |
| Dataset | `/kaggle/input/datasets/mukaffimoin/multibanfakedetect-multimodal-bangla-fake-news` |
| Working dir | `/kaggle/working/ML-Research` |
| Batch size | 8 (with OOM fallback to 4) |
| Effective batch | 32 (batch 8 × accum 4) |
| freeze_text_layers | 6 (of 24) |
| freeze_image | False (ViT fine-tuned) |

---

## Generation Plan — LOCKED

| Model | Strategy | Target | Est. Cost |
|-------|----------|--------|-----------|
| GPT-4o Mini | rewrite | 640 | ~$0.19 |
| GPT-4o Mini | extend | 640 | ~$0.19 |
| GPT-4o Mini | summarize_extend | 640 | ~$0.19 |
| Claude Haiku | rewrite | 360 | ~$0.22 |
| Claude Haiku | extend | 360 | ~$0.22 |
| Claude Haiku | summarize_extend | 360 | ~$0.22 |
| **Total** | | **3,000** | **~$1.23** |

**Why 3,000 not 3,840:** class-weighted loss handles imbalance mathematically.
**Why no Gemini:** rewrite quality issues; fixed prompt untested.
**Why no Llama:** proven slow on Bangla prompts.
**Why all 3 strategies:** rewrite = most realistic misinformation pattern; needed for paper.

---

## Kaggle Notebook — Copy-Paste Cells In Order

### Setup (run every session)
```python
import os, sys

os.environ["MBFD_BASE_DIR"]    = "/kaggle/working/ML-Research"
os.environ["MBFD_DATASET_DIR"] = "/kaggle/input/datasets/mukaffimoin/multibanfakedetect-multimodal-bangla-fake-news"
sys.path.append("/kaggle/working/ML-Research")

os.system("git -C /kaggle/working/ML-Research fetch origin && "
          "git -C /kaggle/working/ML-Research reset --hard origin/master")

from kaggle_secrets import UserSecretsClient
os.environ["OPENROUTER_API_KEY"] = UserSecretsClient().get_secret("OPENROUTER_API_KEY")

for key in list(sys.modules.keys()):
    if any(x in key for x in ['generate','configs','dataset','model','build','train','evaluate','explain','qc']):
        del sys.modules[key]

from src.build_manifest import build_manifest
build_manifest()

from configs import config as cfg
print("Ready | Total LLM-fake so far:", 0)
```

### Day 1-3 — Generation (repeat per model/strategy)
```python
from src.generate_llm_fake import generate_batch, total_generated

# Run one at a time — Save Version after each
generate_batch("gpt-4o-mini",  "rewrite",          640)
# → SAVE VERSION
generate_batch("gpt-4o-mini",  "extend",            640)
# → SAVE VERSION
generate_batch("gpt-4o-mini",  "summarize_extend",  640)
# → SAVE VERSION
generate_batch("claude-haiku", "rewrite",           360)
# → SAVE VERSION
generate_batch("claude-haiku", "extend",            360)
# → SAVE VERSION
generate_batch("claude-haiku", "summarize_extend",  360)
# → SAVE VERSION

print("Total generated:", total_generated())
```

### Day 3 — QC + Manifest
```python
from src.qc_sample import draw_sample
from src.build_manifest import build_manifest

draw_sample()        # → give CSV to 2 annotators
build_manifest()     # → rebuilds 3-class manifest
# → SAVE VERSION
```

### Day 4 — Memory test + Train
```python
from src.train import memory_test, train

memory_test()                                    # verify batch=8 fits
train("cmaf_ternary", mode="full")               # → SAVE VERSION
train("text_only",    mode="text_only")          # → SAVE VERSION
train("image_only",   mode="image_only")         # → SAVE VERSION
```

### Day 5 — Evaluate + IG
```python
from src.evaluate import evaluate_all
from src.explain  import run_ig

evaluate_all()   # → SAVE VERSION
run_ig()         # → SAVE VERSION
```

### Day 5 (offline) — QC scoring
```python
from src.qc_sample import score_agreement
score_agreement()   # after annotators fill the CSV
```

---

## File Structure

```
configs/config.py           ← ALL settings — only file to edit per environment
src/generate_llm_fake.py    ← OpenRouter generation v5 (autosaves after every sample)
src/qc_sample.py            ← QC sampling + inter-annotator kappa
src/build_manifest.py       ← 3-class manifest builder
src/dataset.py              ← PyTorch Dataset (confirmed working)
src/model.py                ← BanglaBERT+ViT+CMAF v3 (all bugs fixed)
src/train.py                ← Training loop (OOM fallback, per-class F1)
src/evaluate.py             ← Test evaluation + per-generator analysis
src/explain.py              ← Integrated Gradients + plausibility template
requirements.txt            ← pip install -r requirements.txt
```

---

## Key Bugs Fixed (for paper methodology section)

| Bug | Impact | Fix |
|-----|--------|-----|
| Gate residual dropped image when gate→0 | Silent — wrong fusion | Average (text+image)/2 |
| _freeze_text_layers silent fail on Electra | Froze nothing | try/except with warning |
| ForwardWrapper arg order wrong | Silent — corrupted IG attributions | (embeds, pixels, mask) |
| Grad accum last batch never flushed | Weights not updated last N steps | Flush after epoch loop |
| Class weight explosion for missing class | Loss explosion | Only weight present classes |
| System prompt leaked project name | Model wrote project name in output | Removed all identifying info |
| Generation log not per-sample | Crash = lost all progress | Save after every accepted sample |

---

## Success Criteria

| Metric | Minimum | Good | Target |
|--------|---------|------|--------|
| Ternary macro-F1 | 0.75 | 0.82 | 0.87 |
| Real F1 | 0.85 | 0.90 | 0.93 |
| Human-Fake F1 | 0.70 | 0.78 | 0.83 |
| LLM-Fake F1 | 0.75 | 0.85 | 0.90 |
| LLM-Fake recall > Human-Fake recall | Required | Required | Required |

---

## Outstanding TODOs

- [ ] Identify second QC annotator (must read Bangla) — needed before Day 3
- [ ] Locate "Explainable FND in Bengali via LLM-Guided Hybrid Representations" paper — verify novelty
- [ ] Add credits to OpenRouter if balance drops below $0.30 during generation
- [ ] Run memory test (Day 4) before committing to full training
