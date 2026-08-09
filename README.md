# MultiBanFakeDetect

**Multimodal Ternary Bangla Fake News Detection with Integrated Gradients Explainability**

BanglaBERT-Large + ViT-B/16 + Cross-Modal Attention Fusion (CMAF)  
3-class: Real / Human-Fake / LLM-Fake  
Target: EMNLP 2026 Findings | Course: CSE-4877 | Institution: IIUC  
Team: Maruf Khan, Minhazul Alam, Saptarshi Barua | Supervisor: Nurul Absar

---

## Results

| Model | Test Macro-F1 | Real F1 | Human-Fake F1 | LLM-Fake F1 |
|-------|--------------|---------|---------------|-------------|
| Image-only ablation | 0.4475 | 0.470 | 0.464 | 0.408 |
| Text-only ablation | 0.8964 | 0.853 | 0.838 | 0.999 |
| **CMAF (ours)** | **0.9285** | **0.894** | **0.896** | **0.996** |
| MBM-CTNet* | 0.942 | — | — | — |

*binary task, reported in their paper — not directly comparable

**Key finding:** LLM-Fake recall (0.996) >> Human-Fake recall (0.912) — detectability asymmetry confirmed in Bangla multimodal setting, consistent with English literature.

**Per-generator LLM-fake recall:** Claude Haiku = 0.992, GPT-4o Mini = 1.000

Val macro-F1 (checkpoint selection): 0.9409 (epoch 3, early stop at epoch 6)

---

## Dataset

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Real | 3,840 | 480 | 480 | 4,800 |
| Human-Fake | 3,840 | 480 | 480 | 4,800 |
| LLM-Fake | 3,840 | 480 | 480 | 4,800 |
| **Total** | **11,520** | **1,440** | **1,440** | **14,400** |

Source dataset: `mukaffimoin/multibanfakedetect-multimodal-bangla-fake-news` (Kaggle)  
Checkpoints: `maruf99khan/multibanfakedetect-checkpoints` (Kaggle)

---

## LLM-Fake Generation Methodology

Generation is complete (4,800 samples). Documented here for paper reproducibility.

### Models
- **GPT-4o Mini** (`openai/gpt-4o-mini` via OpenRouter) — 2,400 samples
- **Claude Haiku** (`anthropic/claude-3-haiku` via OpenRouter) — 2,400 samples
- Dropped: Gemini 2.5 Flash (43% rejection rate on rewrite), Llama 3.3 70B (unacceptable latency on Bangla)

### Strategies (800 samples each per model)
- **rewrite** — change ≥7 specific facts, no sentence verbatim, ≥150 words
- **extend** — from headline only, fabricate full article ≥200 words
- **summarize_extend** — keep main topic, add ≥3 fabricated claims, ≥200 words

### Quality Filters (v5 final)
1. Length ≥ 200 chars
2. Bangla character ratio ≥ 0.70
3. First-line Bangla ratio ≥ 0.80 (catches English headline before Bangla body)
4. Sentence count ≥ 2 (Bangla terminator: ।)
5. 3-gram source overlap ≤ 0.65
6. Refusal pattern check
7. System prompt leak check
8. Meta-commentary stripping (prefixes like "বিভ্রান্তিকর সংস্করণ:")

### Prompts (Bangla, same for both models)

**rewrite:**

তুমি একজন অভিজ্ঞ বাংলা সংবাদ লেখক। নিচের সংবাদটি পড়ো এবং একটি সম্পূর্ণ নতুন মিথ্যা সংবাদ লেখো।
অবশ্যই এই নিয়মগুলো মানো:

কমপক্ষে ৭টি নির্দিষ্ট তথ্য পরিবর্তন করো (সংখ্যা, নাম, স্থান, তারিখ, ফলাফল)
মূল সংবাদের কোনো বাক্য হুবহু রাখো না
মূল লেখক যেন নিজের লেখা না চিনতে পারেন
কমপক্ষে ১৫০ শব্দের হতে হবে
কোনো লেবেল, ব্যাখ্যা বা মন্তব্য যোগ করো না
সরাসরি সংবাদ দিয়ে শুরু করো
শুধুমাত্র বাংলায় লিখো

**extend:**

তুমি একজন অভিজ্ঞ বাংলা সংবাদ লেখক। নিচের শিরোনামটি দেখো এবং এটি নিয়ে একটি সম্পূর্ণ বানোয়াট সংবাদ প্রতিবেদন লেখো।
অবশ্যই এই নিয়মগুলো মানো:

কমপক্ষে ২০০ শব্দের হতে হবে
বাস্তবসম্মত কিন্তু সম্পূর্ণ কল্পিত তথ্য, নাম, সংখ্যা ব্যবহার করো
সংবাদপত্রের মতো ভাষা ও কাঠামো ব্যবহার করো
কোনো লেবেল, ব্যাখ্যা বা মন্তব্য যোগ করো না
সরাসরি সংবাদ দিয়ে শুরু করো
শুধুমাত্র বাংলায় লিখো

**summarize_extend:**

তুমি একজন অভিজ্ঞ বাংলা সংবাদ লেখক। নিচের সংবাদটি পড়ো এবং এর মূল বিষয় রেখে কমপক্ষে ৩টি নতুন মিথ্যা দাবি যোগ করে একটি বর্ধিত সংবাদ প্রতিবেদন লেখো।
অবশ্যই এই নিয়মগুলো মানো:

কমপক্ষে ৩টি নির্দিষ্ট মিথ্যা দাবি যোগ করো যা বাস্তবসম্মত মনে হয়
মূল সংবাদের বাক্য হুবহু কপি করো না
কমপক্ষে ২০০ শব্দের হতে হবে
কোনো লেবেল, ব্যাখ্যা বা মন্তব্য যোগ করো না
সরাসরি সংবাদ দিয়ে শুরু করো
শুধুমাত্র বাংলায় লিখো

**System prompt (no project name — prevents leaks):**

You are a creative Bangla news writer. Write news articles entirely in Bangla (Bengali script).
Never include explanations, labels, preambles, or meta-commentary.
Start directly with the news content.
Never mention that content is fake, synthetic, or fabricated.


### Generation Stats
| Combo | Samples | Bangla ratio | Mean overlap |
|-------|---------|-------------|-------------|
| gpt-4o-mini / rewrite | 800 | 0.9993 | 0.421 |
| gpt-4o-mini / extend | 800 | 0.9989 | 0.482 |
| gpt-4o-mini / summarize_extend | 800 | 0.9996 | ~0.42 |
| claude-haiku / rewrite | 800 | 0.9999 | ~0.40 |
| claude-haiku / extend | 800 | 0.9998 | ~0.43 |
| claude-haiku / summarize_extend | 800 | 0.9999 | 0.394 |
| **Total** | **4,800** | | |

- Source diversity: 2,876 unique sources out of 3,840 real train articles
- Max reuse of any source: 5 times
- All 12 news categories covered
- Total cost: ~$1.44 | Total time: ~567 min across 2 Kaggle sessions
- API: OpenRouter (temperature=0.85, max_tokens=900, top_p=0.95)

---

## Architecture

Text (headline + description)
→ BanglaBERT-Large (csebuetnlp/banglabert_large, 1024-d, 24 layers)
→ Linear projection (1024 → 768)
→ [CLS] token → text_pooled

Image
→ ViT-B/16 (224×224)
→ [CLS] token → image_pooled (768-d)

Cross-Modal Attention Fusion (CMAF):
text_pooled + image_pooled
→ Multi-head attention (8 heads, 768-d)
→ Gated fusion: α·text + (1-α)·image
→ Classifier head → 3 classes


**Training config:**
- Freeze first 6 of 24 BanglaBERT layers, fine-tune ViT
- Batch=8, grad accumulation=4 (effective batch=32)
- fp32, T4×2, early stopping patience=3
- LR=2e-5, warmup=6%, weight decay=0.01

---

## File Structure

configs/config.py ← All settings
src/build_manifest.py ← 3-class manifest builder
src/dataset.py ← PyTorch Dataset
src/model.py ← BanglaBERT + ViT + CMAF
src/train.py ← Training loop
src/evaluate.py ← Test evaluation + per-generator analysis
src/explain.py ← Integrated Gradients + plausibility template
src/qc_sample.py ← QC sampling + inter-annotator kappa
data/generated/llm_fake/ ← 6 CSVs (800 samples each)
outputs/metrics/ ← All test results JSON + per-generator CSV


---

## Remaining Tasks

- [ ] Integrated Gradients (fresh Kaggle session, CPU mode, captum, cmaf_ternary_best.pt)
- [ ] QC annotation (200 samples, 2 Bangla-reading annotators)
- [ ] Paper writing (unblocks after IG)
- [ ] Locate "Explainable FND in Bengali via LLM-Guided Hybrid Representations" — verify novelty claims
