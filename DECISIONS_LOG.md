# MultiBanFakeDetect — Decisions, Problems & Fixes Log

This file feeds directly into the paper's Methodology, Implementation Details,
and Limitations sections. Every significant decision, problem encountered, and
fix applied is recorded here with the reasoning. Never delete entries — add new
ones as the project progresses.

---

## Architecture Decisions

### D1 — Why BanglaBERT-Large over BanglaBERT-Base or XLM-RoBERTa
**Decision:** Use `csebuetnlp/banglabert_large` (Electra discriminator, 1024-d, 24 layers)
**Alternatives considered:** BanglaBERT-base (768-d), XLM-RoBERTa-large (multilingual)
**Reasoning:**
- BanglaBERT-Large is trained exclusively on Bangla — better domain alignment than multilingual models
- Larger hidden dimension (1024-d) gives richer token representations before fusion
- MBM-CTNet (our main baseline) also uses BanglaBERT, so we can isolate the contribution of our fusion mechanism vs theirs
**Tradeoff:** Higher memory cost — required freezing 12/24 layers on Kaggle T4

### D2 — Why ViT-B/16 over ResNet/DenseNet
**Decision:** Use `vit_base_patch16_224` (Vision Transformer, 768-d patch tokens)
**Alternatives considered:** ResNet-101, DenseNet-169 (used in original MultiBanFakeDetect baseline)
**Reasoning:**
- ViT produces a sequence of patch tokens (197 tokens including CLS) — enables token-level cross-attention with text, which CNNs cannot do
- Cross-attention fusion (CMAF) requires sequence inputs from both modalities; CNN outputs are pooled vectors, not sequences
- ViT-B/16 is the standard in recent multimodal work (CLIP, BLIP, etc.)
**Tradeoff:** Higher memory than CNN; resolved by freezing ViT and using gradient checkpointing

### D3 — Why Cross-Modal Attention Fusion (CMAF) over simple concatenation
**Decision:** Bidirectional cross-attention (text attends to image, image attends to text) + learned gate
**Alternatives considered:** Early fusion (concatenate embeddings), late fusion (average logits), MBM-CTNet co-attention
**Reasoning:**
- Concatenation loses inter-modal interaction — the model can't learn which image regions are relevant to which text tokens
- MBM-CTNet uses co-attention (one-directional text→image) — our bidirectional design is an explicit architectural improvement
- The learned gate allows the model to modulate how much the fused representation relies on the cross-attended vs raw pooled signal
**Paper framing:** CMAF vs. MBM-CTNet co-attention is one of our ablation comparisons

### D4 — Why Integrated Gradients over SHAP/LIME/attention
**Decision:** Captum IntegratedGradients, applied to text token embeddings and image pixel values
**Alternatives considered:** SHAP (used by HEMT-Fake), LIME, attention weight visualization
**Reasoning:**
- IG is the only method that provides exact attribution along a straight-line path from a baseline, with a completeness axiom guarantee (attributions sum to the output difference)
- SHAP and LIME are model-agnostic approximations — less theoretically grounded for neural networks
- Attention weights are not reliable attribution scores (they don't measure contribution to the output)
- No prior Bangla FND paper uses IG — gives us a distinct methodological claim
**Tradeoff:** IG is slower than attention visualization; we run it only on the test set qualitative examples

### D5 — Why 4 generator models instead of 1
**Decision:** Gemini 2.5 Flash, GPT-4o Mini, Llama 3.3 70B, Claude 3 Haiku
**Alternatives considered:** Single model (GPT-4o), single model (GPT-5)
**Reasoning:**
- Using 4 generators tests whether detectability varies by generator (the "per-generator asymmetry" analysis in Phase 5)
- Mirrors real-world threat model — actual LLM misinformation comes from diverse generators
- Enables the paper's Phase 5 analysis: which generator's fakes are hardest to detect?
- More expensive models (GPT-5, Claude Opus) would produce higher-quality fakes that might be trivially detectable as "too fluent" — counterproductive for the research question
**Total cost:** ~$2.51 for 3,800 samples

### D6 — Why reuse images from real dataset for LLM-fake class
**Decision:** Pair LLM-generated text with images reused/lightly edited from the real dataset
**Alternatives considered:** Generate synthetic images (Stable Diffusion, DALL-E), source new images
**Reasoning:**
- Real misinformation repurposes real images under false captions — this is realistic
- Synthetic image generation would introduce a confound (model must detect both fake text AND fake image)
- Keeps the research question focused: can we detect LLM-generated TEXT given a real image?
- Avoids copyright issues with sourcing new images
**Paper framing:** Explicitly stated as a design choice in Dataset Construction section

---

## Problems Encountered & Fixes

### P1 — OOM at BATCH_SIZE=16 on Kaggle T4
**Error:** `OutOfMemoryError: CUDA out of memory` at step 3 of epoch 1
**Root cause:** BanglaBERT-Large (429M params) + ViT-B/16 together exceed 14.56GB VRAM at batch=16
**Fix applied:**
- BATCH_SIZE: 16 → 4
- GRAD_ACCUM_STEPS: 2 → 8 (keeps effective batch = 32)
- freeze_text_layers=12 (freeze first 12 of 24 BERT layers)
- freeze_image=True (freeze ViT entirely)
- gradient_checkpointing=True on text encoder
**Result:** 159M trainable / 429M total params, 9.39GB free after model load
**Paper note:** Report frozen vs unfrozen configuration; note this as a compute constraint

### P2 — Class weight explosion when LLM-fake class has 0 samples
**Error:** Class weight for `llm_fake` = 2,560 (inverse of near-zero count)
**Root cause:** `compute_class_weights` used `reindex(range(NUM_CLASSES), fill_value=1)` which set count=1 for missing classes, giving 1/1 * total/3 = enormous weight
**Fix applied:** Only weight classes that actually appear in the split; missing classes get weight=1.0 (neutral)
**Paper note:** Document that binary training used equal weights; ternary training uses inverse-frequency weights

### P3 — OpenRouter 404 errors on generation
**Error:** `404 Client Error: Not Found` for all generation attempts
**Root cause:** Model IDs in config were outdated:
- `google/gemini-flash-1.5` → does not exist
- `anthropic/claude-3-5-haiku` → wrong format
**Fix applied:** Verified live model IDs via OpenRouter `/api/v1/models` endpoint:
- `google/gemini-2.5-flash` ✓
- `openai/gpt-4o-mini` ✓
- `meta-llama/llama-3.3-70b-instruct` ✓
- `anthropic/claude-3-haiku` ✓
**Paper note:** Include model version verification date (July 30, 2026)

### P4 — Gate residual dropped image signal
**Error:** Silent — no crash, but wrong behavior
**Root cause:** `fused = gate_val * fused_candidate + (1 - gate_val) * text_pooled`
When gate=0, the fused output equals text_pooled only — image_pooled disappears entirely
**Fix applied:** `residual = (text_pooled + image_pooled) / 2` — both modalities always present
**Paper note:** Report this as part of the CMAF design description; affects ablation results

### P5 — _freeze_text_layers silent failure on Electra
**Error:** Silent — no crash, freeze did nothing
**Root cause:** Code assumed BERT structure (`model.embeddings`, `model.encoder.layer`) but BanglaBERT-Large is Electra-based with different attribute names
**Fix applied:** Wrapped in try/except with explicit print; verified freeze actually worked (187 frozen params confirmed)
**Paper note:** Note that BanglaBERT-Large is an Electra discriminator, not a standard BERT

### P6 — Last gradient accumulation batch never flushed
**Error:** Silent — no crash, but last partial batch of every epoch was not used for weight updates
**Root cause:** `optimizer.step()` only called when `(step + 1) % GRAD_ACCUM_STEPS == 0`; if total steps not divisible by GRAD_ACCUM_STEPS, final steps are dropped
**Fix applied:** Added explicit flush after the training loop
**Paper note:** Minor implementation detail; mention gradient accumulation with effective batch size

### P7 — Captum IntegratedGradients argument order bug
**Error:** Would have silently swapped `pixel_values` and `attention_mask`
**Root cause:** Captum calls `forward_func(*inputs, *additional_forward_args)`, concatenating positionally. Original `ForwardWrapper.forward(inputs_embeds, attention_mask, pixel_values)` received pixel_values where attention_mask was expected
**Fix applied:** Corrected to `forward(inputs_embeds, pixel_values, attention_mask)` — matching Captum's call order exactly
**Paper note:** Document in implementation details; this is a known Captum gotcha

### P8 — FUSION_HIDDEN_DIM missing from config
**Error:** `AttributeError: module 'configs.config' has no attribute 'FUSION_HIDDEN_DIM'`
**Root cause:** Removed during config cleanup but still used as default arg in CrossModalAttentionFusion
**Fix applied:** Added back to config.py

---

## Model Selection Rationale (for Related Work / Methodology)

### Why our model outperforms MBM-CTNet conceptually
| Aspect | MBM-CTNet | Ours |
|--------|-----------|------|
| Task | Binary (real/fake) + multitask | Ternary (real/human-fake/llm-fake) |
| Fusion | Co-attention (one direction) | CMAF (bidirectional) |
| Explainability | None | Integrated Gradients |
| LLM-fake class | Not addressed | Core contribution |
| Dataset | MultiBanFakeDetect (9,600) | Extended (13,400+) |

### Why our explainability is distinct from HEMT-Fake
| Aspect | HEMT-Fake | Ours |
|--------|-----------|------|
| Language | Hindi/Gujarati/Marathi/Telugu/En | Bangla |
| XAI method | SHAP + LIME + attention | Integrated Gradients |
| Task | Binary FND | Ternary FND |
| IG guarantee | None (approximation) | Completeness axiom |

---

## Training Observations (update as training progresses)

### Binary baseline (Real vs Human-Fake, frozen ViT + 12 frozen BERT layers)
| Epoch | Val Loss | Macro-F1 | Notes |
|-------|----------|----------|-------|
| 1 | 0.4248 | 0.8170 | — |
| 2 | 0.3468 | 0.8541 | — |
| 3 | 0.4962 | 0.8792 | Best — saved |
| 4 | 0.5979 | 0.8770 | Val loss diverging |
| 5 | 0.6314 | 0.8771 | Early stopping triggered |

**Observation:** Val loss starts rising at epoch 3 while macro-F1 plateaus — model overfitting to train set. Consider label smoothing or stronger dropout for ternary training.

### Ternary model (after LLM-fake generation)
*To be filled after Phase 3*

---

## Generation Log Summary (update as generation progresses)

| Model | Strategy | Target | Accepted | Rejected | Acceptance % | Cost |
|-------|----------|--------|----------|----------|-------------|------|
| gemini-2.5-flash | rewrite | 100 | — | — | — | — |
| gemini-2.5-flash | extend | 400 | — | — | — | — |
| gemini-2.5-flash | summarize_extend | 700 | — | — | — | — |
| gpt-4o-mini | rewrite | 350 | — | — | — | — |
| gpt-4o-mini | extend | 350 | — | — | — | — |
| gpt-4o-mini | summarize_extend | 300 | — | — | — | — |
| llama-3.3-70b | rewrite | 270 | — | — | — | — |
| llama-3.3-70b | extend | 265 | — | — | — | — |
| llama-3.3-70b | summarize_extend | 265 | — | — | — | — |
| claude-haiku | rewrite | 270 | — | — | — | — |
| claude-haiku | extend | 265 | — | — | — | — |
| claude-haiku | summarize_extend | 265 | — | — | — | — |

---

## Generation Batch 1 — Deep Inspection Results (July 30, 2026)

### Batch: gemini-2.5-flash, rewrite strategy, 100 samples

| Metric | Value |
|--------|-------|
| API calls made | 194 |
| Accepted | 100 |
| After quality cleaning | 57 |
| Final rejection rate | 43% |
| Bangla ratio (mean) | 0.998 |
| English contamination | 0 samples |

### Problems found (led to v4 fixes)

**CRITICAL — System prompt leak (1 sample)**
Sample [30] contained "মাল্টিব্যানফেকডিটেক্ট (MultiBanFakeDetect)" verbatim.
Root cause: system prompt explicitly named the project. Model repeated it.
Fix: system prompt rewritten to never mention project name, purpose, or "fake news detection classifier". Title changed to "Bangla News Generation".

**CRITICAL — Incomplete strip prefix list (2+ samples)**
Variants not caught: "সিন্থেটিক/ভুয়া সংবাদ (Synthetic/Fake News):", "সিন্থেটিক/নকল:", "**সিন্থেটিক...**"
Fix: Added all variants found in batch 1 to STRIP_PREFIXES.

**HIGH — 38 samples too similar to source (0.70-0.79)**
Root cause: rewrite strategy — model does light paraphrasing not genuine distortion.
Fix: Threshold lowered 0.70 -> 0.65. Rewrite prompt now explicitly says "change at least 5 specific facts" and "do not copy sentences".

**MEDIUM — 2 outright refusals**
Model returned prompt instructions instead of fake news.
Fix: REFUSAL_PATTERNS list expanded and checked before quality filters.

**LOW — Min length 200 too short (several borderline samples)**
Fix: Raised to 250 chars.

### Strategy assessment
- rewrite: 43% acceptance — model copies too much, NOT recommended for Gemini 2.5 Flash
- extend: Expected higher — model creates new content from scratch
- summarize_extend: Expected moderate — model creates extended content
- Recommended generation order: extend first, summarize_extend second, rewrite last

### v4 generation script changes summary
- System prompt: removed all project-specific language
- LEAK_PATTERNS: added (rejects outputs mentioning project name)
- STRIP_PREFIXES: added all batch 1 variants
- MIN_OUTPUT_CHARS: 200 -> 250
- MAX_SOURCE_OVERLAP: 0.70 -> 0.65
- Rewrite prompt: added "change at least 5 specific facts", "do not copy sentences"
- Added run_quality_check() function: callable after every batch, automated
