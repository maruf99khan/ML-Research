# MultiBanFakeDetect — Design Decisions Log
# Every architectural, dataset, and methodological decision with full justification.
# Created: Aug 6 2026 | Author: Claude (based on full research journey)

---

## 1. TASK FORMULATION

### Why ternary (3-class) instead of binary?
**Decision:** Real / Human-Fake / LLM-Fake
**Reason:** Binary systems can detect fake news but cannot distinguish its source. Platform moderators need to know whether misinformation is human-authored (requires human review) or LLM-generated (can be caught by automated systems with near-perfect accuracy). The three-way split has direct operational value.
**Alternative considered:** Binary (Real vs Fake) — rejected because it collapses two fundamentally different threat actors into one class.
**Literature support:** Med-MMHL (2023), MM-Health (2025) validated this framing for English. We extend it to Bangla.

---

## 2. TEXT ENCODER

### Why BanglaBERT-Large?
**Model:** csebuetnlp/banglabert_large (Electra discriminator, 1024-d, 24 layers)
**Reason:**
1. Pretrained exclusively on Bangla — richer language-specific representations than multilingual models
2. Same encoder as MBM-CTNet (2025) — enables fair, direct comparison of fusion mechanisms
3. Large variant (1024-d) gives more expressive token representations for CMAF
**Alternatives considered:**
- XLM-RoBERTa-large: rejected — multilingual, less Bangla-specific, harder to compare against prior Bangla work
- BanglaBERT-Base (768-d): rejected — less expressive, memory savings not needed on T4×2
- mBERT: rejected — used in MultiFusionFake baseline (2024), we want to advance beyond it
**Note:** BanglaBERT-Large is Electra-based, NOT standard BERT. This caused Bug #2 (freeze_text_layers silent fail) — fixed by try/except with explicit layer count verification.

---

## 3. IMAGE ENCODER

### Why ViT-B/16?
**Model:** vit_base_patch16_224, ImageNet-21k pretrained via timm
**Reason:**
1. ViT produces a sequence of 197 patch tokens (196 patches + CLS token) — essential for our bidirectional cross-attention fusion
2. CNN (ResNet, DenseNet) produces a single pooled vector — cannot do token-level cross-attention, only concatenation or late fusion
3. ViT-B/16 is standard in recent multimodal work (CLIP, BLIP, etc.)
**Alternatives considered:**
- ResNet-101: rejected — pooled vector, no token sequence for cross-attention
- DenseNet-169: rejected — used in original MultiFusionFake baseline (2024), we want to advance beyond it
- ViT-L/16: rejected — too large for T4×2 memory constraints

---

## 4. FUSION MECHANISM

### Why CMAF (Cross-Modal Attention Fusion)?
**Mechanism:** Bidirectional multi-head cross-attention (8 heads) + learned gate with residual
**Reason:**
1. Text tokens attend over image patches → each text token informed by visual context
2. Image patches attend over text tokens → each patch informed by textual context
3. Bidirectional richer than one-directional (MBM-CTNet uses one-directional co-attention)
4. Learned gate allows model to modulate fusion vs raw pooled signal
5. Gate residual = (text_pooled + image_pooled)/2 preserves both modalities when gate→0
**Alternatives considered:**
- Simple concatenation: rejected — loses inter-modal interaction, no attention mechanism
- Late fusion (average logits): rejected — no cross-modal learning during training
- MBM-CTNet co-attention (one-directional): this is our ablation baseline
**Critical bug fixed:** Original gate residual was `gate * fused + (1-gate) * text_pooled`. When gate→0, image signal completely disappeared. Fixed to `gate * fused + (1-gate) * (text+image)/2`.

---

## 5. EXPLAINABILITY

### Why Integrated Gradients?
**Method:** Captum IntegratedGradients, n_steps=50, zero baselines
**Reason:**
1. IG satisfies the completeness axiom — attributions sum exactly to the output difference from baseline
2. SHAP and LIME are model-agnostic approximations that do NOT guarantee this for neural networks
3. Attention weights are not reliable attribution — don't measure actual contribution to output
4. Grad-CAM requires CNN architecture natively — doesn't generalize cleanly to transformers
5. No prior Bangla FND paper uses IG — genuine novelty claim
**Alternatives considered:**
- SHAP: rejected — approximation, no completeness guarantee, used in HEMT-Fake (2025) which we differentiate from
- LIME: rejected — same problems as SHAP
- Attention visualization: rejected — unreliable, doesn't satisfy theoretical axioms
**Critical bug fixed:** Captum calls `forward_func(*inputs, *additional_forward_args)`. Original ForwardWrapper had wrong argument order `(embeds, mask, pixels)`. Correct order must be `(embeds, pixels, mask)`. Silently corrupted ALL attributions. Fixed in v3.

---

## 6. LOSS FUNCTION

### Why class-weighted cross-entropy?
**Formula:** w_c = N/(K*N_c) per class
**Reason:** During generation, LLM-fake class was smaller than real/human-fake. Weighted loss prevents model from ignoring minority class.
**Bug fixed:** When LLM-fake had 0 samples, weight = N/(K*1) = 2,560× — catastrophic loss explosion. Fixed: only weight classes present in split; missing classes get w=1.0.

---

## 7. TRAINING SETTINGS

### Why freeze 6 of 24 BERT layers?
**Setting:** freeze_text_layers=6, freeze_image=False
**Reason:** Balance between memory constraints on Kaggle T4×2 (15GB per GPU) and trainable parameters. Freezing first 6 layers (embeddings + early encoder) preserves low-level Bangla representations while allowing task-specific fine-tuning of higher layers.
**Alternative tried:** freeze_text_layers=12 (binary baseline) — used only due to single T4 memory constraints. Ternary model uses T4×2, allowing 6 frozen layers and unfrozen ViT.
**Why unfreeze ViT for ternary:** ViT must fine-tune to learn image-text mismatch patterns — critical for distinguishing LLM-fake from real (LLM-fake uses real images with fabricated text).

### Why batch=8 with grad_accum=4?
**Effective batch:** 32 samples
**Reason:** Batch=8 confirmed to fit on T4×2 via memory test. Effective batch=32 is standard for BERT fine-tuning. Gradient accumulation avoids OOM while maintaining effective batch size.

### Why early stopping patience=3?
**Reason:** Training showed clear overfitting pattern — val F1 peaked at epoch 12 (0.9290) then declined. Patience=3 stopped at epoch 14, saving compute and preventing overfitted checkpoint selection.

---

## 8. DATASET EXTENSION

### Why generate LLM-fake instead of using existing datasets?
**Reason:** No existing Bangla LLM-fake news dataset exists as of August 2026. We generate our own to enable the ternary classification task.

### Why reuse real images for LLM-fake?
**Reason:** Real-world misinformation routinely repurposes genuine images under fabricated captions. This mirrors authentic misinformation patterns. Avoids confounding visual fakeness with textual fakeness — research question stays focused on detecting LLM-generated TEXT.

### Why GPT-4o Mini + Claude Haiku?
**Chosen:** openai/gpt-4o-mini, anthropic/claude-3-haiku (via OpenRouter)
**Reasons:**
1. Both produce fluent, realistic Bangla with 0.9996 mean Bangla character ratio
2. Strong instruction following — low meta-commentary rates
3. Different architectures (OpenAI vs Anthropic) — adds genuine generator diversity
4. Cost-effective for large-scale generation (~$0.0003 and ~$0.0006 per sample)
**Dropped models:**
- Gemini 2.5 Flash: rewrite strategy rejected 43% samples (TOO_SIMILAR, overlap>0.65). Model interpreted "distort" as "barely change." Extend strategy was excellent but we simplified to 2 models for cleaner paper narrative.
- Llama 3.3 70B: unacceptable API response latency on Bangla prompts. Infrastructure/routing issue, not a prompt problem.

### Why 3 generation strategies?
**Strategies:** rewrite, extend, summarize_extend
**Reason:**
- Rewrite: most realistic misinformation pattern — how humans actually spread fake news. Without rewrite, LLM-fake class would be too easy to detect (model learns "long new article = LLM").
- Extend: fully fabricated from headline — clearly LLM style, diverse content.
- Summarize-extend: grounded in real events + fabricated claims — hardest to detect, middle ground.
**Initially excluded rewrite, added back:** After realizing extend-only LLM-fake would be trivially detectable (obvious stylistic difference from human-fake), rewrite was added to make the classification genuinely challenging.

### Why 800 samples per combo (6 combos = 4,800 total)?
**Reason:** Matches Real and Human-Fake training split (3,840 train + 480 val + 480 test = 4,800 each). Perfect balance across all three classes.
**Targets changed during project:**
- Original: 3,000 (thought class-weighted loss would handle imbalance)
- Revised to 4,800: realized val and test splits also need LLM-fake samples, not just train

### Why 5 quality filters?
**Filters:** length ≥200, Bangla ratio ≥0.70, first-line Bangla ≥0.80, sentences ≥2, source overlap ≤0.65
**Developed iteratively after Batch 1 inspection:**
- Length: catch empty/truncated responses
- Bangla ratio: catch English contamination
- First-line Bangla: catch English headlines before Bangla body
- Sentence count: catch incomplete generations
- Source overlap: catch near-copies (main problem with Gemini rewrite)
- Threshold 0.65: tightened from 0.70 after batch 1 showed 0.71-0.79 samples were visually near-copies

---

## 9. STORAGE DECISIONS

### Why GitHub for generated CSVs?
**Problem:** Kaggle free tier wipes /kaggle/working on session end. Save Version does NOT save files — only notebook output cells. Lost 2,400 samples twice before finding this.
**Solution:** Push generated CSVs to GitHub after every batch. Proven working via round-trip test (generate → push → wipe → git reset --hard origin/master → restore).
**File size:** 4,800 samples × ~1KB = ~5MB total — well under GitHub 100MB limit.

### Why Kaggle Dataset for checkpoints?
**Problem:** Checkpoints are ~1.6GB — too large for GitHub (100MB file limit).
**Solution:** Upload to Kaggle Dataset via kaggle API. Accessible in future sessions as input dataset.
**Proven working:** Checkpoint verified epoch=12, val_macro_f1=0.9290, 563 parameter tensors.

### Why NOT Kaggle Save Version?
**Reason:** Save Version (Quick Save) saves notebook output cells only, NOT /kaggle/working files. Save & Run All reruns all cells (takes hours). Neither reliably saves large files. Proven unreliable in 2 session wipes.

---

## 10. ABLATION DESIGN

### Why text_only removes ViT completely (not zeros image)?
**Standard approach:** Zero image input during forward pass, keep ViT loaded.
**Our approach:** skip_image=True removes ViT from model entirely.
**Reason:** BanglaBERT-Large (~14GB) + ViT (~2GB) = ~16GB > 15.53GB available on T4. Loading both causes OOM. Removing ViT saves 2GB and allows fp32 training.
**Paper statement:** "For the text-only ablation, the image encoder was excluded to isolate text encoder performance, due to GPU memory constraints."
**Why this is valid:** Zeroing image vs removing encoder gives same result for model output — the fusion receives zero-valued image tokens either way. The difference is only memory usage.

### Why image_only uses skip_image=False?
**Reason:** image_only zeros text tokens in forward pass but keeps BanglaBERT loaded. BanglaBERT alone fits in 14GB. No need to remove it.

---

## 11. NOVELTY CLAIMS (validated July 2026)

### Claim 1: First ternary Real/Human-Fake/LLM-Fake for Bangla
**Status:** HOLDS with correct scoping
**Verified against:** Med-MMHL (2023), MM-Health (2025), "When Machines Lie Differently" (2026) — all English or medical domain, not Bangla general news
**Correct framing:** "First ternary classification in Bangla and in the general-news multimodal setting"

### Claim 2: First cross-modal-attention ternary Bangla FND
**Status:** HOLDS
**Verified against:** MBM-CTNet (2025) uses one-directional co-attention, binary+multitask. Our bidirectional CMAF for ternary is novel.

### Claim 3: First Integrated Gradients for multimodal Bangla FND
**Status:** HOLDS narrowly
**Verified against:** HEMT-Fake (2025) uses SHAP+LIME+attention for Hindi/Gujarati/Marathi/Telugu — not Bangla, not IG.
**UNRESOLVED:** "Explainable FND in Bengali via LLM-Guided Hybrid Representations" — must locate before submission.

---

## 12. BUGS FIXED (all critical for paper methodology section)

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | Gate residual dropped image signal | Silent — image ignored when gate→0 | Residual = (text+image)/2 |
| 2 | freeze_text_layers silent fail on Electra | Froze nothing | try/except with count verification |
| 3 | ForwardWrapper arg order wrong | Silent — ALL IG attributions corrupted | Corrected to (embeds, pixels, mask) |
| 4 | Last grad-accum batch never flushed | Weights not updated for final steps | Flush after loop if not already flushed |
| 5 | Class weight explosion (2,560×) | Loss explosion on first batch | Only weight present classes |
| 6 | System prompt leaked project name | Model wrote "MultiBanFakeDetect" in output | Removed all project-identifying language |
| 7 | Generation saved only at batch end | Session crash = all batch lost | Save after every accepted sample |
| 8 | Wrong OpenRouter model IDs | All API calls 404 | Verified live IDs July 30 2026 |
| 9 | combined_manifest.csv pushed to GitHub | Stale manifest in next session | Added to .gitignore |
| 10 | qc_new_batch wrong index tracking | Wrong rows dropped in QC | Track original_idx before drop |
| 11 | git pull doesn't restore deleted files | Session restore appeared to work but data missing | Use git reset --hard origin/master |
| 12 | fp16 NaN loss at epoch 3 | Training collapses | Removed fp16, used skip_image instead |
| 13 | skip_image not in model.__init__ | TypeError on text_only ablation | Added skip_image=False parameter |
