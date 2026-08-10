# MultiBanFakeDetect — Paper Issues, Weaknesses & Fixes
# Created: Aug 9 2026
# Purpose: Track every known weakness before EMNLP 2026 Findings submission.
# Status codes: 🔴 Critical | 🟡 Moderate | 🟢 Minor | ✅ Resolved

---

## ISSUE 1 — Image modality barely helps LLM-fake detection
**Severity:** 🔴 Critical (most likely reviewer pushback)
**Status:** Needs narrative fix before paper writing

### The problem
Ablation table exposes this directly:
  Text-only LLM F1:  0.999
  CMAF      LLM F1:  0.996  ← image modality made it slightly WORSE

The image modality contributes almost nothing to LLM-fake detection.
A reviewer can argue: "this is not truly multimodal — it's a text model with a ViT attached."

### Why it happened
By design, LLM-fake samples reuse real images. The model has zero visual signal
to distinguish LLM-fake from Real. This mirrors real-world misinformation but
creates an uncomfortable ablation result.

### Where image DOES help
Text-only Human-Fake F1:  0.838
CMAF      Human-Fake F1:  0.896  ← +0.058 lift from image modality

This is the real multimodal contribution. Lead with this.

### Fix (narrative — no retraining needed)
Reframe the paper around two separate claims:
  Claim A: Multimodal fusion improves Real vs Human-Fake discrimination (+0.058 F1)
  Claim B: The ternary split is itself the contribution — knowing WHICH type of fake
           it is matters operationally, regardless of which modality detects it.

Add this sentence to abstract and conclusion:
  "While LLM-generated fake news carries strong textual signals detectable without
  visual context, Human-Fake detection benefits significantly from multimodal fusion,
  underscoring the complementary role of visual and textual modalities in a ternary
  classification framework."

### Action items
- [ ] Rewrite abstract to lead with Human-Fake lift, not overall macro-F1
- [ ] Add per-modality-per-class breakdown table to paper (already have the numbers)
- [ ] Explicitly call out LLM textual fingerprint as a finding, not a limitation

---

## ISSUE 2 — No inter-annotator agreement for Human-Fake labels
**Severity:** 🟡 Moderate
**Status:** Cannot be fixed experimentally — handle in Limitations section

### The problem
Human-Fake labels come from the original MultiBanFakeDetect dataset.
We did not conduct independent re-annotation.
If original labels are noisy, our Human-Fake F1 (0.896) is measured against
a noisy gold standard. Cannot distinguish model errors from label errors.

### Fix
One paragraph in Limitations section:
  "Human-Fake labels are inherited from the MultiBanFakeDetect dataset [cite].
  We did not conduct independent re-annotation; label noise in the Human-Fake
  class may affect the upper bound of achievable F1 for that class."

### Action items
- [ ] Check if original dataset paper reports IAA — if yes, cite it
- [ ] Add Limitations paragraph (1-2 sentences, do not over-apologize)

---

## ISSUE 3 — Per-strategy detection rates not reported
**Severity:** 🟡 Moderate (missed opportunity more than weakness)
**Status:** Easy fix — 10 lines of code, no retraining

### The problem
We have 3 generation strategies: rewrite, extend, summarize_extend.
We never measured whether they differ in detectability.
Hypothesis: rewrite samples (stylistically closest to human writing) will have
lower recall than extend or summarize_extend.
If true, this is a novel finding worth reporting.

### Fix — add to evaluate.py
Add this block after per-generator analysis in evaluate_checkpoint():

```python
strategy_col = "strategy"
if strategy_col in mdf.columns:
    llm_sub2 = pred_df[(pred_df["true"] == llm_id)].copy()
    llm_sub2["strategy"] = llm_sub2["sample_id"].map(
        lambda sid: mdf.loc[sid, strategy_col]
        if sid in mdf.index and pd.notna(mdf.loc[sid, strategy_col]) else None
    )
    llm_sub2 = llm_sub2[llm_sub2["strategy"].notna()]
    if len(llm_sub2) > 0:
        per_strat = llm_sub2.groupby("strategy").apply(
            lambda g: (g["pred"] == llm_id).mean()
        ).rename("recall_llm_fake")
        print(f"\n  Per-strategy LLM-fake recall:")
        print(per_strat)
        per_strat.to_csv(os.path.join(cfg.METRICS_DIR, f"{run_name}_per_strategy.csv"))
```

Then run: evaluate_checkpoint("cmaf_ternary", "full")

### Action items
- [ ] Add per-strategy analysis to evaluate.py
- [ ] Run evaluate_checkpoint("cmaf_ternary", "full") to get numbers
- [ ] If rewrite recall < extend recall by >0.05: add table to paper as finding
- [ ] If all strategies similar: mention in one sentence and move on

---

## ISSUE 4 — MBM-CTNet comparison is not apples-to-apples
**Severity:** 🟡 Moderate
**Status:** Cannot be fixed experimentally — handle in paper framing

### The problem
MBM-CTNet reports 0.942 on binary task (Real vs Fake) with different dataset split.
We report 0.9285 on ternary task (Real / Human-Fake / LLM-Fake) — incomparable.
A reviewer will flag this immediately if the table implies direct competition.

### Fix
Two things:
1. Table footnote: "binary task, different label space, reported for reference only — not directly comparable"
2. Use our binary baseline (0.8792 val) as the real comparison point.
   Story: "We extended the existing binary Bangla multimodal problem to ternary
   while maintaining competitive performance (0.9285 ternary vs 0.8792 binary baseline
   on the same data)."

### Action items
- [ ] Add clear footnote to results table
- [ ] Reframe paper narrative around binary→ternary extension story
- [ ] Do NOT claim to beat MBM-CTNet — different tasks

---

## ISSUE 5 — Unresolved competing paper (HIGHEST PRIORITY)
**Severity:** 🔴 Critical (existential for Claim 3 if paper exists)
**Status:** UNRESOLVED — must locate before writing explainability section

### The problem
"Explainable Fake News Detection in Bengali via LLM-Guided Hybrid Representations"
Referenced on ResearchGate. Full paper not found as of Aug 9 2026.
If this paper:
  - Is multimodal AND uses Integrated Gradients → Claim 3 completely invalidated
  - Is text-only OR uses SHAP/LIME → Claim 3 survives, just needs narrower framing
  - Does not exist / not published → safe to proceed

### Search plan (do this TODAY before writing explainability section)
1. Semantic Scholar: search "explainable fake news Bengali"
2. ACL Anthology: search "Bengali fake news explainable"
3. Google Scholar: search "Bangla fake news integrated gradients"
4. ResearchGate: search exact title "Explainable Fake News Detection in Bengali via LLM-Guided Hybrid Representations"

### If found — what survives
  - Multimodal + IG → drop Claim 3 entirely, reframe as "first TERNARY explainable Bangla FND"
  - Text-only + IG → narrow Claim 3 to "first IG for MULTIMODAL Bangla FND"
  - Uses SHAP/LIME → Claim 3 fully survives ("first IG for Bangla FND of any modality")

### If not found after thorough search
Write: "To our knowledge, no prior work has applied Integrated Gradients to
multimodal Bangla fake news detection." This is defensible.

### Action items
- [ ] Search all 4 sources above TODAY
- [ ] Paste abstract here for analysis if found
- [ ] Do NOT finalize explainability novelty claim until this is resolved
- [ ] Do NOT write the explainability Related Work section until this is resolved

---

## ISSUE 6 — Retrain non-determinism
**Severity:** 🟢 Minor
**Status:** Already handled — just needs paper footnote

### The problem
Original run: test macro-F1 = 0.9207 (checkpoint lost)
Retrain run:  test macro-F1 = 0.9285 (checkpoint saved)
Difference of 0.0078 due to DataParallel non-determinism.
A reviewer trying to reproduce may get a third number.

### Fix
Report 0.9285 as main result (this is the checkpoint we have).
Add footnote: "Results may vary by ±0.01 due to DataParallel non-determinism;
reported numbers correspond to the checkpoint available at [Kaggle dataset link]."

### Action items
- [ ] Add reproducibility footnote to paper
- [ ] Link Kaggle dataset in paper (maruf99khan/multibanfakedetect-checkpoints)

---

## ISSUE 7 — IG qualitative analysis on only 30 samples
**Severity:** 🟢 Minor
**Status:** Standard for EMNLP Findings — just needs kappa reported

### The problem
30 samples (10 per class) is qualitative, not quantitative.
Reviewer may ask: are highlighted tokens cherry-picked or systematically meaningful?
Human plausibility template (40 samples, 2 annotators) is our defense — but only
if kappa is reported and is reasonable.

### Kappa targets
  kappa > 0.6 → strong result, report prominently
  kappa 0.4-0.6 → acceptable for subjective task, report with note
  kappa < 0.4 → problem — IG attributions are not reliably interpretable

### Annotator calibration
Before sending template, give annotators concrete examples:
  Score 1: model highlights "এবং" (and), "কিন্তু" (but), "এর" (of) — function words, no signal
  Score 3: model highlights topic words but misses the fabricated claim
  Score 5: model highlights the specific fabricated statistic or emotionally charged phrase

### Action items
- [ ] Find 2 Bangla-reading annotators (teammates? supervisor?)
- [ ] Send calibration examples WITH the template
- [ ] Compute weighted kappa after annotation
- [ ] Report kappa in paper regardless of value — hiding it is worse than a low score

---

## PRIORITY ORDER (do these in this sequence)

1. 🔴 Search for the missing paper (Issue 5) — TODAY, before anything else
2. 🟡 Add per-strategy analysis to evaluate.py and run it (Issue 3) — 30 min
3. 🔴 Rewrite abstract/intro with corrected narrative (Issue 1) — when paper writing starts
4. 🟡 Add Limitations section (Issues 2, 6, 7) — when paper writing starts
5. 🟡 Fix results table footnote (Issue 4) — when paper writing starts
6. 🟢 Find annotators for IG plausibility (Issue 7) — after IG session completes

---

## WHAT IS NOT A PROBLEM

- Overall macro-F1 (0.9285) — strong result for ternary task
- LLM-Fake detection (0.996) — near-perfect, both generators caught
- Architecture validity — CMAF, BanglaBERT-Large, ViT-B/16 all well-motivated
- Novelty Claims 1 and 2 — both hold, well-scoped, verified against literature
- Dataset size (14,400) — competitive with related work
- Generation quality (bangla ratio 0.999, overlap 0.39-0.48) — solid

The paper is fundamentally sound. The issues above are presentation
and framing problems, not experimental design failures. All are fixable
before submission without any retraining.
