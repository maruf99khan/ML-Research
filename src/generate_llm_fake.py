"""
LLM-Fake class generation pipeline — v4

FIXES vs v3 (from batch 1 deep inspection, July 30 2026):

CRITICAL FIXES:
1. System prompt no longer mentions "MultiBanFakeDetect", "fake-news detection
   classifier", or any project-specific terms — these leaked into model outputs
   (sample [30] contained "MultiBanFakeDetect" verbatim)
2. Added system prompt leak detector: rejects any output containing project
   name, "synthetic", "fake news detection classifier" etc.
3. Added all strip prefix variants found in batch 1:
   "সিন্থেটিক/ভুয়া সংবাদ", "সিন্থেটিক/নকল:", "**সিন্থেটিক...**"
4. Similarity threshold lowered 0.70 -> 0.65 (batch 1 showed 0.71-0.79
   samples were still near-copies on visual inspection)
5. Min output chars raised 200 -> 250
6. Rewrite prompt strengthened: explicitly says "do not copy sentences",
   "change at least 5 specific facts"

STRATEGY NOTES (from batch 1):
- rewrite: 43% acceptance rate with Gemini — model copies too much
- extend/summarize_extend: expected higher acceptance — model creates new content
- Recommended order: extend first, summarize_extend second, rewrite last

CONFIRMED WORKING:
- Batch 1 (rewrite, gemini-2.5-flash): 57/100 accepted after cleaning
- Bangla ratio: 0.998 mean — pure Bangla
- Zero English contamination
"""
import json
import os
import random
import re
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg

PROMPT_VERSION = "v4.0"

# ---------------------------------------------------------------------------
# Prompt templates — v4 (stronger distortion instructions)
# ---------------------------------------------------------------------------
PROMPTS = {
    "rewrite": (
        "তুমি একজন সৃজনশীল বাংলা সংবাদ লেখক। নিচের সংবাদটি পড়ো এবং একটি নতুন সংবাদ লেখো "
        "যেখানে মূল ঘটনার পরিবর্তে ভুল তথ্য দেওয়া হবে। অবশ্যই এই নিয়মগুলো মানো:\n"
        "- কমপক্ষে ৫টি নির্দিষ্ট তথ্য পরিবর্তন করো (সংখ্যা, নাম, স্থান, তারিখ, ফলাফল)\n"
        "- মূল সংবাদের বাক্য হুবহু কপি করো না\n"
        "- নতুন সংবাদ কমপক্ষে ১৫০ শব্দের হতে হবে\n"
        "- কোনো লেবেল, ব্যাখ্যা বা মন্তব্য যোগ করো না\n"
        "- সরাসরি সংবাদ দিয়ে শুরু করো\n"
        "- শুধুমাত্র বাংলায় লিখো\n\n"
        "মূল সংবাদ:\n{source_text}\n\n"
        "নতুন সংবাদ:"
    ),
    "extend": (
        "তুমি একজন সৃজনশীল বাংলা সংবাদ লেখক। নিচের শিরোনামটি দেখো এবং এটি নিয়ে "
        "একটি সম্পূর্ণ বানোয়াট সংবাদ প্রতিবেদন লেখো। অবশ্যই এই নিয়মগুলো মানো:\n"
        "- প্রতিবেদনটি কমপক্ষে ২০০ শব্দের হতে হবে\n"
        "- বাস্তবসম্মত কিন্তু সম্পূর্ণ কল্পিত তথ্য দিয়ে পূর্ণ করো\n"
        "- সংবাদের মতো ভাষা ও কাঠামো ব্যবহার করো\n"
        "- কোনো লেবেল, ব্যাখ্যা বা মন্তব্য যোগ করো না\n"
        "- সরাসরি সংবাদ দিয়ে শুরু করো\n"
        "- শুধুমাত্র বাংলায় লিখো\n\n"
        "শিরোনাম: {headline}\n\n"
        "সংবাদ প্রতিবেদন:"
    ),
    "summarize_extend": (
        "তুমি একজন সৃজনশীল বাংলা সংবাদ লেখক। নিচের সংবাদটি পড়ো এবং এর মূল বিষয় রেখে "
        "নতুন মিথ্যা তথ্য যোগ করে একটি বর্ধিত সংবাদ প্রতিবেদন লেখো। অবশ্যই এই নিয়মগুলো মানো:\n"
        "- মূল বিষয়বস্তু রেখে কমপক্ষে ৩টি নতুন মিথ্যা দাবি যোগ করো\n"
        "- প্রতিবেদনটি কমপক্ষে ২০০ শব্দের হতে হবে\n"
        "- মূল সংবাদের বাক্য হুবহু কপি করো না\n"
        "- কোনো লেবেল, ব্যাখ্যা বা মন্তব্য যোগ করো না\n"
        "- সরাসরি সংবাদ দিয়ে শুরু করো\n"
        "- শুধুমাত্র বাংলায় লিখো\n\n"
        "মূল সংবাদ:\n{source_text}\n\n"
        "বর্ধিত সংবাদ:"
    ),
}

# CRITICAL FIX v4: system prompt no longer mentions project name or purpose
# Previous version mentioned "MultiBanFakeDetect" which leaked into outputs
SYSTEM_PROMPT = (
    "You are a creative Bangla news writer. "
    "Write news articles entirely in Bangla (Bengali script). "
    "Never include explanations, labels, preambles, or meta-commentary. "
    "Start directly with the news content. "
    "Never mention that content is fake, synthetic, or fabricated."
)

# ---------------------------------------------------------------------------
# Quality filters — v4 (stricter than v3)
# ---------------------------------------------------------------------------
MIN_OUTPUT_CHARS   = 250   # raised from 200
MIN_BANGLA_RATIO   = 0.5
MAX_SOURCE_OVERLAP = 0.65  # lowered from 0.70

BANGLA_RANGE = re.compile(r'[\u0980-\u09FF]')

# Refusal patterns
REFUSAL_PATTERNS = [
    "দয়া করে মূল",
    "নির্দেশাবলী অনুসরণ করে",
    "আমি আপনার নির্দেশ",
    "মূল প্রতিবেদনটি এখানে দিন",
    "আমাকে মূল",
    "প্রতিবেদনটি প্রদান করুন",
    "এখানে দিন যার উপর",
]

# System prompt leak patterns — CRITICAL: reject if model repeated our instructions
LEAK_PATTERNS = [
    "MultiBanFakeDetect",
    "মাল্টিব্যানফেকডিটেক্ট",
    "academic research project",
    "fake-news detection",
    "labeled dataset",
]

# Meta-commentary prefixes to strip — v4: added all variants found in batch 1
STRIP_PREFIXES = [
    "এখানে মূল প্রতিবেদনের একটি বিভ্রান্তিকর সংস্করণ তৈরি করা হলো:",
    "সিনথেটিক/নকল খবর:",
    "সিন্থেটিক/নকল খবর:",
    "সিন্থেটিক/ভুয়া সংবাদ (Synthetic/Fake News):",
    "**সিন্থেটিক/ভুয়া সংবাদ (Synthetic/Fake News):**",
    "নকল খবর:",
    "ভুয়া খবর:",
    "বিভ্রান্তিকর সংস্করণ:",
    "এখানে বিভ্রান্তিকর প্রতিবেদন:",
    "এখানে একটি বিভ্রান্তিকর",
    "Synthetic/Fake News:",
    "সিন্থেটিক/নকল:",
]

# Keywords that indicate meta-commentary in first line
META_KEYWORDS = [
    'বিভ্রান্তিকর সংস্করণ', 'নকল খবর', 'সিনথেটিক',
    'সিন্থেটিক', 'ভুয়া সংবাদ', 'Synthetic', 'Fake News',
]


def strip_meta_commentary(text: str) -> str:
    """Remove meta-commentary prefixes the model added."""
    text = text.strip()
    for prefix in STRIP_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Strip first line if it contains meta keywords
    lines = text.split('\n')
    if lines and any(kw in lines[0] for kw in META_KEYWORDS):
        text = '\n'.join(lines[1:]).strip()
    return text


def is_quality_output(text: str, source_text: str) -> tuple[bool, str]:
    """Returns (pass, reason). Reason non-empty if failed."""

    # Check for system prompt leaks — CRITICAL
    for pattern in LEAK_PATTERNS:
        if pattern in text:
            return False, f"system_prompt_leak ({pattern})"

    # Check for model refusals
    for pattern in REFUSAL_PATTERNS:
        if pattern in text:
            return False, f"model_refusal"

    if len(text.strip()) < MIN_OUTPUT_CHARS:
        return False, f"too_short ({len(text.strip())} chars)"

    bangla_chars = len(BANGLA_RANGE.findall(text))
    total_alpha  = len(re.findall(r'[a-zA-Z\u0980-\u09FF]', text))
    if total_alpha > 0 and bangla_chars / total_alpha < MIN_BANGLA_RATIO:
        return False, f"low_bangla_ratio ({bangla_chars}/{total_alpha})"

    # n-gram similarity
    def ngrams(s, n=3):
        return Counter([s[i:i+n] for i in range(len(s)-n+1)])
    src_ng    = ngrams(source_text)
    out_ng    = ngrams(text)
    overlap   = sum((src_ng & out_ng).values())
    src_total = sum(src_ng.values())
    if src_total > 0 and overlap / src_total > MAX_SOURCE_OVERLAP:
        return False, f"too_similar_to_source (overlap={overlap/src_total:.2f})"

    return True, ""


def call_openrouter(model_id: str, user_prompt: str,
                    max_retries: int = 4, timeout: int = 90) -> str:
    api_key = os.environ.get(cfg.OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {cfg.OPENROUTER_API_KEY_ENV} environment variable first.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/maruf99khan/ML-Research",
        "X-Title": "Bangla News Generation",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.85,
        "max_tokens": 900,
        "top_p": 0.95,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{cfg.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers, json=payload, timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{max_retries}] {e} -- waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"OpenRouter call failed after {max_retries} retries")


def generate_batch_from_manifest(
    strategy:      str,
    generator_key: str,
    n:             int,
    manifest_path: str,
    output_dir:    str,
    seed:          int = cfg.SEED,
) -> pd.DataFrame:
    """
    Generate n LLM-fake samples. Saves to output_dir as CSV.
    Supports resuming — if CSV already exists, only generates remaining needed.
    Returns DataFrame of all accepted samples (including previously saved).
    """
    random.seed(seed)

    if strategy not in PROMPTS:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(PROMPTS)}")
    if generator_key not in cfg.GENERATOR_MODELS:
        raise ValueError(f"Unknown generator: {generator_key}. Choose from {list(cfg.GENERATOR_MODELS)}")

    generator_model_id = cfg.GENERATOR_MODELS[generator_key]["model_id"]

    manifest = pd.read_csv(manifest_path)
    real_df  = manifest[
        (manifest['label'] == 'real') & (manifest['split'] == 'train')
    ].copy()
    print(f"Source pool: {len(real_df)} real train articles")

    os.makedirs(output_dir, exist_ok=True)
    log_path     = os.path.join(output_dir, "generation_log.jsonl")
    samples_path = os.path.join(output_dir, f"{generator_key}_{strategy}_samples.csv")

    # Resume support
    if os.path.exists(samples_path):
        existing     = pd.read_csv(samples_path)
        already_done = len(existing)
        remaining    = n - already_done
        print(f"Resuming: {already_done} already saved, need {remaining} more")
        if remaining <= 0:
            print("Already complete.")
            return existing
    else:
        existing     = pd.DataFrame()
        already_done = 0
        remaining    = n

    accepted = []
    rejected_counts = {}
    attempts = 0
    used_ids = set()

    print(f"\nGenerating {remaining} samples | model={generator_key} | strategy={strategy} | v={PROMPT_VERSION}")
    print("-" * 60)

    while len(accepted) < remaining:
        attempts += 1
        available = real_df[~real_df['sample_id'].isin(used_ids)]
        src       = (available if len(available) > 0 else real_df).sample(1).iloc[0]
        used_ids.add(src['sample_id'])

        if strategy == "extend":
            headline = src['text'].split('।')[0][:120].strip()
            prompt   = PROMPTS["extend"].format(headline=headline)
        else:
            prompt = PROMPTS[strategy].format(source_text=src['text'][:800])

        current = already_done + len(accepted) + 1
        total   = already_done + remaining
        print(f"  [{current}/{total}] attempt={attempts} src={src['sample_id']}", end=" ")

        try:
            raw_text = call_openrouter(generator_model_id, prompt)
        except Exception as e:
            print(f"-> API ERROR: {e}")
            continue

        cleaned_text = strip_meta_commentary(raw_text)
        passed, reason = is_quality_output(cleaned_text, src['text'])

        if not passed:
            print(f"-> REJECTED ({reason})")
            key = reason.split(' ')[0]
            rejected_counts[key] = rejected_counts.get(key, 0) + 1
            continue

        print(f"-> OK ({len(cleaned_text)} chars)")

        sample_id = f"llmfake_{generator_key}_{strategy}_{uuid.uuid4().hex[:10]}"

        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(json.dumps({
                "sample_id":          sample_id,
                "source_article_id":  str(src['sample_id']),
                "generator_key":      generator_key,
                "generator_model_id": generator_model_id,
                "strategy":           strategy,
                "prompt_version":     PROMPT_VERSION,
                "timestamp_utc":      datetime.now(timezone.utc).isoformat(),
                "input_chars":        len(prompt),
                "output_chars_raw":   len(raw_text),
                "output_chars_clean": len(cleaned_text),
                "meta_stripped":      raw_text != cleaned_text,
                "attempt_number":     attempts,
            }, ensure_ascii=False) + "\n")

        accepted.append({
            "sample_id":         sample_id,
            "text":              cleaned_text,
            "image_path":        src['image_path'],
            "label":             "llm_fake",
            "label_id":          2,
            "generator":         generator_key,
            "strategy":          strategy,
            "source_article_id": src['sample_id'],
            "split":             None,
        })

        time.sleep(1.0)

    # Combine with existing and save
    new_df = pd.DataFrame(accepted)
    if len(existing) > 0:
        result_df = pd.concat([existing, new_df], ignore_index=True)
    else:
        result_df = new_df

    result_df.to_csv(samples_path, index=False, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Accepted this run : {len(accepted)}")
    print(f"  Total attempts    : {attempts}")
    print(f"  Acceptance rate   : {len(accepted)/attempts*100:.1f}%")
    print(f"  Total in file     : {len(result_df)}")
    print(f"  Rejection breakdown: {rejected_counts}")
    print(f"  Saved -> {samples_path}")
    return result_df


def run_quality_check(csv_path: str, manifest_path: str) -> pd.DataFrame:
    """
    Run full quality check on a generated CSV.
    Removes bad samples and saves cleaned version.
    Returns cleaned DataFrame.
    CALL THIS AFTER EVERY GENERATION BATCH before proceeding.
    """
    df = pd.read_csv(csv_path)
    manifest = pd.read_csv(manifest_path)
    real_df  = manifest[manifest['label'] == 'real'].set_index('sample_id')

    print(f"Quality check: {csv_path}")
    print(f"Loaded: {len(df)} samples")

    # Step 1: Strip meta-commentary
    df['text'] = df['text'].apply(strip_meta_commentary)

    # Step 2: Check every sample
    issues = []
    for idx, row in df.iterrows():
        text    = str(row['text'])
        src_id  = str(row['source_article_id'])
        src_txt = real_df.loc[src_id, 'text'] if src_id in real_df.index else ""

        probs = []

        # Leak check
        for pattern in LEAK_PATTERNS:
            if pattern in text:
                probs.append(f"LEAK({pattern[:15]})")
                break

        # Refusal check
        if is_quality_output(text, src_txt)[1].startswith("model_refusal"):
            probs.append("REFUSAL")

        if len(text) < MIN_OUTPUT_CHARS:
            probs.append(f"TOO_SHORT({len(text)})")

        bangla_chars = len(BANGLA_RANGE.findall(text))
        total_alpha  = len(re.findall(r'[a-zA-Z\u0980-\u09FF]', text))
        if total_alpha > 0 and bangla_chars / total_alpha < MIN_BANGLA_RATIO:
            probs.append(f"LOW_BANGLA({bangla_chars/total_alpha:.2f})")

        def ngrams(s, n=3):
            return Counter([s[i:i+n] for i in range(len(s)-n+1)])
        src_ng  = ngrams(src_txt)
        out_ng  = ngrams(text)
        overlap = sum((src_ng & out_ng).values())
        total   = sum(src_ng.values())
        sim     = overlap / total if total > 0 else 0
        if sim > MAX_SOURCE_OVERLAP:
            probs.append(f"TOO_SIMILAR({sim:.2f})")

        if probs:
            issues.append({
                'idx':     idx,
                'id':      row['sample_id'],
                'problems': ', '.join(probs),
                'preview': text[:80],
            })

    print(f"\n{'='*60}")
    print(f"QUALITY REPORT — {len(df)} samples | Issues: {len(issues)}")
    print(f"{'='*60}")

    if issues:
        print("\nPROBLEMATIC SAMPLES:")
        for i in issues:
            print(f"\n  [{i['idx']}] {i['id']}")
            print(f"  Problems : {i['problems']}")
            print(f"  Preview  : {i['preview']}")
    else:
        print("\n✅ ALL SAMPLES PASS")

    bad_idx   = [i['idx'] for i in issues]
    df_clean  = df.drop(index=bad_idx).reset_index(drop=True)
    df_clean.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"\nRemoved  : {len(bad_idx)}")
    print(f"Remaining: {len(df_clean)}")
    print(f"{'✅ Cleaned and saved' if bad_idx else '✅ No changes needed'}")

    return df_clean
