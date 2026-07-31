"""
LLM-Fake generation pipeline — v5 FINAL
2 models (GPT-4o Mini, Claude Haiku) x 3 strategies (rewrite, extend, summarize_extend)
Target: 3,000 clean samples total

FIXES vs all previous versions:
- Gemini dropped permanently (rewrite quality issues, fixed prompt untested)
- Llama dropped permanently (proven slow on Bangla prompts)
- All 3 strategies have strong prompts with explicit minimum change requirements
- Rewrite prompt now says "change at least 7 specific facts, no sentence verbatim"
- System prompt contains NO project-specific language (prevented leaks before)
- Autosave after every accepted sample (not just at batch end)
- Resume support: reads existing CSV count and only generates what's missing
- Credit exhaustion detected via 402/403 and logged clearly

SAVE VERSION REMINDER:
    After every batch of 150 samples → click Save Version in Kaggle immediately.

Usage:
    from src.generate_llm_fake import generate_batch
    generate_batch("gpt-4o-mini", "rewrite", 150)
"""
import json, os, random, re, sys, time, uuid
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg

PROMPT_VERSION = "v5.0"

# ============================================================
# PROMPTS — all three strategies, both models use same prompts
# ============================================================
PROMPTS = {
    "rewrite": (
        "তুমি একজন অভিজ্ঞ বাংলা সংবাদ লেখক। নিচের সংবাদটি পড়ো এবং "
        "একটি সম্পূর্ণ নতুন মিথ্যা সংবাদ লেখো। অবশ্যই এই নিয়মগুলো মানো:\n"
        "- কমপক্ষে ৭টি নির্দিষ্ট তথ্য পরিবর্তন করো (সংখ্যা, নাম, স্থান, তারিখ, ফলাফল)\n"
        "- মূল সংবাদের কোনো বাক্য হুবহু রাখো না\n"
        "- মূল লেখক যেন নিজের লেখা না চিনতে পারেন\n"
        "- কমপক্ষে ১৫০ শব্দের হতে হবে\n"
        "- কোনো লেবেল, ব্যাখ্যা বা মন্তব্য যোগ করো না\n"
        "- সরাসরি সংবাদ দিয়ে শুরু করো\n"
        "- শুধুমাত্র বাংলায় লিখো\n\n"
        "মূল সংবাদ:\n{source_text}\n\n"
        "নতুন মিথ্যা সংবাদ:"
    ),
    "extend": (
        "তুমি একজন অভিজ্ঞ বাংলা সংবাদ লেখক। নিচের শিরোনামটি দেখো এবং "
        "এটি নিয়ে একটি সম্পূর্ণ বানোয়াট সংবাদ প্রতিবেদন লেখো। অবশ্যই এই নিয়মগুলো মানো:\n"
        "- কমপক্ষে ২০০ শব্দের হতে হবে\n"
        "- বাস্তবসম্মত কিন্তু সম্পূর্ণ কল্পিত তথ্য, নাম, সংখ্যা ব্যবহার করো\n"
        "- সংবাদপত্রের মতো ভাষা ও কাঠামো ব্যবহার করো\n"
        "- কোনো লেবেল, ব্যাখ্যা বা মন্তব্য যোগ করো না\n"
        "- সরাসরি সংবাদ দিয়ে শুরু করো\n"
        "- শুধুমাত্র বাংলায় লিখো\n\n"
        "শিরোনাম: {headline}\n\n"
        "সংবাদ প্রতিবেদন:"
    ),
    "summarize_extend": (
        "তুমি একজন অভিজ্ঞ বাংলা সংবাদ লেখক। নিচের সংবাদটি পড়ো এবং "
        "এর মূল বিষয় রেখে কমপক্ষে ৩টি নতুন মিথ্যা দাবি যোগ করে "
        "একটি বর্ধিত সংবাদ প্রতিবেদন লেখো। অবশ্যই এই নিয়মগুলো মানো:\n"
        "- কমপক্ষে ৩টি নির্দিষ্ট মিথ্যা দাবি যোগ করো যা বাস্তবসম্মত মনে হয়\n"
        "- মূল সংবাদের বাক্য হুবহু কপি করো না\n"
        "- কমপক্ষে ২০০ শব্দের হতে হবে\n"
        "- কোনো লেবেল, ব্যাখ্যা বা মন্তব্য যোগ করো না\n"
        "- সরাসরি সংবাদ দিয়ে শুরু করো\n"
        "- শুধুমাত্র বাংলায় লিখো\n\n"
        "মূল সংবাদ:\n{source_text}\n\n"
        "বর্ধিত মিথ্যা সংবাদ:"
    ),
}

# NO project name, NO "fake news detection", NO "classifier" — prevents leaks
SYSTEM_PROMPT = (
    "You are a creative Bangla news writer. "
    "Write news articles entirely in Bangla (Bengali script). "
    "Never include explanations, labels, preambles, or meta-commentary. "
    "Start directly with the news content. "
    "Never mention that content is fake, synthetic, or fabricated."
)

# ============================================================
# QUALITY FILTERS
# ============================================================
BANGLA_RE = re.compile(r'[\u0980-\u09FF]')

REFUSAL_PATTERNS = [
    "দয়া করে মূল", "নির্দেশাবলী অনুসরণ", "আমি আপনার নির্দেশ",
    "মূল প্রতিবেদনটি এখানে দিন", "আমাকে মূল",
    "প্রতিবেদনটি প্রদান করুন",
    "MultiBanFakeDetect", "মাল্টিব্যানফেকডিটেক্ট",
    "academic research", "fake-news detection", "fake news detection",
]

STRIP_PREFIXES = [
    "এখানে মূল প্রতিবেদনের একটি বিভ্রান্তিকর সংস্করণ তৈরি করা হলো:",
    "সিনথেটিক/নকল খবর:", "সিন্থেটিক/নকল খবর:",
    "সিন্থেটিক/ভুয়া সংবাদ (Synthetic/Fake News):",
    "**সিন্থেটিক/ভুয়া সংবাদ (Synthetic/Fake News):**",
    "নকল খবর:", "ভুয়া খবর:", "বিভ্রান্তিকর সংস্করণ:",
    "এখানে বিভ্রান্তিকর প্রতিবেদন:", "এখানে একটি বিভ্রান্তিকর",
    "Synthetic/Fake News:", "সিন্থেটিক/নকল:", "বর্ধিত সংবাদ:",
    "নতুন মিথ্যা সংবাদ:", "মিথ্যা সংবাদ:",
]

META_KEYWORDS = [
    'বিভ্রান্তিকর সংস্করণ', 'নকল খবর', 'সিনথেটিক',
    'সিন্থেটিক', 'ভুয়া সংবাদ', 'Synthetic', 'Fake News',
    'বর্ধিত সংবাদ', 'নতুন মিথ্যা',
]


def clean_text(text: str) -> str:
    text = re.sub(r'\*\*', '', str(text)).strip()
    text = re.sub(r'^বর্ধিত সংবাদ:\s*\n*', '', text).strip()
    for p in STRIP_PREFIXES:
        if text.startswith(p):
            text = text[len(p):].strip()
    lines = text.split('\n')
    if lines and any(kw in lines[0] for kw in META_KEYWORDS):
        text = '\n'.join(lines[1:]).strip()
    return text


def is_bad(text: str, src_txt: str = "") -> str | None:
    """Returns reason string if bad, None if good."""
    text = str(text)
    if any(p in text for p in REFUSAL_PATTERNS):
        return "REFUSAL"
    if len(text) < cfg.MIN_OUTPUT_CHARS:
        return f"SHORT({len(text)})"
    if len(text) > cfg.MAX_OUTPUT_CHARS:
        return f"LONG({len(text)})"
    b = len(BANGLA_RE.findall(text))
    t = len(re.findall(r'[a-zA-Z\u0980-\u09FF]', text))
    if t > 0 and b / t < cfg.MIN_BANGLA_RATIO:
        return f"LOW_BANGLA({b/t:.2f})"
    first = text.split('\n')[0]
    fb = len(BANGLA_RE.findall(first))
    ft = len(re.findall(r'[a-zA-Z\u0980-\u09FF]', first))
    if ft > 0 and fb / ft < 0.80:
        return "BAD_FIRST_LINE"
    if text.count('।') < cfg.MIN_SENTENCES:
        return "FEW_SENTENCES"
    if src_txt:
        def ngrams(s, n=3):
            return Counter([s[i:i+n] for i in range(len(s)-n+1)])
        src_ng = ngrams(src_txt)
        out_ng = ngrams(text)
        ov     = sum((src_ng & out_ng).values())
        tot    = sum(src_ng.values())
        if tot > 0 and ov / tot > cfg.MAX_SOURCE_OVERLAP:
            return f"SIMILAR({ov/tot:.2f})"
    return None


def is_credit_error(e) -> bool:
    msg = str(e).lower()
    return any(x in msg for x in ['402', '403', 'credit', 'payment', 'insufficient', 'quota'])


def call_openrouter(model_id: str, prompt: str, max_retries: int = 4) -> str:
    api_key = os.environ.get(cfg.OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {cfg.OPENROUTER_API_KEY_ENV} before generating.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/maruf99khan/ML-Research",
        "X-Title":       "Bangla News Writing",
    }
    payload = {
        "model":       model_id,
        "messages":    [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.85,
        "max_tokens":  900,
        "top_p":       0.95,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{cfg.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers, json=payload, timeout=90,
            )
            if is_credit_error(Exception(str(resp.status_code))):
                raise RuntimeError(f"Credit error: {resp.status_code}")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if is_credit_error(e):
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{max_retries}] {e} — waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries")


def csv_path(generator_key: str, strategy: str) -> str:
    return os.path.join(cfg.LLM_FAKE_DIR, f"{generator_key}_{strategy}_samples.csv")


def current_count(generator_key: str, strategy: str) -> int:
    path = csv_path(generator_key, strategy)
    if not os.path.exists(path):
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


def total_generated() -> int:
    total = 0
    if not os.path.exists(cfg.LLM_FAKE_DIR):
        return 0
    for f in os.listdir(cfg.LLM_FAKE_DIR):
        if f.endswith('_samples.csv'):
            try:
                total += len(pd.read_csv(os.path.join(cfg.LLM_FAKE_DIR, f)))
            except Exception:
                pass
    return total


def generate_batch(
    generator_key: str,
    strategy:      str,
    n:             int,
    seed:          int = cfg.SEED,
) -> pd.DataFrame:
    """
    Generate n clean samples for (generator_key, strategy).
    Resumes automatically if CSV already exists.
    Saves after EVERY accepted sample.

    REMINDER: Click Save Version in Kaggle after this completes.
    """
    random.seed(seed)

    if generator_key not in cfg.GENERATOR_MODELS:
        raise ValueError(f"Unknown generator: {generator_key}")
    if strategy not in PROMPTS:
        raise ValueError(f"Unknown strategy: {strategy}")

    model_id = cfg.GENERATOR_MODELS[generator_key]["model_id"]

    # Load source pool (real train articles)
    manifest  = pd.read_csv(cfg.COMBINED_MANIFEST)
    real_df   = manifest[(manifest['label'] == 'real') &
                          (manifest['split'] == 'train')].copy()
    real_idx  = real_df.set_index('sample_id')
    print(f"Source pool: {len(real_df)} real train articles")

    # Resume: load existing accepted samples
    path = csv_path(generator_key, strategy)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        already  = len(existing)
        needed   = n - already
        print(f"Resuming: {already} already saved, need {needed} more")
        if needed <= 0:
            print("Already complete.")
            return existing
        accepted = existing.to_dict('records')
    else:
        accepted = []
        needed   = n

    log_path = os.path.join(cfg.LLM_FAKE_DIR, "generation_log.jsonl")
    used_ids = set(r['source_article_id'] for r in accepted)
    rejected = []
    attempts = 0

    print(f"\nGenerating {needed} samples | {generator_key} | {strategy} | prompt_v={PROMPT_VERSION}")
    print("-" * 60)

    while len(accepted) - (n - needed) < needed:
        attempts += 1
        available = real_df[~real_df['sample_id'].isin(used_ids)]
        src       = (available if len(available) > 0 else real_df).sample(1).iloc[0]
        used_ids.add(src['sample_id'])

        if strategy == "extend":
            headline = src['text'].split('।')[0][:120].strip()
            prompt   = PROMPTS["extend"].format(headline=headline)
        else:
            prompt = PROMPTS[strategy].format(source_text=src['text'][:800])

        current = len(accepted) - (n - needed)
        print(f"  [{current+1}/{needed}] attempt={attempts} src={src['sample_id']}", end=" ")

        try:
            raw = call_openrouter(model_id, prompt)
        except Exception as e:
            if is_credit_error(e):
                print(f"\n❌ CREDITS EXHAUSTED: {e}")
                print("Add credits to OpenRouter and re-run — progress is saved.")
                break
            print(f"-> API ERROR: {e}")
            continue

        cleaned = clean_text(raw)
        src_txt = str(real_idx.loc[src['sample_id'], 'text']) if src['sample_id'] in real_idx.index else ""
        reason  = is_bad(cleaned, src_txt)

        if reason:
            print(f"-> REJECTED ({reason})")
            rejected.append({"reason": reason, "source": src['sample_id']})
            continue

        print(f"-> OK ({len(cleaned)} chars)")

        sample_id = f"llmfake_{generator_key}_{strategy}_{uuid.uuid4().hex[:10]}"

        # Log for reproducibility
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps({
                "sample_id":          sample_id,
                "source_article_id":  str(src['sample_id']),
                "generator_key":      generator_key,
                "generator_model_id": model_id,
                "strategy":           strategy,
                "prompt_version":     PROMPT_VERSION,
                "timestamp_utc":      datetime.now(timezone.utc).isoformat(),
                "input_chars":        len(prompt),
                "output_chars_raw":   len(raw),
                "output_chars_clean": len(cleaned),
                "meta_stripped":      raw != cleaned,
                "attempt_number":     attempts,
            }, ensure_ascii=False) + "\n")

        accepted.append({
            "sample_id":         sample_id,
            "text":              cleaned,
            "image_path":        str(src['image_path']),
            "label":             "llm_fake",
            "label_id":          2,
            "generator":         generator_key,
            "strategy":          strategy,
            "source_article_id": str(src['sample_id']),
            "split":             None,
        })

        # Save after EVERY accepted sample
        pd.DataFrame(accepted).to_csv(path, index=False, encoding="utf-8")
        time.sleep(1.0)

    result = pd.DataFrame(accepted)
    result.to_csv(path, index=False, encoding="utf-8")

    n_this_run = len(accepted) - (n - needed)
    acc_rate   = n_this_run / attempts * 100 if attempts > 0 else 0
    reasons    = {}
    for r in rejected:
        k = r['reason'].split('(')[0]
        reasons[k] = reasons.get(k, 0) + 1

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Accepted this run : {n_this_run}")
    print(f"  Total in file     : {len(result)}")
    print(f"  Attempts          : {attempts}")
    print(f"  Acceptance rate   : {acc_rate:.1f}%")
    print(f"  Rejections        : {reasons}")
    print(f"  Saved → {path}")
    print(f"\n⚠️  CLICK SAVE VERSION IN KAGGLE NOW")

    return result
