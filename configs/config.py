"""
MultiBanFakeDetect — Central Configuration
FINAL VERSION — locked July 31 2026, no more changes

KAGGLE SETUP:
    os.environ["MBFD_BASE_DIR"]    = "/kaggle/working/ML-Research"
    os.environ["MBFD_DATASET_DIR"] = "/kaggle/input/datasets/mukaffimoin/multibanfakedetect-multimodal-bangla-fake-news"
"""
import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR    = os.environ.get("MBFD_BASE_DIR",    "/kaggle/working/ML-Research")
DATASET_DIR = os.environ.get("MBFD_DATASET_DIR",
              "/kaggle/input/datasets/mukaffimoin/multibanfakedetect-multimodal-bangla-fake-news")

COMBINED_MANIFEST = os.path.join(BASE_DIR, "data", "combined_manifest.csv")
LLM_FAKE_DIR      = os.path.join(BASE_DIR, "data", "generated", "llm_fake")
CHECKPOINT_DIR    = os.path.join(BASE_DIR, "outputs", "checkpoints")
LOG_DIR           = os.path.join(BASE_DIR, "outputs", "logs")
EXPLAIN_DIR       = os.path.join(BASE_DIR, "outputs", "explanations")
METRICS_DIR       = os.path.join(BASE_DIR, "outputs", "metrics")
QC_DIR            = os.path.join(BASE_DIR, "outputs", "qc")

for _d in [LLM_FAKE_DIR, CHECKPOINT_DIR, LOG_DIR, EXPLAIN_DIR, METRICS_DIR, QC_DIR]:
    os.makedirs(_d, exist_ok=True)

# ============================================================
# LABELS
# ============================================================
LABEL2ID    = {"real": 0, "human_fake": 1, "llm_fake": 2}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)

# ============================================================
# MODEL — LOCKED
# ============================================================
TEXT_MODEL_NAME  = "csebuetnlp/banglabert_large"   # Electra, 1024-d, 24 layers
IMAGE_MODEL_NAME = "vit_base_patch16_224"           # ViT-B/16, 768-d, 197 tokens

# Fallback (kept for backward compat — not used in final training)
FALLBACK_TEXT_MODEL_NAME  = "csebuetnlp/banglabert"
FALLBACK_IMAGE_MODEL_NAME = "vit_base_patch16_224"
USE_FALLBACK              = False

TEXT_HIDDEN_DIM   = 1024
IMAGE_HIDDEN_DIM  = 768
PROJECTION_DIM    = 768    # W_t: 1024 -> 768
FUSION_HIDDEN_DIM = 768    # CMAF attention dim
FUSION_HEADS      = 8
DROPOUT           = 0.2
MAX_TEXT_LEN      = 256

# ============================================================
# TRAINING — LOCKED
# ============================================================
SEED                    = 42
BATCH_SIZE              = 8      # T4x2
EVAL_BATCH_SIZE         = 16
GRAD_ACCUM_STEPS        = 4      # effective batch = 32
NUM_EPOCHS              = 15     # early stopping handles it
LR                      = 2e-5
WEIGHT_DECAY            = 0.01
WARMUP_RATIO            = 0.06
MAX_GRAD_NORM           = 1.0
EARLY_STOPPING_PATIENCE = 3
NUM_WORKERS             = 2

# Memory settings — confirmed on Kaggle T4x2
FREEZE_TEXT_LAYERS = 6      # freeze first 6 of 24 BERT layers
FREEZE_IMAGE       = False  # fine-tune ViT

# Fallback if OOM
FALLBACK_FREEZE_TEXT_LAYERS = 12
FALLBACK_FREEZE_IMAGE       = True
FALLBACK_BATCH_SIZE         = 4

# ============================================================
# GENERATION — COMPLETE (2 models x 3 strategies x 800 = 4,800 samples)
# Generation is done. These values are kept for paper reproducibility.
# ============================================================
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL    = "https://openrouter.ai/api/v1"

# Model IDs verified July 30 2026 via OpenRouter /api/v1/models
GENERATOR_MODELS = {
    "gpt-4o-mini": {
        "model_id":     "openai/gpt-4o-mini",
        "target_count": 2400,   # 800 x 3 strategies
    },
    "claude-haiku": {
        "model_id":     "anthropic/claude-3-haiku",
        "target_count": 2400,   # 800 x 3 strategies
    },
}

# Per-strategy targets (actual final counts)
STRATEGY_TARGETS = {
    "gpt-4o-mini": {
        "rewrite":          800,
        "extend":           800,
        "summarize_extend": 800,
    },
    "claude-haiku": {
        "rewrite":          800,
        "extend":           800,
        "summarize_extend": 800,
    },
}

GENERATION_STRATEGIES  = ["rewrite", "extend", "summarize_extend"]
TOTAL_LLM_FAKE_TARGET  = 4800   # 4,800 total — perfectly balanced with Real and Human-Fake
GENERATION_BATCH_SIZE  = 150    # samples per API batch
GENERATION_MAX_FAIL    = 3      # consecutive failures before skipping combo

# Quality filters — locked after batch 1 inspection
MIN_OUTPUT_CHARS   = 200
MIN_BANGLA_RATIO   = 0.70
MAX_SOURCE_OVERLAP = 0.65   # 3-gram overlap threshold
MAX_OUTPUT_CHARS   = 2500
MIN_SENTENCES      = 2      # minimum Bangla sentence terminators (।)

# ============================================================
# QC
# ============================================================
QC_SAMPLE_SIZE    = 200
QC_NUM_ANNOTATORS = 2

# ============================================================
# EXPLAINABILITY
# ============================================================
IG_N_STEPS                            = 50
IG_NUM_QUALITATIVE_EXAMPLES_PER_CLASS = 10
IG_HUMAN_CHECK_SAMPLE_SIZE            = 40

# ============================================================
# SPLITS
# ============================================================
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
