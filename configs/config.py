"""
Central configuration for MultiBanFakeDetect.
KAGGLE SETUP: set MBFD_BASE_DIR and MBFD_DATASET_DIR as environment variables.
"""
import os

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR    = os.environ.get("MBFD_BASE_DIR", "/kaggle/working/ML-Research")
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

# ============================================================================
# LABELS
# ============================================================================
LABEL2ID    = {"real": 0, "human_fake": 1, "llm_fake": 2}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
TEXT_MODEL_NAME  = "csebuetnlp/banglabert_large"  # Electra-based, 1024-d hidden
IMAGE_MODEL_NAME = "vit_base_patch16_224"          # ViT-B/16, 768-d

# Kept for backward compatibility with model.py references
FALLBACK_TEXT_MODEL_NAME  = "csebuetnlp/banglabert"  # base, 768-d
FALLBACK_IMAGE_MODEL_NAME = "vit_base_patch16_224"
USE_FALLBACK              = False

TEXT_HIDDEN_DIM  = 1024
IMAGE_HIDDEN_DIM = 768
PROJECTION_DIM   = 768   # W_t: projects text 1024-d -> 768-d to match ViT
FUSION_HIDDEN_DIM = 768  # CMAF attention dimension
FUSION_HEADS     = 8
DROPOUT          = 0.2
MAX_TEXT_LEN     = 256

# Memory-safe settings confirmed on Kaggle T4 (14.56GB VRAM)
FREEZE_TEXT_LAYERS = 12   # freeze first 12 of 24 encoder layers
FREEZE_IMAGE       = True  # freeze ViT, only train fusion+classifier
# Result: 159M trainable / 429M total, 9.39GB free after model load

# ============================================================================
# TRAINING (confirmed working on Kaggle T4)
# ============================================================================
SEED                    = 42
BATCH_SIZE              = 4    # OOM at 16, safe at 4
GRAD_ACCUM_STEPS        = 8    # effective batch = 32
EVAL_BATCH_SIZE         = 8
NUM_EPOCHS              = 10
LR                      = 2e-5
WEIGHT_DECAY            = 0.01
WARMUP_RATIO            = 0.06
MAX_GRAD_NORM           = 1.0
EARLY_STOPPING_PATIENCE = 3
NUM_WORKERS             = 2

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# ============================================================================
# LLM-FAKE GENERATION (OpenRouter) — model IDs verified July 30, 2026
# ============================================================================
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL    = "https://openrouter.ai/api/v1"

GENERATOR_MODELS = {
    "gemini-2.5-flash": {
        "model_id": "google/gemini-2.5-flash",
        "target_count": 1200,
    },
    "gpt-4o-mini": {
        "model_id": "openai/gpt-4o-mini",
        "target_count": 1000,
    },
    "llama-3.3-70b": {
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "target_count": 800,
    },
    "claude-haiku": {
        "model_id": "anthropic/claude-3-haiku",
        "target_count": 800,
    },
}
GENERATION_STRATEGIES = ["rewrite", "extend", "summarize_extend"]
TOTAL_LLM_FAKE_TARGET = sum(v["target_count"] for v in GENERATOR_MODELS.values())

# Quality filters
MIN_OUTPUT_CHARS   = 100
MIN_BANGLA_RATIO   = 0.5
MAX_SOURCE_OVERLAP = 0.80

# ============================================================================
# QC
# ============================================================================
QC_SAMPLE_SIZE    = 200
QC_NUM_ANNOTATORS = 2

# ============================================================================
# EXPLAINABILITY
# ============================================================================
IG_N_STEPS                            = 50
IG_NUM_QUALITATIVE_EXAMPLES_PER_CLASS = 10
IG_HUMAN_CHECK_SAMPLE_SIZE            = 40
