import os

BASE = "/mnt/scratch/djagbapr/dual-lora-mp-project"

MODEL_PATH = "/mnt/home/djagbapr/Saley/ABIA_EUNICE/meta-prompting/Math/outputs/models/Llama-3.3-70B-Instruct"

MATH_DATA_DIR = "/mnt/home/djagbapr/Saley/ABIA_EUNICE/meta-prompting/Math/data/math"
GSM8K_DATA_DIR = "/mnt/home/djagbapr/Saley/ABIA_EUNICE/meta-prompting/Math/data/gsm8k"

MATH_TRACES_DIR = "/mnt/home/djagbapr/Saley/ABIA_EUNICE/meta-prompting/Math/outputs/models/Llama-3.3-70B-Instruct/math"

MATH_TRACES_FILE = os.path.join(MATH_TRACES_DIR, "train_mp_-1_seed0_t0.0_s0_e7500_05-04_16-14_Unknown.jsonl")

TRACES_DIR = os.path.join(BASE, "traces")
GSM8K_TRACES_DIR = os.path.join(TRACES_DIR, "gsm8k")
GSM8K_TRACES_FILE = os.path.join(GSM8K_TRACES_DIR, "gsm8k_train_traces.jsonl")

MATH_PREP_DIR = os.path.join(BASE, "data/math")
GSM8K_PREP_DIR = os.path.join(BASE, "data/gsm8k")

MATH_OUTPUT = os.path.join(BASE, "outputs/math")
GSM8K_OUTPUT = os.path.join(BASE, "outputs/gsm8k")
MATH_4BIT_OUTPUT = os.path.join(BASE, "outputs/math_4bit")

MATH_EVAL_OUT = os.path.join(BASE, "outputs/eval_math.jsonl")
GSM8K_EVAL_OUT = os.path.join(BASE, "outputs/eval_gsm8k.jsonl")

LORA_R1 = 16
LORA_R2 = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
WARMUP_STEPS = 20

MAX_SEQ_LEN = 4096
GSM8K_MAX_SEQ_LEN = 2048
PER_DEVICE_BATCH_SIZE = 1
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
MAX_GRAD_NORM = 1.0
SEED = 42

VLLM_TENSOR_PARALLEL = 4
VLLM_MAX_TOKENS = 1024
VLLM_TEMPERATURE = 0.0
