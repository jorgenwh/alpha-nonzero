from .policy_index import policy_index

# --- State and action details ---
POLICY_INDEX         = policy_index
POLICY_SIZE          = len(POLICY_INDEX)
VOCAB                = sorted(list(set(c for c in "PpRrNnBbQqKkabcdefgh12345678wb.09")))
VOCAB_SIZE           = len(VOCAB)
BLOCK_SIZE           = 76

# --- Annotation settings ---
THREADS              = 6
HASH                 = 8192
NODES_PER_ANNOTATION = 1000000

# --- Transformer settings ---
D_MODEL              = 256
N_HEADS              = 8
N_LAYERS             = 8

# --- ConvNet settings ---
pass

# --- Training settings ---
LEARNING_RATE        = 0.001
BATCH_SIZE           = 256
TRAINING_ITERS       = 1000000
DEVICE               = "cuda:0"
OUTPUT_DIR           = "tmp"
MAX_DATA_POINTS      = None

# --- Misc ---
CHAR_TO_IDX          = {c:i for i, c in enumerate(VOCAB)}

