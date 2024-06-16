from .policy_index import policy_index

# --- State and action details ---
POLICY_INDEX         = policy_index
POLICY_SIZE          = len(POLICY_INDEX)
VOCAB                = sorted(list(set(c for c in "PpRrNnBbQqKkabcdefgh12345678wb.09")))

# 6 + 6 + 4 + 1 + 1 + 1 = 19 : (white pieces, black pieces, castling, en passant, halfmove, fullmove)
BOARD_CONV_CHANNELS  = 19 

# --- Annotation settings ---
THREADS              = 6
HASH                 = 8192
NODES_PER_ANNOTATION = 1000000

# --- Transformer settings ---
VOCAB_SIZE           = len(VOCAB)
BLOCK_SIZE           = 76
D_MODEL              = 256
N_HEADS              = 8
N_LAYERS             = 8

# --- ResNet settings ---
N_BLOCKS             = 6

# --- Training settings ---
EPOCHS               = 10
BATCH_SIZE           = 64
LEARNING_RATE        = 0.001
DEVICE               = "cuda:0"

# --- Misc ---
CHAR_TO_IDX          = {c:i for i, c in enumerate(VOCAB)}

