import torch
from .policy_index import policy_index

# --- State and action details ---
POLICY_INDEX          = policy_index
POLICY_SIZE           = len(POLICY_INDEX)
VOCAB                 = sorted(list(set(c for c in "PpRrNnBbQqKkabcdefgh12345678wb.09")))

# 6 + 6 + 4 + 1 + 1 + 1 = 19 : (white pieces, black pieces, castling, en passant, halfmove, fullmove)
BOARD_CONV_CHANNELS   = 19 

# --- Annotation settings ---
THREADS               = 6
HASH                  = 8192
NODES_PER_ANNOTATION  = 1000000

# --- MCTS settings ---
CPUCT                 = 1.0
TEMPERATURE           = 0
DEFAULT_MCTS_ROLLOUTS = 40

# --- Transformer settings ---
VOCAB_SIZE            = len(VOCAB)
BLOCK_SIZE            = 76
D_MODEL               = 512
D_OUTPUT              = 1028
N_HEADS               = 8
N_LAYERS              = 8

# --- ResNet settings ---
N_BLOCKS              = 6

# --- Training settings ---
EPOCHS                = 100
BATCH_SIZE            = 128
LEARNING_RATE         = 0.001
USE_CUDA              = True

# --- Misc ---
CHAR_TO_IDX           = {c:i for i, c in enumerate(VOCAB)}
DEVICE                = "cuda" if torch.cuda.is_available() and USE_CUDA else "cpu"
