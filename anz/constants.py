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
D_MODEL               = 256
D_OUTPUT              = 1028
N_HEADS               = 8
N_LAYERS              = 8

# --- ResNet settings ---
N_BLOCKS              = 6

# --- Training settings ---
EPOCHS                = 1
BATCH_SIZE            = 256
LEARNING_RATE         = 0.001
USE_CUDA              = True

# --- Misc ---
CHAR_TO_IDX           = {c:i for i, c in enumerate(VOCAB)}
DEVICE                = "cuda" if torch.cuda.is_available() and USE_CUDA else "cpu"

EMPTY_TOKEN           = 0
ZERO_TOKEN            = 1
ONE_TOKEN             = 2
TWO_TOKEN             = 3
THREE_TOKEN           = 4
FOUR_TOKEN            = 5
FIVE_TOKEN            = 6
SIX_TOKEN             = 7
SEVEN_TOKEN           = 8
EIGHT_TOKEN           = 9
NINE_TOKEN            = 10
BU_TOKEN              = 11
KU_TOKEN              = 12
NU_TOKEN              = 13
PU_TOKEN              = 14
QU_TOKEN              = 15
RU_TOKEN              = 16
AL_TOKEN              = 17
BL_TOKEN              = 18
CL_TOKEN              = 19
DL_TOKEN              = 20
EL_TOKEN              = 21
FL_TOKEN              = 22
GL_TOKEN              = 23
HL_TOKEN              = 24
KL_TOKEN              = 25
NL_TOKEN              = 26
PL_TOKEN              = 27
QL_TOKEN              = 28
RL_TOKEN              = 29
WL_TOKEN              = 30

