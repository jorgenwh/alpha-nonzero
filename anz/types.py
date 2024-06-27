from typing import Union
from dataclasses import dataclass
from enum import Enum

class InferenceType(Enum):
    UNKNOWN = 0
    RAW = 1
    POLICY_ONLY = 2
    VALUE_ONLY = 3
    MCTS = 4

@dataclass
class InferenceResult:
    fen: str
    move: Union[str, None]
    value: Union[float, None]
    top5: Union[list, None]
    inference_type: InferenceType
    mcts_rollouts: Union[int, None]

@dataclass
class PuzzleEvaluationResult:
    accuracy: float
    inference_type: InferenceType
    mcts_rollouts: Union[int, None]

