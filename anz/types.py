from typing import Union
from dataclasses import dataclass
from enum import Enum

class InferenceType(Enum):
    UNKNOWN = 0
    POLICY_ONLY = 1
    VALUE_ONLY = 2
    MCTS = 3

@dataclass
class InferenceResult:
    fen: str
    move: Union[str, None]
    value: Union[float, None]
    inference_type: InferenceType
    mcts_rollouts: Union[int, None]

@dataclass
class PuzzleEvaluationResult:
    accuracy: float
    inference_type: InferenceType
    mcts_rollouts: Union[int, None]

