from typing import Union
from dataclasses import dataclass

@dataclass
class InferenceResult:
    move: Union[str, None]
    value: Union[float, None]

@dataclass
class PuzzleEvaluationResult:
    accuracy: float
