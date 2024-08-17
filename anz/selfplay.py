import torch
from typing import List, Union
from dataclasses import dataclass

from .constants import BLOCK_SIZE, POLICY_SIZE
from .models import Transformer, ResNet
from .mcts import MCTS
from .helpers import allocate_zero_tensor, load_model, save_model

@dataclass
class SPConfig:
    replay_memory_size: int = 50000
    iterations: int = 2
    num_games: int = 100
    eval_games: int = 40
    win_threshold: float = 0.55
    cuda: bool = True
    fp16: bool = False
    compile_model: bool = False
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 10
    monte_carlo_rollouts: int = 10
    model_type: str = "transformer"
    temperature: float = 0.5
    model: Union[str, None] = None
    output_dir: Union[str, None] = "tmp"

@dataclass
class HistoryElement:
    position: torch.Tensor
    pi: torch.Tensor
    v: float


class ReplayMemory():
    def __init__(self, config: SPConfig):
        self.config = config

        self.positions = allocate_zero_tensor((self.config.replay_memory_size, BLOCK_SIZE), torch.int64)
        self.vs = allocate_zero_tensor((self.config.replay_memory_size, 1), torch.float32)
        self.pis = allocate_zero_tensor((self.config.replay_memory_size, POLICY_SIZE), torch.float32)
        self.cntr = 0
        self.size = 0

    def add(self, history: List[HistoryElement]):
        for i in range(len(history)):
            self.positions[self.cntr] = history[i].position
            self.vs[self.cntr] = history[i].v
            self.pis[self.cntr] = history[i].pi
            self.cntr = (self.cntr + 1) % self.config.replay_memory_size
            self.size = min(self.size + 1, self.config.replay_memory_size)

    def sample(self, batch_size: int):
        raise NotImplementedError("ReplayMemory.sample is not implemented yet")


def selfplay_game(config: SPConfig, model: Union[Transformer, ResNet], mcts: MCTS) -> List[HistoryElement]:
    return []


class SelfPlayManager():
    model_class = Transformer

    def __init__(self, config: SPConfig):
        self.config = config

        if self.config.model_type == "transformer":
            self.model_class = Transformer
        if self.config.model_type == "resnet":
            raise NotImplementedError("Selfplay with ResNet is not implemented yet")

        self.checkpoint = 0
        self.games_played = 0

        self.replay_memory = ReplayMemory(config)

        self.champion = self.model_class()
        self.create_new_checkpoint(self.champion)

    def create_new_checkpoint(self, model):
        save_model({
            "model_state_dict": model.state_dict(),
            "games_played": self.games_played
        }, f"{self.config.output_dir}/checkpoint_{self.checkpoint}.pt")
        self.checkpoint += 1

    def selfplay(self):
        for _ in range(self.config.num_games):
            history = selfplay_game(self.config, self.champion, MCTS(self.champion, self.config.model_type))
            self.replay_memory.add(history)
            self.games_played += 1

    def train(self):
        pass

    def evaluate(self):
        pass

    def run_iteration(self):
        self.selfplay()
        self.train()
        self.evaluate()

    def start(self):
        for i in range(self.config.iterations):
            print("Selfplay iteration: " + str(i))
            self.run_iteration()
