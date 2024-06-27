import math
import torch
import chess
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm

from .helpers import flip_fen, flip_fen_if_black_turn, flip_inference_result, model_forward_pass
from .types import InferenceResult, InferenceType
from .constants import (
    DEFAULT_MCTS_ROLLOUTS,
    CPUCT,
    TEMPERATURE, 
    POLICY_INDEX,
    POLICY_SIZE
)


@dataclass
class NodeCache:
    visited = False
    legal_moves = []


class MCTS():
    def __init__(self, model: torch.nn.Module, model_type: str):
        self.model = model
        self.model_type = model_type

        self.N = defaultdict(lambda: 0) # Number of times a state-action pair has been visited
        self.W = defaultdict(float) # Total value of a state-action pair
        self.Q = defaultdict(float) # Average value of a state-action pair
        self.P = defaultdict(lambda: torch.zeros(POLICY_SIZE, dtype=torch.float32)) # Prior probability of taking an action in a state
        self.M = defaultdict(lambda: NodeCache()) # Cached legal move list for states

    def go(self, 
           fen: str, 
           rollouts: int = DEFAULT_MCTS_ROLLOUTS, 
           verbose: bool = False
    ) -> InferenceResult:
        self.model.eval()
        canonical_fen = flip_fen_if_black_turn(fen)
        inference_result = InferenceResult(
            canonical_fen, None, None, None, inference_type=InferenceType.MCTS, mcts_rollouts=rollouts)

        it = range(rollouts)
        if verbose:
            print(f"Running MCTS with {rollouts} rollouts on FEN '{fen}'")
            it = tqdm(range(rollouts), desc="MCTS simulations", bar_format="{l_bar}{bar}| update: {n_fmt}/{total_fmt} - elapsed: {elapsed}")

        value = 0
        for _ in it:
            value = self.leaf_rollout(canonical_fen)
        inference_result.value = -value

        raw_pi = [self.N[(canonical_fen, move)] for move in POLICY_INDEX]

        # if temperature = 0, we choose the action deterministically for competitive play
        if TEMPERATURE > 0:
            pi = [N ** (1.0 / TEMPERATURE) for N in raw_pi]
            if sum(pi) == 0:
                pi = [1.0 for _ in range(POLICY_SIZE)]
            pi = [N / sum(pi) for N in pi]
            move_idx = torch.multinomial(torch.tensor(pi), 1)
        else:
            move_idx = raw_pi.index(max(raw_pi))

        pi = [N / sum(raw_pi) for N in raw_pi]

        # get top5 moves
        top5 = sorted(range(len(pi)), key=lambda i: pi[i], reverse=True)[:5]
        top5_moves = [POLICY_INDEX[i] for i in top5]
        top5_values = [pi[i] for i in top5]
        top5 = list(zip(top5_moves, top5_values))
        inference_result.top5 = top5

        # get the best move
        move = POLICY_INDEX[move_idx]
        inference_result.move = move

        if canonical_fen != fen:
            inference_result = flip_inference_result(inference_result)
        return inference_result

    def leaf_rollout(self, fen: str) -> float:
        board = chess.Board(fen)

        # if the search has reached a terminal state, it returns the value according to
        # the game's rules
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            winner = outcome.winner
            if winner == chess.WHITE:
                return -1.0
            elif winner == chess.BLACK:
                return 1.0
            else:
                return 0

        if not self.M[fen].visited:
            self.M[fen].visited = True
            self.M[fen].legal_moves = [str(move) for move in board.legal_moves]

        legal_move_vec = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        for move in self.M[fen].legal_moves:
            move_idx = POLICY_INDEX.index(str(move))
            legal_move_vec[move_idx] = 1

        # if a leaf node is reached, we evaluate using the network
        if fen not in self.P:
            pi, v = model_forward_pass(self.model, self.model_type, fen)
            pi = torch.softmax(pi, dim=1).to("cpu").reshape(POLICY_SIZE)
            v = v.to("cpu").item()

            assert pi.shape == legal_move_vec.shape

            # mask invalid actions from the action probabilities and renormalize
            pi = pi * legal_move_vec
            pi = pi / max(torch.sum(pi).item(), 1e-8)

            self.P[fen] = pi
            return -v

        # we select the move used to continue the search by choosing the move that 
        # maximizes Q(s, a) + U(s, a) in accordance with DeepMind's AlphaGo Zero paper:
        #
        #   a(t) = argmax(Q(s, a) + U(s, a)), where
        #   U(s, a) = cpuct * P(s, a) * (sprt(N(s)) / (1 + N(s, a)))
        #
        # 'cpuct' is a constant determining the level of exploration - a small 
        # cpuct value will give more weight to the move Q-value as opposed to the network's 
        # output probabilities and the amount of times the move has been explored

        qu, selected_move = -torch.inf, None
        for move in self.M[fen].legal_moves:
            N_sum = 0
            for m in POLICY_INDEX:
                N_sum += self.N[(fen, m)]

            q = self.Q[(fen, move)]
            u = CPUCT
            u *= self.P[fen][POLICY_INDEX.index(move)]
            u *= math.sqrt(N_sum)
            u /= (1.0 + self.N[(fen, move)])
            qu_m = q + u

            if qu_m > qu:
                qu = qu_m
                selected_move = move
        assert selected_move is not None

        # simulate the action and get the next positional node
        board.push(chess.Move.from_uci(selected_move))
        next_fen = flip_fen(board.fen())

        # continue search
        v = self.leaf_rollout(next_fen)

        self.N[(fen, selected_move)] += 1
        self.W[(fen, selected_move)] += v
        self.Q[(fen, selected_move)] = self.W[(fen, selected_move)] / self.N[(fen, selected_move)]

        return -v

