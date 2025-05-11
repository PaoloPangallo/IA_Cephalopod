import copy
import pickle
import random

from cephalopod.RL.reward_shaping.rewardshaping import BasicShaper
from cephalopod.core.board import Die
from cephalopod.core.mechanics import (
    find_capturing_subsets,
    choose_capturing_subset,
)


class RLPlayer:
    def __init__(self, name, exp_rate=0.3, lr=0.2, gamma=0.9, debug=False, policy_path=None, reward_shaper=None):
        self.name = name
        self.exp_rate = exp_rate
        self.lr = lr
        self.gamma = gamma
        self.debug = debug
        self.policy_path = policy_path or f"policy_{self.name}.pkl"
        self.reward_shaper = reward_shaper or BasicShaper()

        self.states = []  # Lista di (hash, reward intermedio)
        self.states_value = {}  # Q-table

    def get_hash(self, board):
        return str([[str(cell) if cell else "" for cell in row] for row in board.grid])

    def get_intermediate_reward(self, board, move, my_color):
        opponent_color = "B" if my_color == "W" else "W"

        try:
            # Se lo shaper accetta 5 argomenti (shaper avanzati)
            return self.reward_shaper.compute(board, move, my_color, opponent_color, None)
        except TypeError:
            # Per shaper semplici (basic, riskaware)
            return self.reward_shaper.compute(board, move, my_color)

    def choose_action(self, board, color):
        empty_cells = board.get_empty_cells()
        actions = []

        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                actions.append((r, c, sum_pips, subset))
            else:
                actions.append((r, c, 1, []))

        if random.uniform(0, 1) <= self.exp_rate:
            move = random.choice(actions)
        else:
            best_value = -float("inf")
            move = None
            for a in actions:
                board_copy = copy.deepcopy(board)
                for (rr, cc) in a[3]:
                    board_copy.grid[rr][cc] = None
                board_copy.place_die(a[0], a[1], Die(color, a[2]))
                state_hash = self.get_hash(board_copy)
                value = self.states_value.get(state_hash, 0)
                if value > best_value:
                    best_value = value
                    move = a

        # Salva lo stato simulato + reward intermedio
        board_sim = copy.deepcopy(board)
        for (rr, cc) in move[3]:
            board_sim.grid[rr][cc] = None
        board_sim.place_die(move[0], move[1], Die(color, move[2]))

        state_hash = self.get_hash(board_sim)
        shaped_rwd = self.get_intermediate_reward(board, move, color)
        self.states.append((state_hash, shaped_rwd))

        return move

    def feed_reward(self, final_reward):
        for state_hash, shaped_reward in reversed(self.states):
            if state_hash not in self.states_value:
                self.states_value[state_hash] = 0
            updated_value = self.states_value[state_hash] + self.lr * (
                    self.gamma * final_reward + shaped_reward - self.states_value[state_hash]
            )
            self.states_value[state_hash] = updated_value
            final_reward = updated_value

    def reset(self):
        self.states = []

    def save_policy(self, path=None):
        path = path or self.policy_path
        with open(path, "wb") as f:
            pickle.dump({
                "states_value": self.states_value,
                "exp_rate": self.exp_rate,
                "lr": self.lr,
                "gamma": self.gamma
            }, f)

    def load_policy(self, path=None):
        path = path or self.policy_path
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.states_value = data.get("states_value", {})
            self.exp_rate = data.get("exp_rate", self.exp_rate)
            self.lr = data.get("lr", self.lr)
            self.gamma = data.get("gamma", self.gamma)

    choose_move = choose_action
