import numpy as np
import math
import copy
from cephalopod.core.board import Die
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset, get_opponent

class MCTSNode:
    def __init__(self, game, player, parent=None, prior=0.0):
        self.game = game
        self.player = player
        self.parent = parent
        self.P = prior
        self.N = 0
        self.W = 0
        self.Q = 0
        self.children = {}

    def is_expanded(self):
        return bool(self.children)

    def expand(self, model, current_player):
        legal_moves = self.game.get_empty_cells()
        board_tensor = model.encode_board(self.game, current_player)
        policy, _ = model.model.predict(board_tensor, legal_moves)

        for r, c in legal_moves:
            new_game = self.game.clone()
            captured, sum_pips = choose_capturing_subset(find_capturing_subsets(new_game, r, c))
            top_face = 6 - sum_pips if captured else 1
            for rr, cc in (captured or []):
                new_game.grid[rr][cc] = None
            new_game.place_die(r, c, Die(current_player, top_face))

            move_index = r * 5 + c
            self.children[move_index] = MCTSNode(
                game=new_game,
                player=get_opponent(current_player),
                parent=self,
                prior=policy[move_index]
            )

    def select_child(self, c_puct=1.0):
        total_visits = sum(child.N for child in self.children.values())
        return max(self.children.items(), key=lambda item: item[1].Q + c_puct * item[1].P * math.sqrt(total_visits) / (1 + item[1].N))

    def backpropagate(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
        if self.parent:
            self.parent.backpropagate(-value)

class MCTS:
    def __init__(self, model, num_simulations=50, c_puct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def run(self, root_game, root_player):
        root = MCTSNode(game=root_game.clone(), player=root_player)
        root.expand(self.model, root_player)

        for _ in range(self.num_simulations):
            node = root
            while node.is_expanded():
                _, node = node.select_child(self.c_puct)
            node.expand(self.model, node.player)
            board_tensor = self.model.encode_board(node.game, node.player)
            _, value = self.model.model.predict(board_tensor, node.game.get_empty_cells())
            node.backpropagate(value)

        visits = np.zeros(25)
        for move, child in root.children.items():
            visits[move] = child.N
        return visits / visits.sum() if visits.sum() > 0 else visits
