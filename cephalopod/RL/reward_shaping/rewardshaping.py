import random
from abc import ABC, abstractmethod
from cephalopod.core.mechanics import get_opponent, find_capturing_subsets, choose_capturing_subset
from cephalopod.core.board import Die
import copy

from cephalopod.strategies import AggressiveStrategy
from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5
from cephalopod.strategies.smart_position import SmartPositionalLookaheadStrategy
from strategies import NaiveStrategy


class RewardShaper(ABC):
    @abstractmethod
    def compute(self, board, move, my_color) -> float:
        pass


class BasicShaper(RewardShaper):
    def compute(self, board, move, my_color):
        r, c, top_face, captured = move
        reward = 0
        if top_face == 6:
            reward += 60
        reward += 0.2 * len(captured)
        if top_face == 5:
            reward -= 0.1
        return reward


class RiskAwareShaper(RewardShaper):
    def compute(self, board, move, my_color):
        r, c, top_face, captured = move
        reward = BasicShaper().compute(board, move, my_color)

        board_copy = copy.deepcopy(board)
        for (rr, cc) in captured:
            board_copy.grid[rr][cc] = None
        board_copy.place_die(r, c, Die(my_color, top_face))

        opponent = get_opponent(my_color)
        for (rr, cc) in board_copy.get_empty_cells():
            options = find_capturing_subsets(board_copy, rr, cc)
            if options:
                _, sum_pips = choose_capturing_subset(options)
                if sum_pips == 6:
                    reward -= 60
                    break
        return reward


# ... [import e RewardShaper, BasicShaper, RiskAwareShaper invariati] ...

class AggressiveBoardShaper:
    def compute(self, board, move, my_color, opponent_color, winner):
        reward = 0

        from copy import deepcopy
        from cephalopod.core.board import Die
        from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset

        board_copy = deepcopy(board)
        r, c, top_face, captured = move

        for (rr, cc) in captured:
            board_copy.grid[rr][cc] = None
        board_copy.place_die(r, c, Die(my_color, top_face))

        # ✅ Usa top_face
        for row in board_copy.grid:
            for cell in row:
                if cell:
                    if cell.color == my_color and cell.top_face == 6:
                        reward += 270
                    if cell.color == opponent_color and cell.top_face == 6:
                        reward -= 270

        for (rr, cc) in board_copy.get_empty_cells():
            options = find_capturing_subsets(board_copy, rr, cc)
            if options:
                _, sum_pips = choose_capturing_subset(options)
                if sum_pips == 6:
                    reward -= 60
                    break

        if winner:
            if winner == my_color:
                reward += 3000

            my_ones = sum(
                1 for row in board.grid for cell in row if cell and cell.color == my_color and cell.top_face == 1
            )
            reward += 3 * my_ones

            my_dice = sum(
                1 for row in board.grid for cell in row if cell and cell.color == my_color
            )
            opp_dice = sum(
                1 for row in board.grid for cell in row if cell and cell.color == opponent_color
            )
            reward += (my_dice - opp_dice)

        return reward


class AggressiveBoardShaper2:
    def compute(self, board, move, my_color, opponent_color, winner):
        reward = 0

        from copy import deepcopy
        from cephalopod.core.board import Die
        from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset

        r, c, top_face, captured = move
        board_copy = deepcopy(board)

        for (rr, cc) in captured:
            if board.grid[rr][cc] and board.grid[rr][cc].top_face == 6:
                reward += 10
            board_copy.grid[rr][cc] = None

        board_copy.place_die(r, c, Die(my_color, top_face))

        for row in board_copy.grid:
            for cell in row:
                if cell:
                    if cell.color == my_color and cell.top_face == 6:
                        reward += 20
                    elif cell.color == opponent_color and cell.top_face == 6:
                        reward -= 20

        for (rr, cc) in board_copy.get_empty_cells():
            options = find_capturing_subsets(board_copy, rr, cc)
            for subset, sum_pips in options:
                if sum_pips == 6:
                    reward -= 10

        if winner:
            if winner == my_color:
                reward += 30

            my_ones = sum(
                1 for row in board.grid for cell in row
                if cell and cell.color == my_color and cell.top_face == 1
            )
            reward += 5 * my_ones

            my_dice = sum(
                1 for row in board.grid for cell in row if cell and cell.color == my_color
            )
            opp_dice = sum(
                1 for row in board.grid for cell in row if cell and cell.color == opponent_color
            )
            reward += 2 * max(0, my_dice - opp_dice)
            reward -= 1 * max(0, opp_dice - my_dice)

        return reward

    from abc import ABC, abstractmethod

    class RewardShaper(ABC):
        @abstractmethod
        def compute(self, board, move, my_color) -> float:
            pass


class AdvancedBoardShaper(RewardShaper):
    """
        Shaper avanzato che:
          - Penalizza se la mossa lascia l'avversario la possibilità di catturare con un 6.
          - Tiene conto della differenza nei 6 controllati sul board.
          - Penalizza se la cattura produce un 5 (dato debole).
          - Alla fine della partita, premia:
               * La vittoria (o penalizza la sconfitta).
               * La differenza nel numero di dadi.
               * Il numero di 1 "safe" (cioè che non sono stati catturati, perché ancora presenti sul board).
    """

    def compute(self, board, move, my_color, opponent_color=None, winner=None):
        if not opponent_color:
            opponent_color = get_opponent(my_color)

        r, c, top_face, captured = move
        reward = 0.0

        # Simula la mossa sul board
        board_copy = copy.deepcopy(board)
        for (rr, cc) in captured:
            board_copy.grid[rr][cc] = None
        board_copy.place_die(r, c, Die(my_color, top_face))

        # Penalizza se la cattura produce un 5 (dato debole)
        new_fives = 0
        for row in board_copy.grid:
            for cell in row:
                if cell and cell.color == my_color and cell.top_face == 5:
                    new_fives += 1
        reward -= 1.1 * new_fives  # Penalizzazione proporzionale

        # Penalità se la mossa lascia l'avversario una cattura immediata con somma 6
        for (rr, cc) in board_copy.get_empty_cells():
            options = find_capturing_subsets(board_copy, rr, cc)
            if options:
                _, sum_pips = choose_capturing_subset(options)
                if sum_pips == 6:
                    reward -= 120
                    break

        # Bilancio dei 6 sul board (importanza strategica del controllo dei 6)
        my_6_count = 0
        opp_6_count = 0
        for row in board_copy.grid:
            for cell in row:
                if cell and cell.top_face == 6:
                    if cell.color == my_color:
                        my_6_count += 1
                    else:
                        opp_6_count += 1
        diff_6 = my_6_count - opp_6_count
        if diff_6 > 0:
            reward += 5 * diff_6
        elif diff_6 < 0:
            reward -= 20 * abs(diff_6)

        # Fase finale: se la partita è terminata
        if winner:
            if winner == my_color:
                reward += 70
            else:
                reward -= 70

            my_dice = sum(1 for row in board.grid for cell in row if cell and cell.color == my_color)
            opp_dice = sum(1 for row in board.grid for cell in row if cell and cell.color == opponent_color)
            reward += (my_dice - opp_dice)

            safe_ones = sum(
                1 for row in board.grid for cell in row
                if cell and cell.color == my_color and cell.top_face == 1
            )
            reward += 10 * safe_ones

        return reward


class OpponentManager:
    def __init__(self):
        self.opponents = [
            ("naive", NaiveStrategy(), 0.4),
            ("smartpos", SmartPositionalLookaheadStrategy(), 0.3),
            ("smartlook5", SmartLookaheadStrategy5(), 0.1),
            ("aggressive", AggressiveStrategy(), 0.2)  # incluso ma disattivato inizialmente
        ]
        self.last_opponent_name = ""

    def choose_opponent(self):
        names, strategies, weights = zip(*self.opponents)
        choice = random.choices(list(zip(names, strategies)), weights=weights, k=1)[0]
        name, strategy = choice
        self.last_opponent_name = name
        return strategy

    def set_weights(self, new_weights: dict):
        updated = []
        for name, strategy, _ in self.opponents:
            weight = new_weights.get(name, 0.0)
            updated.append((name, strategy, weight))
        self.opponents = updated

    def scale_difficulty(self, episode: int, thresholds=(1000, 3000)):
        if episode < thresholds[0]:
            self.set_weights({"naive": 0.6, "smartpos": 0.3, "smartlook5": 0.1, "expectimax": 0.0, "aggressive": 0.0})
        elif episode < thresholds[1]:
            self.set_weights({"naive": 0.2, "smartpos": 0.4, "smartlook5": 0.3, "expectimax": 0.1, "aggressive": 0.0})
        else:
            self.set_weights({"naive": 0.1, "smartpos": 0.2, "smartlook5": 0.3, "expectimax": 0.2, "aggressive": 0.2})
