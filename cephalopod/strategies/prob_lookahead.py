import copy
import random
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.core.board import Die


class ProbabilisticLookaheadStrategy:
    def __init__(self, config=None, debug=False):
        self.name = "ProbabilisticLookahead"
        self.debug = debug
        self.config = config or {
            "six_bonus": 20,
            "low_capture_opportunity_bonus": 2,
            "center_bonus_weight": 2.0,
            "exposure_penalty_weight": 0.5,
            "opponent_response_penalty_weight": 1.0
        }
    def choose_move(self, board, color):
        empty_cells = board.get_empty_cells()
        best_score = float("-inf")
        best_move = None

        for (r, c) in empty_cells:
            # Prova tutte le opzioni di cattura in quella cella
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                for subset, sum_pips in capturing_options:
                    move = (r, c, sum_pips, subset)
                    score = self.evaluate_move(board, move, color)
                    if score > best_score:
                        best_score = score
                        best_move = move

            # Considera anche il piazzamento semplice (non capturing)
            simple_move = (r, c, 1, [])
            score = self.evaluate_move(board, simple_move, color)
            if score > best_score:
                best_score = score
                best_move = simple_move

        return best_move

    def evaluate_move(self, board, move, my_color):
        """
        Restituisce uno score per la mossa:
        - + valore dei dadi catturati (reward)
        - - penalità basata sul rischio di lasciare opportunità da 6 all’avversario (risk)
        """
        board_copy = copy.deepcopy(board)
        r, c, top_face, captured = move

        # Applica la mossa
        for (rr, cc) in captured:
            board_copy.grid[rr][cc] = None
        board_copy.place_die(r, c, Die(my_color, top_face))

        # Calcola guadagno: somma dei pip catturati
        reward = sum(board.grid[rr][cc].top_face for (rr, cc) in captured) if captured else 0

        # Calcola rischio: probabilità stimata che l'avversario possa fare un 6
        opponent_color = "W" if my_color == "B" else "B"
        risk = self.estimate_opponent_capture_risk(board_copy, opponent_color)

        # Bilanciamento: il fattore 2 può essere calibrato
        score = reward - 2 * risk
        return score

    def estimate_opponent_capture_risk(self, board, opponent_color):
        """
        Stima la probabilità che l’avversario abbia una mossa che produce un 6.
        Valore tra 0 e 1 proporzionale alle celle pericolose.
        """
        empty = board.get_empty_cells()
        dangerous = 0
        for (r, c) in empty:
            options = find_capturing_subsets(board, r, c)
            for subset, sum_pips in options:
                if sum_pips == 6:
                    dangerous += 1
                    break
        if not empty:
            return 0
        return dangerous / len(empty)
