# strategies/naive.py

import random

from cephalopod.cephalopod_game_naive import find_capturing_subsets, choose_capturing_subset


class NaiveStrategy:
    def choose_move(self, board, color):
        """Scelta casuale della mossa (con eventuali catture)."""
        empty_cells = board.get_empty_cells()
        if not empty_cells:
            return None
        r, c = random.choice(empty_cells)
        capturing_options = find_capturing_subsets(board, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            top_face = sum_pips
            captured = subset
        else:
            top_face = 1
            captured = []
        return (r, c, top_face, captured)
