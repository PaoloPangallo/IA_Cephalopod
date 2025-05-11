# strategies/aggressive.py

import random

from cephalopod.cephalopod_game_naive import find_capturing_subsets, choose_capturing_subset


def choose_aggressive_move(board, color):
    """
    Strategia che massimizza immediatamente le catture.
    Per ogni cella vuota, cerca i sottoinsiemi catturabili e sceglie quella con il maggior numero di dadi.
    Se non trova catture, ritorna una mossa fallback.
    """
    empty_cells = board.get_empty_cells()
    if not empty_cells:
        return None

    best_move = None
    max_captured = -1
    fallback_move = None

    for (r, c) in empty_cells:
        capturing_options = find_capturing_subsets(board, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            capture_size = len(subset)
            if capture_size > max_captured:
                max_captured = capture_size
                best_move = (r, c, sum_pips, subset)
        else:
            if fallback_move is None:
                fallback_move = (r, c, 1, [])
    if best_move is not None:
        return best_move
    return fallback_move


class AggressiveStrategy:
    def choose_move(self, board, color):
        """Utilizza la funzione 'choose_aggressive_move' per massimizzare le catture immediate."""
        return choose_aggressive_move(board, color)
