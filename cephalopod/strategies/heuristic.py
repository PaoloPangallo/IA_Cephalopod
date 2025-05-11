# strategies/heuristic.py

import random

from cephalopod.cephalopod_game_naive import find_capturing_subsets, choose_capturing_subset
from strategies import NaiveStrategy


def choose_best_move(board, color):
    """
    Euristica semplice: per ogni cella vuota,
    valuta la somma dei pips catturabili e sceglie la mossa con il valore massimo.
    In caso di parità o se non ci sono catture, si ricorre alla strategia naive.
    """
    empty_cells = board.get_empty_cells()
    best_moves = []
    best_sum = -1
    for (r, c) in empty_cells:
        capturing_options = find_capturing_subsets(board, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            if sum_pips > best_sum:
                best_sum = sum_pips
                best_moves = [(r, c, sum_pips, subset)]
            elif sum_pips == best_sum:
                best_moves.append((r, c, sum_pips, subset))
    if best_moves:
        return random.choice(best_moves)
    else:
        return NaiveStrategy().choose_move(board, color)

class HeuristicStrategy:
    def choose_move(self, board, color):
        """Utilizza una funzione euristica per scegliere la mossa migliore."""
        return choose_best_move(board, color)
