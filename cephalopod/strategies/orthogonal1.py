# strategies/orthogonal1.py

import random

from cephalopod.strategies import NaiveStrategy


class Orthogonal1Strategy:
    def choose_move(self, board, color):
        """
        Se esiste una cella vuota ortogonalmente adiacente a un dado con top_face == 5,
        piazza un dado con top_face = 1; altrimenti, esegue una mossa casuale.
        """
        candidates = []
        for (r, c) in board.get_empty_cells():
            for (nr, nc) in board.orthogonal_neighbors(r, c):
                neighbor = board.grid[nr][nc]
                if neighbor is not None and neighbor.top_face == 5:
                    candidates.append((r, c))
                    break
        if candidates:
            chosen = random.choice(candidates)
            return (chosen[0], chosen[1], 1, [])
        else:
            return NaiveStrategy().choose_move(board, color)
