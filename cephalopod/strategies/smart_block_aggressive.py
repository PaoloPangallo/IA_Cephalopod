import random
from cephalopod.strategies.naive import find_capturing_subsets, choose_capturing_subset


class SmartBlockAggressiveStrategy:
    def choose_move(self, board, color):
        empty_cells = board.get_empty_cells()
        fallback_move = None

        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                if sum_pips == 6:
                    # massima cattura → prioritaria
                    return r, c, sum_pips, subset
                elif sum_pips <= 4:
                    # cattura aggressiva → ok
                    return r, c, sum_pips, subset
                else:
                    # somma 5 → non ci piace, saltiamo
                    continue
            else:
                if fallback_move is None:
                    fallback_move = (r, c, 1, [])

        # Se non abbiamo trovato catture buone, blocchiamo i nemici con top_face=5
        for (r, c) in empty_cells:
            for (nr, nc) in board.orthogonal_neighbors(r, c):
                neighbor = board.grid[nr][nc]
                if neighbor and neighbor.color != color and neighbor.top_face == 5:
                    return (r, c, 1, [])

        # Nessun blocco possibile, fallback
        return fallback_move or random.choice(empty_cells) + (1, [])
