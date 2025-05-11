# core/mechanics.py

from itertools import combinations

def get_opponent(color):
    """Restituisce il colore avversario (B -> W, W -> B)."""
    return "W" if color == "B" else "B"

def find_capturing_subsets(board, r, c):
    """
    Data la posizione (r,c) in cui verrà piazzato un dado,
    restituisce tutti i sottoinsiemi (di dimensione 2..n) dei dadi adiacenti (ortogonalmente)
    la cui somma dei pips è <= 6.
    Ogni opzione è una tupla (subset, sum_pips) in cui subset è una lista di coordinate.
    """
    adjacent_positions = []
    for (rr, cc) in board.orthogonal_neighbors(r, c):
        if board.grid[rr][cc] is not None:
            adjacent_positions.append((rr, cc))

    capturing_options = []
    for size in range(2, len(adjacent_positions) + 1):
        for combo in combinations(adjacent_positions, size):
            sum_pips = sum(board.grid[pos[0]][pos[1]].top_face for pos in combo)
            if sum_pips <= 6:
                capturing_options.append((list(combo), sum_pips))
    return capturing_options

def choose_capturing_subset(capturing_options):
    """
    Se sono disponibili più sottoinsiemi catturabili,
    sceglie quello di dimensione maggiore (in caso di parità, il primo).
    """
    if not capturing_options:
        return None, None
    capturing_options.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
    best_subset, best_sum = capturing_options[0]
    return best_subset, best_sum
