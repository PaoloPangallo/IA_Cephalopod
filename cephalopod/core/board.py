# core/board.py

BOARD_SIZE = 5


class Board:



    def __init__(self, size=BOARD_SIZE):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]

    def is_full(self):
        """Ritorna True se tutte le celle sono occupate."""
        return all(self.grid[r][c] is not None for r in range(self.size) for c in range(self.size))

    def get_empty_cells(self):
        """Ritorna una lista di tuple (r, c) delle celle vuote."""
        return [(r, c) for r in range(self.size)
                for c in range(self.size) if self.grid[r][c] is None]

    def in_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def orthogonal_neighbors(self, r, c):
        """Restituisce i vicini ortogonali di (r,c)."""
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if self.in_bounds(rr, cc):
                yield rr, cc

    def place_die(self, r, c, die):
        self.grid[r][c] = die

    def clone(self):
        from copy import deepcopy
        new_board = Board()
        new_board.grid = deepcopy(self.grid)
        return new_board


class Die:
    def __init__(self, color, top_face):
        self.color = color
        self.top_face = top_face

    def __repr__(self):
        return f"({self.color},{self.top_face})"
