import random
from itertools import combinations

BOARD_SIZE = 5


def get_opponent(color):
    """Restituisce il colore avversario (B->W, W->B)."""
    return "W" if color == "B" else "B"


class Board:
    def __init__(self, size=BOARD_SIZE):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]

    def is_full(self):
        """True se tutte le celle sono occupate."""
        return all(self.grid[r][c] is not None for r in range(self.size) for c in range(self.size))

    def get_empty_cells(self):
        """Restituisce una lista di tuple (r, c) delle celle vuote."""
        return [(r, c) for r in range(self.size)
                for c in range(self.size) if self.grid[r][c] is None]

    def in_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def orthogonal_neighbors(self, r, c):
        """Restituisce le celle ortogonalmente adiacenti a (r,c)."""
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if self.in_bounds(rr, cc):
                yield (rr, cc)

    def place_die(self, r, c, die):
        self.grid[r][c] = die


class Die:
    def __init__(self, color, top_face):
        self.color = color
        self.top_face = top_face

    def __repr__(self):
        return f"({self.color},{self.top_face})"


def find_capturing_subsets(board, r, c):
    """
    Dato che stiamo per piazzare un dado in (r,c),
    restituisce tutti i sottoinsiemi (di dimensione 2..n)
    dei dadi adiacenti (ortogonalmente) la cui somma dei pips è <= 6.
    Ogni opzione viene restituita come (subset, sum_pips)
    dove subset è una lista di coordinate.
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
    Se ci sono più sottoinsiemi catturabili, scegliamo quello di massima dimensione,
    e in caso di parità il primo.
    """
    if not capturing_options:
        return None, None
    capturing_options.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
    best_subset, best_sum = capturing_options[0]
    return best_subset, best_sum


class CephalopodGameNaive:
    def __init__(self, max_dice_per_player=24):
        self.board = Board()
        self.current_player = "B"
        self.moves_log = []
        self.move_num = 1

    def simulate_move(self):
        empty_cells = self.board.get_empty_cells()
        if not empty_cells:
            return False

        r, c = random.choice(empty_cells)
        capturing_options = find_capturing_subsets(self.board, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            for (rr, cc) in subset:
                self.board.grid[rr][cc] = None
            top_face = sum_pips
            captured = subset
        else:
            top_face = 1
            captured = []

        new_die = Die(self.current_player, top_face)
        self.board.place_die(r, c, new_die)
        # Non decrementiamo più i dadi: la partita continua finché esistono mosse legali

        self.moves_log.append({
            "move_num": self.move_num,
            "player": self.current_player,
            "row": r,
            "col": c,
            "top_face": top_face,
            "captured": captured
        })
        self.move_num += 1
        self.current_player = get_opponent(self.current_player)
        return True

    def simulate_game(self):
        # Il ciclo continua finché la board non è piena oppure simulate_move() restituisce False
        while not self.board.is_full():
            if not self.simulate_move():
                break

        b_count = sum(1 for r in range(self.board.size) for c in range(self.board.size)
                      if self.board.grid[r][c] is not None and self.board.grid[r][c].color == "B")
        w_count = sum(1 for r in range(self.board.size) for c in range(self.board.size)
                      if self.board.grid[r][c] is not None and self.board.grid[r][c].color == "W")
        winner = "B" if b_count > w_count else "W"
        self.moves_log.append({
            "move_num": self.move_num,
            "player": "END",
            "row": -1,
            "col": -1,
            "top_face": -1,
            "captured": f"FinalCount => B:{b_count}, W:{w_count}"
        })
        self.moves_log.append({
            "move_num": self.move_num + 1,
            "player": "WINNER",
            "row": -1,
            "col": -1,
            "top_face": -1,
            "captured": winner
        })
        return self.moves_log


if __name__ == "__main__":
    game = CephalopodGameNaive()
    log = game.simulate_game()
    for move in log:
        print(move)
