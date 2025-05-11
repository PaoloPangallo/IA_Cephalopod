# cephalopod_game_heuristic.py
import copy
from .cephalopod_game_naive import (
    Board,
    Die,
    get_opponent,
    find_capturing_subsets,
    choose_capturing_subset,
    BOARD_SIZE
)

def evaluate_board(board, player):
    """
    Funzione di valutazione che tiene conto di:
      1. Vantaggio numerico.
      2. Opportunità di cattura.
      3. Controllo posizionale.
    Restituisce un punteggio: valori più alti indicano posizioni favorevoli per 'player'.
    """
    opponent = "W" if player == "B" else "B"
    score = 0

    # 1. Vantaggio numerico: +10 per ogni dado del giocatore, -10 per ogni dado avversario.
    player_count = 0
    opponent_count = 0
    for r in range(board.size):
        for c in range(board.size):
            if board.grid[r][c] is not None:
                d = board.grid[r][c]
                if d.color == player:
                    player_count += 1
                elif d.color == opponent:
                    opponent_count += 1
    score += (player_count - opponent_count) * 10

    # 2. Opportunità di cattura: bonus per ogni cella vuota adiacente a dadi avversari.
    capture_bonus = 0
    for r in range(board.size):
        for c in range(board.size):
            if board.grid[r][c] is None:
                adjacent = []
                for (rr, cc) in board.orthogonal_neighbors(r, c):
                    if board.grid[rr][cc] is not None:
                        d = board.grid[rr][cc]
                        if d.color == opponent:
                            adjacent.append(d.top_face)
                if len(adjacent) >= 2 and sum(adjacent) <= 6:
                    capture_bonus += 5 * len(adjacent)
    score += capture_bonus

    # 3. Controllo posizionale: bonus per il centro.
    weights = [
        [1, 1, 1, 1, 1],
        [1, 2, 2, 2, 1],
        [1, 2, 3, 2, 1],
        [1, 2, 2, 2, 1],
        [1, 1, 1, 1, 1]
    ]
    pos_bonus = 0
    for r in range(board.size):
        for c in range(board.size):
            if board.grid[r][c] is not None:
                d = board.grid[r][c]
                if d.color == player:
                    pos_bonus += weights[r][c]
                elif d.color == opponent:
                    pos_bonus -= weights[r][c]
    score += pos_bonus * 3
    return score

def choose_best_move(board, player):
    """
    Itera su tutte le celle vuote della board, simula il piazzamento e valuta il punteggio.
    Ritorna la mossa migliore come (r, c, top_face, captured).
    """
    best_score = -float('inf')
    best_move = None
    for (r, c) in board.get_empty_cells():
        board_copy = copy.deepcopy(board)
        capturing_options = find_capturing_subsets(board_copy, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            for (rr, cc) in subset:
                board_copy.grid[rr][cc] = None
            top_face = sum_pips
            captured = subset
        else:
            top_face = 1
            captured = []
        board_copy.place_die(r, c, Die(player, top_face))
        score = evaluate_board(board_copy, player)
        if score > best_score:
            best_score = score
            best_move = (r, c, top_face, captured)
    return best_move

class CephalopodGameHeuristic:
    def __init__(self, max_dice_per_player=24):
        self.board = Board()
        self.max_dice_per_player = max_dice_per_player
        self.dice_remaining = {"B": max_dice_per_player, "W": max_dice_per_player}
        self.current_player = "B"
        self.moves_log = []
        self.move_num = 1

    def simulate_move(self):
        empty_cells = self.board.get_empty_cells()
        if not empty_cells:
            return False

        best_move = choose_best_move(self.board, self.current_player)
        if best_move is None:
            return False
        r, c, top_face, captured = best_move

        for (rr, cc) in captured:
            self.board.grid[rr][cc] = None
        new_die = Die(self.current_player, top_face)
        self.board.place_die(r, c, new_die)
        self.dice_remaining[self.current_player] -= 1

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
    game = CephalopodGameHeuristic(max_dice_per_player=24)
    log = game.simulate_game()
    for move in log:
        print(move)
