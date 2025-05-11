import copy
import random
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.core.board import Die

class TunableMinimaxStrategy:
    def __init__(self, weights=None, depth=30):
        self.weights = weights or {
            "capture_6": 6,
            "avoid_enemy_6": -12,
            "safe_capture": 2,
            "border_position": 2,
            "random_1": 0
        }
        self.depth = depth

    def choose_move(self, board, color):
        _, move = self.minimax(board, self.depth, color, True, color)
        return move

    def minimax(self, board, depth, player, maximizing_player, original_player, alpha=float("-inf"), beta=float("inf")):
        if depth == 0 or board.is_full():
            return self.evaluate_board(board, original_player), None

        possible_moves = self.generate_moves(board, player)
        if not possible_moves:
            return self.evaluate_board(board, original_player), None

        best_move = None

        if maximizing_player:
            best_score = -float("inf")
            for move in possible_moves:
                new_board = self.simulate_move(board, move, player)
                score, _ = self.minimax(new_board, depth - 1, self.get_opponent(player), False, original_player, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return best_score, best_move
        else:
            best_score = float("inf")
            for move in possible_moves:
                new_board = self.simulate_move(board, move, player)
                score, _ = self.minimax(new_board, depth - 1, self.get_opponent(player), True, original_player, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return best_score, best_move

    def generate_moves(self, board, color):
        moves = []
        for (r, c) in board.get_empty_cells():
            capturing = find_capturing_subsets(board, r, c)
            if capturing:
                subset, sum_pips = choose_capturing_subset(capturing)
                if sum_pips == 6 or sum_pips <= 4:
                    moves.append((r, c, sum_pips, subset))
                elif sum_pips == 5:
                    continue
            else:
                moves.append((r, c, 1, []))
        return moves

    def simulate_move(self, board, move, color):
        board_copy = copy.deepcopy(board)
        r, c, top, captured = move
        for rr, cc in captured:
            board_copy.grid[rr][cc] = None
        board_copy.place_die(r, c, Die(color, top))
        return board_copy

    def evaluate_board(self, board, my_color):
        opponent = self.get_opponent(my_color)
        score = 0
        for r in range(len(board.grid)):
            for c in range(len(board.grid[0])):
                die = board.grid[r][c]
                if die is not None:
                    if die.color == my_color:
                        score += 1
                        if die.top_face == 6:
                            score += self.weights["capture_6"]
                    elif die.color == opponent:
                        score -= 1
                        if die.top_face == 6:
                            score += self.weights["avoid_enemy_6"]
        return score

    def is_border_cell(self, r, c, rows, cols):
        return r == 0 or c == 0 or r == rows - 1 or c == cols - 1

    def get_opponent(self, color):
        return "W" if color == "B" else "B"

class EliteStrategy(TunableMinimaxStrategy):
    def __init__(self):
        super().__init__(weights={
            "capture_6": 6,
            "avoid_enemy_6": -12,
            "safe_capture": 0,
            "border_position": 0,
            "random_1": 0
        }, depth=6)
