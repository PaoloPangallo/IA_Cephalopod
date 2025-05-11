# minimax_strategies.py
import copy

from mAIN.utils.strategy_utils import get_opponent, get_all_legal_moves, simulate_move


def evaluate_board(board, player):
    opponent = get_opponent(player)
    score = 0
    for r in range(board.size):
        for c in range(board.size):
            cell = board.board[r][c]
            if cell is not None:
                color, pip = cell
                if color == player:
                    score += 1
                    if pip == 6:
                        score += 3
                elif color == opponent:
                    score -= 1
                    if pip == 6:
                        score -= 2
    return score


def minimax(board, depth, player, maximizing_player, original_player, alpha=-float("inf"), beta=float("inf")):
    if depth == 0 or board.is_full():
        return evaluate_board(board, original_player), None

    possible_moves = get_all_legal_moves(board, player)
    best_move = None

    if maximizing_player:
        max_score = -float("inf")
        for move in possible_moves:
            next_board = simulate_move(board, move, player)
            score, _ = minimax(next_board, depth - 1, get_opponent(player), False, original_player, alpha, beta)
            if score > max_score:
                max_score = score
                best_move = move
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return max_score, best_move

    else:
        min_score = float("inf")
        for move in possible_moves:
            next_board = simulate_move(board, move, player)
            score, _ = minimax(next_board, depth - 1, get_opponent(player), True, original_player, alpha, beta)
            if score < min_score:
                min_score = score
                best_move = move
            beta = min(beta, score)
            if beta <= alpha:
                break
        return min_score, best_move


class MinimaxStrategy:
    def __init__(self, depth=3):
        self.depth = depth

    def choose_move(self, board, color):
        _, best_move = minimax(
            board,
            depth=self.depth,
            player=color,
            maximizing_player=True,
            original_player=color
        )
        return best_move
