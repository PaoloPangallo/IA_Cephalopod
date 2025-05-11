
from mAIN.utils.strategy_utils import get_all_legal_moves, simulate_move, get_opponent
import random


def evaluate_board(board, player, opponent):
    score = 0
    for r in range(board.size):
        for c in range(board.size):
            die = board.board[r][c]
            if die:
                if die[0] == player:
                    score += 1
                    if die[1] == 6:
                        score += 3
                elif die[0] == opponent:
                    score -= 1
                    if die[1] == 6:
                        score -= 2
    return score


def simulate_opponent_response(board, player, move):
    new_board = simulate_move(board, move, player)
    opponent = get_opponent(player)
    opponent_moves = get_all_legal_moves(new_board, opponent)

    penalty = 0
    for move in opponent_moves:
        _, pip, captured = move
        if captured:
            if pip == 6:
                penalty += 4
            elif len(captured) >= 2:
                penalty += 2
    return penalty


def reward_own_capture(board, move, player):
    _, _, captured = move
    reward = 0
    for (rr, cc) in captured:
        die = board.board[rr][cc]
        if die and die[0] == get_opponent(player):
            reward += 2
    return reward


def minimax(board, depth, player, maximizing, original_player, alpha=-float("inf"), beta=float("inf")):
    if depth == 0 or board.is_full():
        return evaluate_board(board, original_player, get_opponent(original_player)), None

    moves = get_all_legal_moves(board, player)
    best_move = None

    if maximizing:
        max_score = -float("inf")
        for move in moves:
            simulated = simulate_move(board, move, player)
            score, _ = minimax(simulated, depth - 1, get_opponent(player), False, original_player, alpha, beta)
            score -= simulate_opponent_response(board, player, move)
            score += reward_own_capture(board, move, player)
            if score > max_score:
                max_score = score
                best_move = move
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return max_score, best_move
    else:
        min_score = float("inf")
        for move in moves:
            simulated = simulate_move(board, move, player)
            score, _ = minimax(simulated, depth - 1, get_opponent(player), True, original_player, alpha, beta)
            score += simulate_opponent_response(board, player, move)
            score -= reward_own_capture(board, move, player)
            if score < min_score:
                min_score = score
                best_move = move
            beta = min(beta, score)
            if beta <= alpha:
                break
        return min_score, best_move


class ResilientMinimaxStrategy:
    def __init__(self, depth=3):
        self.depth = depth

    def choose_move(self, game, board, color):  # 👈 AGGIUNGI `game`
        _, best_move = minimax(board, self.depth, color, True, color)
        return best_move


def playerStrategy(game, state):
    strategy = ResilientMinimaxStrategy(depth=3)
    return strategy.choose_move(state, state.to_move)