import copy
import random
from cephalopod.core.board import Die
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset


def evaluate_board(board, player):
    opponent = "W" if player == "B" else "B"
    score = 0
    for r in range(board.size):
        for c in range(board.size):
            die = board.grid[r][c]
            if die is not None:
                if die.color == player:
                    score += 1
                    if die.top_face == 6:
                        score += 3
                elif die.color == opponent:
                    score -= 1
                    if die.top_face == 6:
                        score -= 2
    return score


def get_all_legal_moves(board, player):
    moves = []
    for (r, c) in board.get_empty_cells():
        capturing_options = find_capturing_subsets(board, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            if sum_pips == 6 or sum_pips <= 4:
                moves.append((r, c, sum_pips, subset))
            elif sum_pips == 5:
                continue  # scartiamo catture "pericolose"
        else:
            moves.append((r, c, 1, []))
    return moves


def simulate_move(board, move, player):
    new_board = copy.deepcopy(board)
    r, c, top_face, captured = move
    for (rr, cc) in captured:
        new_board.grid[rr][cc] = None
    new_board.place_die(r, c, Die(player, top_face))
    return new_board


def get_opponent(player):
    return "W" if player == "B" else "B"


def minimax(board, depth, player, maximizing_player, original_player, alpha=float("-inf"), beta=float("inf")):
    if depth == 0 or board.is_full():
        return evaluate_board(board, original_player), None

    possible_moves = get_all_legal_moves(board, player)
    if not possible_moves:
        return evaluate_board(board, original_player), None

    best_move = None

    if maximizing_player:
        best_score = -float("inf")
        for move in possible_moves:
            next_board = simulate_move(board, move, player)
            score, _ = minimax(next_board, depth - 1, get_opponent(player), False, original_player, alpha, beta)
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
            next_board = simulate_move(board, move, player)
            score, _ = minimax(next_board, depth - 1, get_opponent(player), True, original_player, alpha, beta)
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, score)
            if beta <= alpha:
                break
        return best_score, best_move


class SmartMinimaxStrategy:
    def __init__(self, depth=3):
        self.depth = depth

    def choose_move(self, board, color):
        _, move = minimax(board, self.depth, color, True, color)
        return move