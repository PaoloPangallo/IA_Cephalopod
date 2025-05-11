import copy
from cephalopod.core.board import Die
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset


def get_opponent(color):
    return "W" if color == "B" else "B"


def simulate_move(board, move, player):
    board_copy = copy.deepcopy(board)
    r, c, top_face, captured = move
    for rr, cc in captured:
        board_copy.grid[rr][cc] = None
    board_copy.place_die(r, c, Die(player, top_face))
    return board_copy


def get_all_legal_moves(board, player):
    moves = []
    for (r, c) in board.get_empty_cells():
        capturing_options = find_capturing_subsets(board, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            moves.append((r, c, sum_pips, subset))
        else:
            moves.append((r, c, 1, []))
    return moves


def evaluate_board(board, player, opponent):
    score = 0
    for r in range(board.size):
        for c in range(board.size):
            die = board.grid[r][c]
            if die:
                if die.color == player:
                    score += 1
                    if die.top_face == 6:
                        score += 3
                elif die.color == opponent:
                    score -= 1
                    if die.top_face == 6:
                        score -= 2
    return score


def simulate_opponent_response(board, player, move):
    new_board = simulate_move(board, move, player)
    opponent = get_opponent(player)
    opponent_moves = get_all_legal_moves(new_board, opponent)

    penalty = 0
    for move in opponent_moves:
        _, _, sum_pips, subset = move
        if subset:
            if sum_pips == 6:
                penalty += 4  # Peggio: permette un 6
            elif len(subset) >= 2:
                penalty += 2  # Cattura multipla
    return penalty


def reward_own_capture(board, move, player):
    _, _, _, subset = move
    reward = 0
    for (rr, cc) in subset:
        die = board.grid[rr][cc]
        if die and die.color == get_opponent(player):
            reward += 2  # bonus per ogni pezzo avversario catturato
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

    def choose_move(self, board, color):
        _, best_move = minimax(board, self.depth, color, True, color)
        return best_move
