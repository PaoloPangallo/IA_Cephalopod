import copy
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
            moves.append((r, c, sum_pips, subset))
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


# ----------- MINIMAX ------------ #
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


# ----------- EXPECTIMAX ------------ #
def expectimax(board, depth, player, maximizing_player, original_player):
    if depth == 0 or board.is_full():
        return evaluate_board(board, original_player), None

    possible_moves = get_all_legal_moves(board, player)
    best_move = None

    if maximizing_player:
        max_score = -float("inf")
        for move in possible_moves:
            next_board = simulate_move(board, move, player)
            score, _ = expectimax(next_board, depth - 1, get_opponent(player), False, original_player)
            if score > max_score:
                max_score = score
                best_move = move
        return max_score, best_move

    else:
        total_score = 0
        for move in possible_moves:
            next_board = simulate_move(board, move, player)
            score, _ = expectimax(next_board, depth - 1, get_opponent(player), True, original_player)
            total_score += score
        avg_score = total_score / len(possible_moves) if possible_moves else 0
        return avg_score, None


class ExpectimaxStrategy:
    def __init__(self, depth=2):
        self.depth = depth

    def choose_move(self, board, color):
        _, best_move = expectimax(
            board,
            depth=self.depth,
            player=color,
            maximizing_player=True,
            original_player=color
        )
        return best_move
