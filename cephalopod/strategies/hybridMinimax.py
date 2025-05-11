import copy
from cephalopod.core.board import Die
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset


def opponent_can_capture_six(board, opponent_color):
    for (r, c) in board.get_empty_cells():
        capturing_options = find_capturing_subsets(board, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            if sum_pips == 6:
                return True
    return False


def evaluate_board_weighted(board, player):
    opponent = "W" if player == "B" else "B"
    score = 0
    for r in range(board.size):
        for c in range(board.size):
            die = board.grid[r][c]
            if die is not None:
                if die.color == player:
                    score += 1
                    if die.top_face == 6:
                        score += 4
                else:
                    score -= 1
                    if die.top_face == 6:
                        score -= 3
    return score


def simulate_move(board, move, player):
    new_board = copy.deepcopy(board)
    r, c, top_face, captured = move
    for (rr, cc) in captured:
        new_board.grid[rr][cc] = None
    new_board.place_die(r, c, Die(player, top_face))
    return new_board


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


def get_opponent(player):
    return "W" if player == "B" else "B"


def hybrid_minimax(board, depth, player, maximizing_player, original_player, alpha=-float("inf"), beta=float("inf")):
    if depth == 0 or board.is_full():
        return evaluate_board_weighted(board, original_player), None

    possible_moves = get_all_legal_moves(board, player)
    best_move = None

    if maximizing_player:
        max_eval = -float("inf")
        for move in possible_moves:
            next_board = simulate_move(board, move, player)
            opponent = get_opponent(player)
            if opponent_can_capture_six(next_board, opponent):
                eval = -10
            else:
                eval, _ = hybrid_minimax(next_board, depth - 1, opponent, False, original_player, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for move in possible_moves:
            next_board = simulate_move(board, move, player)
            eval, _ = hybrid_minimax(next_board, depth - 1, get_opponent(player), True, original_player, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move


class HybridMinimaxStrategy:
    def __init__(self, depth=2):
        self.depth = depth

    def choose_move(self, board, color):
        _, move = hybrid_minimax(
            board,
            depth=self.depth,
            player=color,
            maximizing_player=True,
            original_player=color
        )
        return move
