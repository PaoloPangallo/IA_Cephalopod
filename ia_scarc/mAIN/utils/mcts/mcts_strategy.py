import time
from mAIN.utils.strategy_utils import get_opponent, get_all_legal_moves, simulate_move


def evaluate_board(board, player):
    opponent = get_opponent(player)
    score = 0
    for r in range(board.size):
        for c in range(board.size):
            cell = board.board[r][c]
            if cell is not None:
                cell_player, pip = cell
                if cell_player == player:
                    score += 1
                    if pip == 6:
                        score += 3
                elif cell_player == opponent:
                    score -= 1
                    if pip == 6:
                        score -= 3
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


class AdaptiveSimpleMinimaxStrategy:
    def choose_move(self, game, state, player):
        start_time = time.time()

        board = state.board
        total_cells = state.size * state.size
        occupied = sum(1 for row in board for cell in row if cell)
        occupancy_ratio = occupied / total_cells

        if occupancy_ratio < 0.4:
            depth = 2
        elif occupancy_ratio < 0.6:
            depth = 3
        elif occupancy_ratio < 0.8:
            depth = 4
        else:
            depth = 5

        print(f"[ADAPTIVE MINIMAX] Occupazione: {occupancy_ratio:.2f} → Depth: {depth}")

        _, best_move = minimax(
            board=state,
            depth=depth,
            player=player,
            maximizing_player=True,
            original_player=player
        )

        elapsed = time.time() - start_time
        print(f"[ADAPTIVE MINIMAX] Tempo per mossa (depth {depth}): {elapsed:.2f}s")

        return best_move
