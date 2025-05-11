import time
from mAIN.utils.strategy_utils import get_opponent, get_all_legal_moves, simulate_move


class DynamicMinimaxStrategyAdrian:
    def __init__(self, time_limit=3.0):
        self.time_limit = time_limit  # In secondi

    def choose_move(self, game, state, player):
        start = time.perf_counter()
        best_move = None
        depth = 1

        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= self.time_limit:
                break

            score, move = self.minimax(state, depth, player, True, player, start)

            elapsed = time.perf_counter() - start
            if elapsed >= self.time_limit:
                break

            if move is not None:
                best_move = move

            depth += 1

        print(f"[DYNAMIC MINIMAX ADRIAN]  Scelta mossa: {best_move} alla profondità {depth - 1} in {elapsed:.2f}s")
        return best_move

    def minimax(self, board, depth, player, maximizing_player, original_player, start_time, alpha=-float("inf"),
                beta=float("inf")):
        if time.perf_counter() - start_time >= self.time_limit or depth == 0 or board.is_full():
            return self.evaluate_board(board, original_player), None

        possible_moves = get_all_legal_moves(board, player)
        best_move = None

        if maximizing_player:
            max_score = -float("inf")
            for move in possible_moves:
                next_board = simulate_move(board, move, player)
                score, _ = self.minimax(next_board, depth - 1, get_opponent(player), False, original_player, start_time,
                                        alpha, beta)
                if score > max_score:
                    max_score = score
                    best_move = move
                alpha = max(alpha, score)
                if beta <= alpha or time.perf_counter() - start_time >= self.time_limit:
                    break
            return max_score, best_move
        else:
            min_score = float("inf")
            for move in possible_moves:
                next_board = simulate_move(board, move, player)
                score, _ = self.minimax(next_board, depth - 1, get_opponent(player), True, original_player, start_time,
                                        alpha, beta)
                if score < min_score:
                    min_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha or time.perf_counter() - start_time >= self.time_limit:
                    break
            return min_score, best_move

    def evaluate_board(self, board, player):
        opponent = get_opponent(player)
        score = 0
        for r in range(board.size):
            for c in range(board.size):
                cell = board.board[r][c]
                if cell is not None:
                    cell_player, pip = cell
                    if cell_player == player:
                        score += 1 << (pip - 1)

                    elif cell_player == opponent:
                        score -= 1 << (pip - 1)

        return score


