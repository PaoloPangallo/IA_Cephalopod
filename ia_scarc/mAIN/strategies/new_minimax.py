import time
from functools import lru_cache
from mAIN.utils.strategy_utils import get_opponent, get_all_legal_moves, simulate_move


class DynamicMinimaxStrategy2:
    def __init__(self, time_limit=3.0):
        self.time_limit = time_limit

    def choose_move(self, game, state, player):
        start = time.time()
        best_move = None
        depth = 1

        while True:
            if time.time() - start >= self.time_limit:
                break

            score, move = self.minimax(state, depth, player, True, player, start)
            if time.time() - start >= self.time_limit:
                break
            if move is not None:
                best_move = move
            depth += 1

        elapsed = time.time() - start
        print(f"[DYNAMIC MINIMAX] Scelta mossa: {best_move} alla profondità {depth - 1} in {elapsed:.2f}s")
        return best_move

    def minimax(self, board, depth, player, maximizing_player, original_player, start_time, alpha=-float("inf"),
                beta=float("inf")):
        if time.time() - start_time >= self.time_limit or depth == 0 or board.is_full():
            return self.evaluate_board_cached(self.board_to_tuple(board), original_player), None

        possible_moves = self.order_moves(get_all_legal_moves(board, player))
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
                    if max_score == float('inf'):  # early exit
                        break
                alpha = max(alpha, score)
                if beta <= alpha or time.time() - start_time >= self.time_limit:
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
                    if min_score == -float('inf'):  # early exit
                        break
                beta = min(beta, score)
                if beta <= alpha or time.time() - start_time >= self.time_limit:
                    break
            return min_score, best_move

    def order_moves(self, moves):
        # Priorità a pip alti (es. 6) per aiutare pruning
        return sorted(moves, key=lambda m: m[2], reverse=True)

    def board_to_tuple(self, board):
        return tuple(tuple(cell if cell is None else (cell[0], cell[1]) for cell in row) for row in board.board)

    @lru_cache(maxsize=None)
    def evaluate_board_cached(self, board_key, player):
        return self.evaluate_board_from_tuple(board_key, player)

    def evaluate_board_from_tuple(self, board_key, player):
        opponent = get_opponent(player)
        my_pieces = 0
        opponent_pieces = 0

        for row in board_key:
            for cell in row:
                if cell is not None:
                    cell_player, pip = cell
                    if cell_player == player:
                        my_pieces += 1
                    elif cell_player == opponent:
                        opponent_pieces += 1

        return my_pieces - opponent_pieces

