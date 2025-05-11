import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

from mAIN.utils.strategy_utils import get_opponent, get_all_legal_moves, simulate_move


class ParallelDynamicMinimaxStrategy:
    def __init__(self, time_limit=3.0):
        self.time_limit = time_limit

    def choose_move(self, game, state, player):
        start = time.time()
        best_move = None
        depth = 1

        while True:
            if time.time() - start >= self.time_limit:
                break

            result = self.parallel_minimax_root(state, depth, player, start)
            if result is None:
                break
            score, move = result

            if move is not None:
                best_move = move

            if time.time() - start >= self.time_limit:
                break

            depth += 1

        elapsed = time.time() - start
        print(f"[DYNAMIC MINIMAX - PARALLEL] Scelta mossa: {best_move} alla profondità {depth - 1} in {elapsed:.2f}s")
        return best_move

    def parallel_minimax_root(self, board, depth, player, start_time):
        legal_moves = get_all_legal_moves(board, player)
        if not legal_moves:
            return None

        best_score = float('-inf')
        best_moves = []

        with ThreadPoolExecutor(max_workers=min(cpu_count(), len(legal_moves))) as executor:
            futures = {
                executor.submit(
                    lambda b=simulate_move(board, move, player): self.minimax(
                        b, depth - 1, get_opponent(player), False, player, start_time
                    )
                ): move
                for move in legal_moves
            }

            for future in as_completed(futures):
                if time.time() - start_time >= self.time_limit:
                    break
                try:
                    score, _ = future.result()
                    move = futures[future]
                    if score > best_score:
                        best_score = score
                        best_moves = [move]
                    elif score == best_score:
                        best_moves.append(move)
                except Exception as e:
                    continue  # Silenzia eventuali errori

        if best_moves:
            return best_score, random.choice(best_moves)
        return None

    def minimax(self, board, depth, player, maximizing_player, original_player, start_time, alpha=-float("inf"),
                beta=float("inf")):
        if time.time() - start_time >= self.time_limit or depth == 0 or board.is_full():
            return self.evaluate_board(board, original_player), None

        possible_moves = get_all_legal_moves(board, player)
        if not possible_moves:
            return self.evaluate_board(board, original_player), None

        best_move = None

        if maximizing_player:
            max_score = float('-inf')
            for move in possible_moves:
                next_board = simulate_move(board, move, player)
                score, _ = self.minimax(next_board, depth - 1, get_opponent(player), False, original_player, start_time,
                                        alpha, beta)
                if score > max_score:
                    max_score = score
                    best_move = move
                alpha = max(alpha, score)
                if beta <= alpha or time.time() - start_time >= self.time_limit:
                    break
            return max_score, best_move
        else:
            min_score = float('inf')
            for move in possible_moves:
                next_board = simulate_move(board, move, player)
                score, _ = self.minimax(next_board, depth - 1, get_opponent(player), True, original_player, start_time,
                                        alpha, beta)
                if score < min_score:
                    min_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha or time.time() - start_time >= self.time_limit:
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
                        score += 1
                        if pip == 6:
                            score += 3
                    elif cell_player == opponent:
                        score -= 1
                        if pip == 6:
                            score -= 3
        return score
