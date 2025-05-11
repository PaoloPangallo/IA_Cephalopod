import time
from mAIN.utils.strategy_utils import get_opponent, get_all_legal_moves, simulate_move


class SuperMinimaxStrategy:
    def __init__(self, time_limit=3.0):
        self.time_limit = time_limit
        self.transposition_table = {}
        self.cache_hits = 0  # logging

    def choose_move(self, game, state, player):
        start = time.time()
        best_move = None
        depth = 1
        last_completed_depth = 0  # 👈 questa è la profondità reale
        self.transposition_table.clear()
        self.cache_hits = 0

        while True:
            if time.time() - start >= self.time_limit:
                break

            score, move = self.minimax(state, depth, player, True, player, start)
            if time.time() - start >= self.time_limit:
                break
            if move is not None:
                best_move = move
                last_completed_depth = depth  # 👈 aggiorna solo se la mossa è valida
            depth += 1

        elapsed = time.time() - start
        print(f"[SUPER MINIMAX] Scelta mossa: {best_move} alla profondità {last_completed_depth} in {elapsed:.2f}s")
        print(f"[CACHE STATS] Cache hits in questa mossa: {self.cache_hits}")
        return best_move

    def minimax(self, board, depth, player, maximizing_player, original_player, start_time,
                alpha=-float("inf"), beta=float("inf")):

        if time.time() - start_time >= self.time_limit or depth == 0 or board.is_full():
            return self.evaluate_board(board, original_player), None

        board_hash = self.hash_board(board)
        if board_hash in self.transposition_table:
            self.cache_hits += 1
            return self.transposition_table[board_hash]

        possible_moves = get_all_legal_moves(board, player)
        best_move = None

        if maximizing_player:
            max_score = -float("inf")
            for move in possible_moves:
                next_board = simulate_move(board, move, player)
                score, _ = self.minimax(next_board, depth - 1, get_opponent(player), False,
                                        original_player, start_time, alpha, beta)
                if score > max_score:
                    max_score = score
                    best_move = move
                alpha = max(alpha, score)
                if beta <= alpha or time.time() - start_time >= self.time_limit:
                    break
            if depth >= 4:
                self.transposition_table[board_hash] = (max_score, best_move)
            return max_score, best_move
        else:
            min_score = float("inf")
            for move in possible_moves:
                next_board = simulate_move(board, move, player)
                score, _ = self.minimax(next_board, depth - 1, get_opponent(player), True,
                                        original_player, start_time, alpha, beta)
                if score < min_score:
                    min_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha or time.time() - start_time >= self.time_limit:
                    break
            if depth >= 4:
                self.transposition_table[board_hash] = (min_score, best_move)
            return min_score, best_move

    def evaluate_board(self, board, player):
        opponent = get_opponent(player)
        score = 0
        for r in range(board.size):
            for c in range(board.size):
                cell = board.board[r][c]
                if cell is not None:
                    cell_player, pip = cell
                    value = 1 + (3 if pip == 6 else 0)
                    if cell_player == player:
                        score += value
                    elif cell_player == opponent:
                        score -= value
        return score

    def hash_board(self, board):
        flat = []
        for row in board.board:
            for cell in row:
                flat.append((cell[0], cell[1]) if cell else None)
        return tuple(flat)
