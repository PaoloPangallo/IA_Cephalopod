import time
import random
import hashlib
from mAIN.utils.strategy_utils import get_opponent, get_all_legal_moves, simulate_move


class DynamicIterativeMinimaxStrategy2:
    def __init__(self, max_time=3.0):
        self.max_time = max_time
        self.memo = {}

    def choose_move(self, game, state, player):
        start_time = time.time()
        legal_moves = get_all_legal_moves(state, player)

        if len(legal_moves) == 1:
            print("[DYNAMIC MINIMAX] Una sola mossa possibile, scelta immediata.")
            return legal_moves[0]

        board = state.board
        total_cells = len(board) * len(board[0])
        occupied = sum(1 for row in board for cell in row if cell)
        occupancy_ratio = occupied / total_cells
        max_depth_cap = 20 if occupancy_ratio > 0.8 else float("inf")

        best_move = None
        best_score = float("-inf")
        depth = 1

        while True:
            if time.time() - start_time >= self.max_time:
                break
            if depth > max_depth_cap:
                break

            move, score = self.minimax_root(game, state, depth, player, start_time)
            if move is not None:
                best_move = move
                best_score = score

            print(f"[DYNAMIC MINIMAX] ✅ Profondità {depth} testata in {(time.time() - start_time):.2f}s")
            depth += 1

        total_time = time.time() - start_time
        print(f"[DYNAMIC MINIMAX] Scelta mossa: {best_move} alla profondità {depth - 1} in {total_time:.2f}s")
        return best_move

    def minimax_root(self, game, state, depth, player, start_time):
        best_score = float("-inf")
        best_moves = []
        for move in get_all_legal_moves(state, player):
            if time.time() - start_time >= self.max_time:
                break
            new_state = simulate_move(state, move, player)
            score = self.minimax(game, new_state, depth - 1, False, player, start_time)
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
        return (random.choice(best_moves), best_score) if best_moves else (None, best_score)

    def minimax(self, game, state, depth, maximizing_player, player, start_time, alpha=float("-inf"),
                beta=float("inf")):
        if time.time() - start_time >= self.max_time:
            return 0

        key = self.hash_state(state)
        if key in self.memo:
            return self.memo[key]

        if depth == 0 or game.is_terminal(state):
            score = self.evaluate_board(state, player)
            self.memo[key] = score
            return score

        current_player = player if maximizing_player else get_opponent(player)
        legal_moves = get_all_legal_moves(state, current_player)
        if not legal_moves:
            return self.evaluate_board(state, player)

        if maximizing_player:
            value = float("-inf")
            for move in legal_moves:
                new_state = simulate_move(state, move, current_player)
                value = max(value, self.minimax(game, new_state, depth - 1, False, player, start_time, alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:
                    print(f"[PRUNING] 🔪 Pruning alpha a depth={depth}")
                    break
        else:
            value = float("inf")
            for move in legal_moves:
                new_state = simulate_move(state, move, current_player)
                value = min(value, self.minimax(game, new_state, depth - 1, True, player, start_time, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    print(f"[PRUNING] 🔪 Pruning beta a depth={depth}")
                    break

        self.memo[key] = value
        return value

    def hash_state(self, state):
        board_str = ''.join(
            [''.join([f"{cell[0][0]}{cell[1]}" if cell else '.' for cell in row]) for row in state.board])
        return hashlib.md5((board_str + state.to_move).encode()).hexdigest()

    def evaluate_board(self, board, player):
        opponent = get_opponent(player)
        score = 0
        for r in range(board.size):
            for c in range(board.size):
                cell = board.board[r][c]
                if cell is not None:
                    cell_player, pip = cell
                    if cell_player == player:
                        score += 1 + (3 if pip == 6 else 0)
                    elif cell_player == opponent:
                        score -= 1 + (3 if pip == 6 else 0)
        return score
