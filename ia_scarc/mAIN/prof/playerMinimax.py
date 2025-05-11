import itertools
import time


# === UTILITY ===

def get_opponent(player):
    return "Red" if player == "Blue" else "Blue"


def get_all_legal_moves(state, player):
    moves = []
    for r in range(state.size):
        for c in range(state.size):
            if state.board[r][c] is None:
                adjacent = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < state.size and 0 <= nc < state.size:
                        if state.board[nr][nc] is not None:
                            adjacent.append(((nr, nc), state.board[nr][nc][1]))
                capture_moves = []
                if len(adjacent) >= 2:
                    for rset in range(2, len(adjacent) + 1):
                        for comb in itertools.combinations(adjacent, rset):
                            s = sum(pip for _, pip in comb)
                            if 2 <= s <= 6:
                                positions = tuple(pos for pos, _ in comb)
                                capture_moves.append(((r, c), s, positions))
                if capture_moves:
                    moves.extend(capture_moves)
                else:
                    moves.append(((r, c), 1, ()))
    return moves


def simulate_move(state, move, player):
    new_state = state.copy()
    (r, c), pip, captured = move
    new_state.board[r][c] = (player, pip)
    for rr, cc in captured:
        new_state.board[rr][cc] = None
    new_state.last_move = ((r, c), captured)
    new_state.to_move = get_opponent(player)
    return new_state


# === STRATEGIA ===

class DynamicMinimaxStrategy:
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


# === ENTRY POINT ===

def playerStrategy(game, state):
    player = state.to_move
    strategy = DynamicMinimaxStrategy(time_limit=3.0)
    return strategy.choose_move(game, state, player)