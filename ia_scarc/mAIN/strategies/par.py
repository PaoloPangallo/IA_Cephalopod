import time
import signal
from mAIN.utils.strategy_utils import (
    get_opponent,
    get_all_legal_moves,
    simulate_move  # ora restituisce (next_board, delta_score)
)

# Se sei su Unix, puoi scommentare queste righe per usare alarm()
# def _timeout_handler(signum, frame):
#     raise TimeoutError
# signal.signal(signal.SIGALRM, _timeout_handler)

class RobustDynamicMinimax:
    ZOBRIST_TABLE = None  # deve essere inizializzata una volta a partire da board.size

    def __init__(self, time_limit=3.0, safety_margin=0.15):
        self.time_limit = time_limit
        self.safety_margin = safety_margin  # secondi da sottrarre
        self.trans_table = {}
        self.killer_moves = {}
        self.history = {}
        self.nodes = 0

    @staticmethod
    def _init_zobrist(size, max_pip=6):
        import random
        table = {}
        for r in range(size):
            for c in range(size):
                for pip in range(1, max_pip + 1):
                    table[(r, c, pip, 0)] = random.getrandbits(64)  # player 0
                    table[(r, c, pip, 1)] = random.getrandbits(64)  # player 1
        return table

    def choose_move(self, game, state, player):
        if RobustDynamicMinimax.ZOBRIST_TABLE is None:
            RobustDynamicMinimax.ZOBRIST_TABLE = self._init_zobrist(state.size)

        start = time.time()
        deadline = start + self.time_limit - self.safety_margin
        self.nodes = 0
        self.trans_table.clear()

        best_move = None
        best_score = None

        # Se vuoi usare alarm:
        # signal.alarm(int(self.time_limit))

        depth = 1
        try:
            while True:
                if time.time() >= deadline:
                    break
                score, move = self._minimax(
                    board=state,
                    depth=depth,
                    player=player,
                    maximizing=True,
                    original_player=player,
                    alpha=-float("inf"),
                    beta=float("inf"),
                    deadline=deadline
                )
                if time.time() >= deadline:
                    break
                if move is not None:
                    best_move, best_score = move, score
                depth += 1
        except TimeoutError:
            pass
        finally:
            # signal.alarm(0)  # disattiva alarm
            elapsed = time.time() - start
            print(f"[ROBUST MINIMAX] Mossa: {best_move} – depth {depth-1} – nodi {self.nodes} – {elapsed:.2f}s")

        return best_move

    def _minimax(self, board, depth, player, maximizing, original_player,
                 alpha, beta, deadline):
        # Timeout check a grana grossa
        self.nodes += 1
        if self.nodes % 1000 == 0 and time.time() >= deadline:
            raise TimeoutError

        # Zobrist key
        key = self._compute_zobrist(board)
        tt_key = (key, depth, int(maximizing), int(alpha * 1e6), int(beta * 1e6))
        if tt_key in self.trans_table:
            return self.trans_table[tt_key]

        # Terminal or depth=0
        if depth == 0 or board.is_full() or time.time() >= deadline:
            val = self._evaluate_board(board, original_player)
            return val, None

        # Genera mosse e ordina
        moves = get_all_legal_moves(board, player)
        moves.sort(key=lambda m: (
            # killer
            (m == self.killer_moves.get(depth, None)) * 1000,
            # history
            self.history.get(m, 0),
            # euristica base
            self._move_heuristic(m, player, board)
        ), reverse=True)

        best_move = None
        if maximizing:
            value = -float("inf")
            for move in moves:
                next_b, delta = simulate_move(board, move, player)
                score, _ = self._minimax(
                    next_b, depth-1, get_opponent(player), False,
                    original_player, alpha, beta, deadline
                )
                score += delta
                if score > value:
                    value, best_move = score, move
                alpha = max(alpha, value)
                if beta <= alpha:
                    self.killer_moves[depth] = move
                    break
        else:
            value = float("inf")
            for move in moves:
                next_b, delta = simulate_move(board, move, player)
                score, _ = self._minimax(
                    next_b, depth-1, get_opponent(player), True,
                    original_player, alpha, beta, deadline
                )
                score += delta
                if score < value:
                    value, best_move = score, move
                beta = min(beta, value)
                if beta <= alpha:
                    self.killer_moves[depth] = move
                    break

        # Salva nella transposition table
        self.trans_table[tt_key] = (value, best_move)
        # History update
        if best_move:
            self.history[best_move] = self.history.get(best_move, 0) + 1

        return value, best_move

    def _compute_zobrist(self, board):
        h = 0
        for r in range(board.size):
            for c in range(board.size):
                cell = board.board[r][c]
                if cell:
                    player_idx = 0 if cell[0] == board.current_player else 1
                    pip = cell[1]
                    h ^= self.ZOBRIST_TABLE[(r, c, pip, player_idx)]
        return h

    def _evaluate_board(self, board, player):
        """ Valutazione di fallback (se depth=0 o timeout). """
        opponent = get_opponent(player)
        score = 0
        for r in range(board.size):
            for c in range(board.size):
                cell = board.board[r][c]
                if cell:
                    owner, pip = cell
                    weight = 1 + (8 if pip == 6 else 0)
                    score += weight if owner == player else -weight
        return score

    def _move_heuristic(self, move, player, board):
        _, pip, captured = move
        base = pip + (5 if pip == 6 else 0)
        return base + (len(captured) * 2 if captured else 0)
