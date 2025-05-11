import time
import signal
from mAIN.utils.strategy_utils import (
    get_opponent,
    get_all_legal_moves,
    simulate_move  # potrebbe restituire Board o (Board, delta_score)
)

class RobustDynamicMinimaxStrategy:
    # Flag per la Transposition Table
    EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2
    # Tabella Zobrist condivisa
    ZOBRIST_TABLE = None

    def __init__(self, time_limit=3.0, safety_margin=0.15):
        self.time_limit = time_limit
        self.safety_margin = safety_margin
        self.trans_table = {}
        self.killer_moves = {}
        self.history = {}
        self.nodes = 0
        self.player_map = {}

    @staticmethod
    def _init_zobrist(size, max_pip=6):
        import random
        table = {}
        for r in range(size):
            for c in range(size):
                for pip in range(1, max_pip + 1):
                    table[(r, c, pip, 0)] = random.getrandbits(64)
                    table[(r, c, pip, 1)] = random.getrandbits(64)
        return table

    def choose_move(self, game, state, player):
        # Inizializza Zobrist alla prima chiamata
        if RobustDynamicMinimaxStrategy.ZOBRIST_TABLE is None:
            RobustDynamicMinimaxStrategy.ZOBRIST_TABLE = self._init_zobrist(state.size)
        # Mappa giocatori a 0/1 per Zobrist
        opponent = get_opponent(player)
        self.player_map = {player: 0, opponent: 1}

        start = time.time()
        deadline = start + self.time_limit - self.safety_margin
        self.nodes = 0
        self.trans_table.clear()

        best_move = None
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
                    best_move = move
                depth += 1
        except TimeoutError:
            pass
        finally:
            elapsed = time.time() - start
            print(f"[ROBUST MINIMAX] Mossa: {best_move} – depth {depth-1} – nodi {self.nodes} – {elapsed:.2f}s")

        return best_move

    def _minimax(self, board, depth, player, maximizing,
                 original_player, alpha, beta, deadline):
        # Timeout a grana grossa
        self.nodes += 1
        if self.nodes % 1000 == 0 and time.time() >= deadline:
            raise TimeoutError

        # Zobrist hash + profondità
        zkey = self._compute_zobrist(board)
        tt_key = (zkey, depth)
        if tt_key in self.trans_table:
            val, mv, flag = self.trans_table[tt_key]
            if flag == self.EXACT \
               or (flag == self.LOWERBOUND and val > alpha) \
               or (flag == self.UPPERBOUND and val < beta):
                return val, mv

        # Condizioni di terminazione
        if depth == 0 or board.is_full() or time.time() >= deadline:
            val = self._evaluate_board(board, original_player)
            return val, None

        # Genera mosse e le ordina
        moves = get_all_legal_moves(board, player)
        moves.sort(key=lambda m: (
            (m == self.killer_moves.get(depth)) * 1000,
            self.history.get(m, 0),
            self._move_heuristic(m, player, board)
        ), reverse=True)

        orig_alpha, orig_beta = alpha, beta
        best_move = None

        if maximizing:
            value = -float("inf")
            for m in moves:
                sim = simulate_move(board, m, player)
                if isinstance(sim, tuple) and len(sim) == 2:
                    nxt, delta = sim
                else:
                    nxt = sim
                    delta = (
                        self._evaluate_board(nxt, original_player)
                        - self._evaluate_board(board, original_player)
                    )
                score, _ = self._minimax(
                    nxt, depth-1, get_opponent(player),
                    False, original_player, alpha, beta, deadline
                )
                score += delta
                if score > value:
                    value, best_move = score, m
                alpha = max(alpha, value)
                if beta <= alpha:
                    self.killer_moves[depth] = m
                    break
        else:
            value = float("inf")
            for m in moves:
                sim = simulate_move(board, m, player)
                if isinstance(sim, tuple) and len(sim) == 2:
                    nxt, delta = sim
                else:
                    nxt = sim
                    delta = (
                        self._evaluate_board(nxt, original_player)
                        - self._evaluate_board(board, original_player)
                    )
                score, _ = self._minimax(
                    nxt, depth-1, get_opponent(player),
                    True, original_player, alpha, beta, deadline
                )
                score += delta
                if score < value:
                    value, best_move = score, m
                beta = min(beta, value)
                if beta <= alpha:
                    self.killer_moves[depth] = m
                    break

        # Determina il tipo di bound per la TT
        if value <= orig_alpha:
            flag = self.UPPERBOUND
        elif value >= orig_beta:
            flag = self.LOWERBOUND
        else:
            flag = self.EXACT

        # Memorizza in transposition table
        self.trans_table[tt_key] = (value, best_move, flag)
        if best_move:
            self.history[best_move] = self.history.get(best_move, 0) + 1

        return value, best_move

    def _compute_zobrist(self, board):
        h = 0
        table = RobustDynamicMinimaxStrategy.ZOBRIST_TABLE
        for r in range(board.size):
            for c in range(board.size):
                cell = board.board[r][c]
                if cell:
                    # Usa la mappa dei giocatori per l'indice
                    player_idx = self.player_map.get(cell[0], 0)
                    pip = cell[1]
                    h ^= table[(r, c, pip, player_idx)]
        return h

    def _evaluate_board(self, board, player):

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
                            score += 8
                    elif cell_player == opponent:
                        score -= 1
                        if pip == 6:
                            score -= 8
        return score

    def _move_heuristic(self, move, player, board):
        _, pip, captured = move
        base = pip + (5 if pip == 6 else 0)
        return base + (len(captured) * 2 if captured else 0)
