import time
from mAIN.utils.strategy_utils import get_opponent, get_all_legal_moves, simulate_move


class CephalopodEnhancedStrategy:
    """
    Minimax strategy per Cephalopod con:
    - catture forzate
    - iterative deepening + alpha-beta pruning
    - euristica basata su delta di pezzi catturati (inclusi 6)
    - controllo timeout fine-grana e safety margin
    """
    def __init__(self, time_limit=3.0, safety_ratio=0.07, check_interval=300):
        self.time_limit = time_limit
        self.safety_margin = time_limit * safety_ratio
        self.check_interval = check_interval
        self.nodes = 0

    def choose_move(self, game, state, player):
        start = time.time()
        deadline = start + self.time_limit - self.safety_margin
        best_move = None
        depth = 1
        self.nodes = 0

        # Iterative deepening
        while time.time() < deadline:
            score, move = self._minimax(
                board=state,
                depth=depth,
                player=player,
                maximizing=True,
                alpha=-float('inf'),
                beta=float('inf'),
                start_time=start,
                deadline=deadline,
            )
            if time.time() >= deadline:
                break
            if move is not None:
                best_move = move
            depth += 1

        elapsed = time.time() - start
        print(f"[CEPHALOPOD ENHANCED] Move: {best_move} | depth {depth-1} | "
              f"nodes {self.nodes} | time {elapsed:.2f}s")
        return best_move

    def _minimax(self, board, depth, player, maximizing,
                 alpha, beta, start_time, deadline):
        # nodo processato, occasional timeout check
        self.nodes += 1
        if self.nodes % self.check_interval == 0 and time.time() >= deadline:
            return self.evaluate(board, player), None

        # terminale o depth limit
        if depth == 0 or board.is_full() or time.time() >= deadline:
            return self.evaluate(board, player), None

        # generazione mosse e forced capture
        moves = get_all_legal_moves(board, player)
        capture_moves = [m for m in moves if m[2]]
        if capture_moves:
            moves = capture_moves

        # move ordering tramite euristica delta-capture
        moves.sort(
            key=lambda m: self.move_heuristic(board, m, player),
            reverse=True
        )

        best_move = None
        if maximizing:
            value = -float('inf')
            for m in moves:
                if time.time() >= deadline:
                    break
                next_board = simulate_move(board, m, player)
                score, _ = self._minimax(
                    board=next_board,
                    depth=depth-1,
                    player=get_opponent(player),
                    maximizing=False,
                    alpha=alpha,
                    beta=beta,
                    start_time=start_time,
                    deadline=deadline,
                )
                if score > value:
                    value, best_move = score, m
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value, best_move

        else:
            value = float('inf')
            for m in moves:
                if time.time() >= deadline:
                    break
                next_board = simulate_move(board, m, player)
                score, _ = self._minimax(
                    board=next_board,
                    depth=depth-1,
                    player=get_opponent(player),
                    maximizing=True,
                    alpha=alpha,
                    beta=beta,
                    start_time=start_time,
                    deadline=deadline,
                )
                if score < value:
                    value, best_move = score, m
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value, best_move

    def evaluate(self, board, player):
        opp = get_opponent(player)
        score = 0
        for r in range(board.size):
            for c in range(board.size):
                cell = board.board[r][c]
                if cell:
                    p, pip = cell
                    delta = 1 + (8 if pip == 6 else 0)
                    score += delta if p == player else -delta
        return score

    def move_heuristic(self, board, move, player):
        """
        Calcola un punteggio euristico che:
        - penalizza fortemente le mosse che lasciano il nemico in grado di giocare 6 in posizione vantaggiosa
        - premia le catture possibili (numero di pezzi rimossi) esclusi i 6, che non possono essere rimossi
        - massima priorità se piazzo un nostro 6 (impedisco futura mossa del nemico con 6)
        - include un bonus moderato del pip giocato per differenziare mosse equivalenti
        """
        pos, pip, captured = move
        # 1) priorità assoluta ai nostri 6
        if pip == 6:
            return float('inf')

        # 2) valore base: catture immediate (escludendo i 6)
        cap_count = len([c for c in captured if board.board[c[0]][c[1]][1] != 6])
        h = cap_count * 100

        # 3) simulo la mossa
        next_board = simulate_move(board, move, player)
        opp = get_opponent(player)
        opp_moves = get_all_legal_moves(next_board, opp)

        # 4) penalità se l'avversario può giocare un 6 in qualsiasi mossa legale
        can_opp_play_six = any(m[1] == 6 for m in opp_moves)
        if can_opp_play_six:
            # penalizzo molto: evito di lasciare spazi facili per 6
            h -= 1000

        # 5) piccolo bonus per il valore del pip
        h += pip

        return h

    def _count_pieces(self, board, player):
        opp = get_opponent(player)
        opp_count = own_count = 0
        for r in range(board.size):
            for c in range(board.size):
                cell = board.board[r][c]
                if cell:
                    if cell[0] == player:
                        own_count += 1
                    elif cell[0] == opp:
                        opp_count += 1
        return opp_count, own_count
