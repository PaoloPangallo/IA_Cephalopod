import time
from mAIN.utils.strategy_utils import get_opponent, get_all_legal_moves, simulate_move

class AlphaBetaMinimaxStrategyPaoluz:
    def __init__(self, time_limit=3.0):
        self.time_limit = time_limit
        self.margin = 0.05  # margine per evitare timeout preciso

    def choose_move(self, game, state, player):
        start = time.time()
        best_move = None
        depth = 1

        # Iterative deepening con margine
        while time.time() - start < self.time_limit - self.margin:
            score, move = self.minimax(state, depth, player, True, player, start)
            if time.time() - start >= self.time_limit - self.margin:
                break
            if move is not None:
                best_move = move
            depth += 1

        elapsed = time.time() - start
        print(f"[ALPHA-BETA MINIMAX] Scelta mossa: {best_move} alla profondità {depth - 1} in {elapsed:.2f}s")
        return best_move

    def minimax(self, board, depth, player, maximizing_player, original_player, start_time,
                alpha=-float('inf'), beta=float('inf')):
        # Controllo timeout o foglia
        if time.time() - start_time > self.time_limit - self.margin or depth == 0 or board.is_full():
            return self.evaluate_board(board, original_player), None

        possible_moves = get_all_legal_moves(board, player)
        # Ordina le mosse con euristica leggera
        possible_moves.sort(key=lambda m: self.move_heuristic(m, player, board), reverse=True)

        best_move = None
        if maximizing_player:
            max_score = -float('inf')
            for move in possible_moves:
                next_board = simulate_move(board, move, player)
                score, _ = self.minimax(next_board, depth - 1, get_opponent(player), False,
                                        original_player, start_time, alpha, beta)
                if score > max_score:
                    max_score = score
                    best_move = move
                alpha = max(alpha, score)
                if beta <= alpha or time.time() - start_time > self.time_limit - self.margin:
                    break
            return max_score, best_move
        else:
            min_score = float('inf')
            for move in possible_moves:
                next_board = simulate_move(board, move, player)
                score, _ = self.minimax(next_board, depth - 1, get_opponent(player), True,
                                        original_player, start_time, alpha, beta)
                if score < min_score:
                    min_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha or time.time() - start_time > self.time_limit - self.margin:
                    break
            return min_score, best_move

    def evaluate_board(self, board, player):
        opponent = get_opponent(player)
        score = 0

        # 1) +6 se puoi catturare un 6 in un tuo prossimo turno
        for move in get_all_legal_moves(board, player):
            for (r, c) in move[2]:
                # se stai per catturare un dado da 6
                if board.board[r][c][1] == 6:
                    score += 6

        # 2) -6 se l'avversario può catturare un tuo 6 in un solo turno
        for move in get_all_legal_moves(board, opponent):
            for (r, c) in move[2]:
                cell = board.board[r][c]
                if cell and cell[0] == player and cell[1] == 6:
                    score -= 6

        # 3) +4 per ogni tuo 1 che puoi piazzare “safe” (non catturabile subito)
        for move in get_all_legal_moves(board, player):
            if move[1] == 1:
                # simula il piazzamento
                nxt = simulate_move(board, move, player)
                # controlla se in nxt esiste una mossa avversaria che cattura quel 1
                if not any(move[0] in m[2] for m in get_all_legal_moves(nxt, opponent)):
                    score += 4

        # 4) -4 per ogni 1 che l'avversario può piazzare “safe” contro di te
        for move in get_all_legal_moves(board, opponent):
            if move[1] == 1:
                nxt = simulate_move(board, move, opponent)
                if not any(move[0] in m[2] for m in get_all_legal_moves(nxt, player)):
                    score -= 4

        return score

    def move_heuristic(self, move, player, board):
        (row, col), pip, _ = move

        # priorità assoluta: fare 6
        if pip == 6:
            return 80

        # evitare di lasciare all'avversario la possibilità di fare 6
        next_board = simulate_move(board, move, player)
        if any(m[1] == 6 for m in get_all_legal_moves(next_board, get_opponent(player))):
            return -80

        # piazzamento di 1 “safe” → bonus dinamico
        if pip == 1:
            total_pieces = sum(
                1 for r in range(board.size) for c in range(board.size)
                if board.board[r][c] is not None
            )
            dynamic_bonus = min((total_pieces // 10) * 10, 60)

            next_board = simulate_move(board, move, player)
            if not any((row, col) in m[2] for m in get_all_legal_moves(next_board, get_opponent(player))):
                return pip + 20 + dynamic_bonus

        # fallback: valore del dado
        return pip
