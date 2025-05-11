import time
import random
from mAIN.utils.strategy_utils import get_opponent, get_all_legal_moves, simulate_move


class AlphaBetaMinimaxStrategyIA:
    def __init__(self, time_limit=3.0):
        self.time_limit = time_limit
        self.margin = 0.05  # Margine per evitare timeout preciso

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
        # Timeout check con margine
        if time.time() - start_time > self.time_limit - self.margin or depth == 0 or board.is_full():
            return self.evaluate_board(board, original_player), None

        possible_moves = get_all_legal_moves(board, player)
        # Mescola per randomizzare a parità di valore
        random.shuffle(possible_moves)
        # Ordinamento mosse secondo priorità
        possible_moves.sort(key=lambda m: self.move_heuristic(m, player, board), reverse=True)

        best_move = None
        if maximizing_player:
            max_score = -float('inf')
            for move in possible_moves:
                next_board = simulate_move(board, move, player)
                score, _ = self.minimax(
                    next_board, depth - 1, get_opponent(player), False,
                    original_player, start_time, alpha, beta
                )
                if score > max_score:
                    max_score, best_move = score, move
                alpha = max(alpha, score)
                if beta <= alpha or time.time() - start_time > self.time_limit - self.margin:
                    break
            return max_score, best_move
        else:
            min_score = float('inf')
            for move in possible_moves:
                next_board = simulate_move(board, move, player)
                score, _ = self.minimax(
                    next_board, depth - 1, get_opponent(player), True,
                    original_player, start_time, alpha, beta
                )
                if score < min_score:
                    min_score, best_move = score, move
                beta = min(beta, score)
                if beta <= alpha or time.time() - start_time > self.time_limit - self.margin:
                    break
            return min_score, best_move

    def evaluate_board(self, board, player):
        opponent = get_opponent(player)
        score = 0
        # 1) Punteggio base e bonus per 1 vicino a sei
        for r in range(board.size):
            for c in range(board.size):
                cell = board.board[r][c]
                if not cell:
                    continue
                owner, pip = cell
                delta = 1 + (3 if pip == 6 else 0)
                score += delta if owner == player else -delta
                if owner == player and pip == 1:
                    cnt = self.count_adjacent_sixes(board, r, c)
                    if self.is_corner(board, r, c) and cnt >= 2:
                        score += 2
                    elif self.is_edge(board, r, c) and cnt >= 3:
                        score += 2
                    elif self.is_center(board, r, c) and cnt >= 4:
                        score += 2
        # 2) Malus per sei esposto
        opp_moves = get_all_legal_moves(board, opponent)
        for r in range(board.size):
            for c in range(board.size):
                cell = board.board[r][c]
                if cell and cell[0] == player and cell[1] == 6:
                    if any((r, c) in m[2] for m in opp_moves):
                        score -= 7  # Malus: più forte del bonus sei
        return score

    def move_heuristic(self, move, player, board):
        (row, col), pip, _ = move
        # 1) Se puoi piazzare 6: priorità assoluta
        if pip == 6:
            return 1_000_000
        # 2) Se lascia il nemico fare 6: evitalo sempre
        next_board = simulate_move(board, move, player)
        opp = get_opponent(player)
        opp_moves = get_all_legal_moves(next_board, opp)
        if any(m[1] == 6 for m in opp_moves):
            return -1_000_000
        # 3) Altrimenti, se piazzi 1: bonus in base alla posizione
        if pip == 1:
            cnt = self.count_adjacent_sixes(board, row, col)
            if self.is_corner(board, row, col) and cnt >= 2:
                return pip + 10
            elif self.is_edge(board, row, col) and cnt >= 3:
                return pip + 10
            elif self.is_center(board, row, col) and cnt >= 4:
                return pip + 10
        # 4) Fallback: valore del pip
        return pip

    def count_adjacent_sixes(self, board, row, col):
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < board.size and 0 <= nc < board.size:
                cell = board.board[nr][nc]
                if cell and cell[1] == 6:
                    count += 1
        return count

    def is_corner(self, board, row, col):
        size = board.size
        return (row == 0 or row == size - 1) and (col == 0 or col == size - 1)

    def is_edge(self, board, row, col):
        size = board.size
        return ((row == 0 or row == size - 1 or col == 0 or col == size - 1)
                and not self.is_corner(board, row, col))

    def is_center(self, board, row, col):
        return not self.is_edge(board, row, col) and not self.is_corner(board, row, col)
