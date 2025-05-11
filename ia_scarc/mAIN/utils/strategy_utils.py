# strategy_utils.py
import copy


def get_opponent(player):
    return "Red" if player == "Blue" else "Blue"


def get_all_legal_moves(board, player):
    moves = []
    for r in range(board.size):
        for c in range(board.size):
            if board.board[r][c] is not None:
                continue
            adjacent = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board.size and 0 <= nc < board.size:
                    cell = board.board[nr][nc]
                    if cell is not None:
                        adjacent.append(((nr, nc), cell[1]))
            capture_moves = []
            if len(adjacent) >= 2:
                from itertools import combinations
                for r_subset in range(2, len(adjacent) + 1):
                    for subset in combinations(adjacent, r_subset):
                        pip_sum = sum(p for _, p in subset)
                        if 2 <= pip_sum <= 6:
                            capture_moves.append(((r, c), pip_sum, tuple(pos for pos, _ in subset)))
            if capture_moves:
                moves.extend(capture_moves)
            else:
                moves.append(((r, c), 1, ()))
    return moves


def simulate_move(board, move, player):
    new_board = board.copy()
    (r, c), pip, captured = move
    new_board.board[r][c] = (player, pip)
    for rr, cc in captured:
        new_board.board[rr][cc] = None
    new_board.last_move = ((r, c), captured)
    new_board.to_move = get_opponent(player)
    return new_board
