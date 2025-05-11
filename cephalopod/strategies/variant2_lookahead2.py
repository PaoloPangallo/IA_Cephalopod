import copy
import random
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.core.board import Die


def opponent_can_capture_six(board, opponent_color):
    for (r, c) in board.get_empty_cells():
        capturing_options = find_capturing_subsets(board, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            if sum_pips == 6:
                return True
    return False


def is_candidate_position_dangerous(original_board, candidate_move, my_color):
    board_copy = copy.deepcopy(original_board)
    r, c, top_face, captured = candidate_move
    for (rr, cc) in captured:
        board_copy.grid[rr][cc] = None
    board_copy.place_die(r, c, Die(my_color, top_face))
    opponent_color = "W" if my_color == "B" else "B"
    return opponent_can_capture_six(board_copy, opponent_color)


def get_adjacent_faces_sum(board, r, c):
    total = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < len(board.grid) and 0 <= cc < len(board.grid[0]):
                die = board.grid[rr][cc]
                if die is not None:
                    total += die.top_face
    return total


def is_border_cell(r, c, rows, cols):
    return r == 0 or c == 0 or r == rows - 1 or c == cols - 1


def has_six_neighbors(board, r, c):
    count = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < len(board.grid) and 0 <= cc < len(board.grid[0]):
                die = board.grid[rr][cc]
                if die is not None and die.top_face == 6:
                    count += 1
    return count >= 2


class Variant2SmartLookahead:
    def choose_move(self, board, color):
        empty_cells = board.get_empty_cells()
        fallback_move = None

        if all(cell is None for row in board.grid for cell in row):
            r, c = random.choice(empty_cells)
            return (r, c, 1, [])

        # Prima prova a catturare 6 in sicurezza
        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                if sum_pips == 6:
                    candidate_move = (r, c, sum_pips, subset)
                    if not is_candidate_position_dangerous(board, candidate_move, color):
                        return candidate_move

        # Poi catture da 5 o meno se non pericolose
        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                if sum_pips <= 5:
                    candidate_move = (r, c, sum_pips, subset)
                    if not is_candidate_position_dangerous(board, candidate_move, color):
                        return candidate_move

        # Altrimenti posizionamenti con 1 in base alla strategia
        safe_moves = []
        border_six_moves = []

        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                continue  # Non possiamo ignorare una cattura obbligatoria

            candidate_move = (r, c, 1, [])
            if not is_candidate_position_dangerous(board, candidate_move, color):
                if is_border_cell(r, c, len(board.grid), len(board.grid[0])) and has_six_neighbors(board, r, c):
                    border_six_moves.append(candidate_move)
                else:
                    total = get_adjacent_faces_sum(board, r, c)
                    safe_moves.append((total, candidate_move))
            elif fallback_move is None:
                fallback_move = candidate_move

        if border_six_moves:
            return random.choice(border_six_moves)

        if safe_moves:
            _, best = max(safe_moves, key=lambda x: x[0])
            return best

        if fallback_move is None and empty_cells:
            cell = random.choice(empty_cells)
            fallback_move = (cell[0], cell[1], 1, [])

        return fallback_move
