import copy
import random
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.core.board import Die

def opponent_can_capture_six(board, opponent_color):
    for (r, c) in board.get_empty_cells():
        capturing_options = find_capturing_subsets(board, r, c)
        if capturing_options:
            _, sum_pips = choose_capturing_subset(capturing_options)
            if sum_pips == 6:
                return True
    return False

def is_move_dangerous(board, move, color):
    board_copy = copy.deepcopy(board)
    r, c, top_face, captured = move
    for (rr, cc) in captured:
        board_copy.grid[rr][cc] = None
    board_copy.place_die(r, c, Die(color, top_face))
    opponent_color = "W" if color == "B" else "B"
    return opponent_can_capture_six(board_copy, opponent_color)

def is_border_cell(r, c, rows=5, cols=5):
    return r == 0 or c == 0 or r == rows - 1 or c == cols - 1

class WeightedTacticalStrategy:
    def choose_move(self, board, color):
        empty_cells = board.get_empty_cells()
        fallback = None

        if all(cell is None for row in board.grid for cell in row):
            r, c = random.choice(empty_cells)
            return (r, c, 1, [])

        capture_six = []
        capture_five_safe = []
        capture_low_safe = []
        safe_ones_on_border = []

        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                candidate_move = (r, c, sum_pips, subset)

                if sum_pips == 6:
                    if not is_move_dangerous(board, candidate_move, color):
                        capture_six.append(candidate_move)
                elif sum_pips == 5:
                    if not is_move_dangerous(board, candidate_move, color):
                        capture_five_safe.append(candidate_move)
                elif 2 <= sum_pips <= 4:
                    if not is_move_dangerous(board, candidate_move, color):
                        capture_low_safe.append(candidate_move)
                continue  # non possiamo ignorare una cattura

            # non-capturing candidate
            candidate_move = (r, c, 1, [])
            if not is_move_dangerous(board, candidate_move, color):
                if is_border_cell(r, c):
                    safe_ones_on_border.append(candidate_move)
            elif fallback is None:
                fallback = candidate_move

        if capture_six:
            return random.choice(capture_six)
        if capture_five_safe:
            return random.choice(capture_five_safe)
        if safe_ones_on_border:
            return random.choice(safe_ones_on_border)
        if capture_low_safe:
            return random.choice(capture_low_safe)
        if fallback:
            return fallback
        # Ultima spiaggia: uno a caso
        r, c = random.choice(empty_cells)
        return (r, c, 1, [])
