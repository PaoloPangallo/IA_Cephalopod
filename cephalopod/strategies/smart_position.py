
import random
import copy

from cephalopod.core.board import Die
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset


def opponent_can_capture_six(board, opponent_color):
    for (r, c) in board.get_empty_cells():
        options = find_capturing_subsets(board, r, c)
        if options:
            subset, sum_pips = choose_capturing_subset(options)
            if sum_pips == 6:
                return True
    return False


def is_candidate_position_dangerous(board, move, my_color):
    temp = copy.deepcopy(board)
    r, c, top_face, captured = move
    for (rr, cc) in captured:
        temp.grid[rr][cc] = None
    temp.place_die(r, c, Die(my_color, top_face))
    opponent = "W" if my_color == "B" else "B"
    return opponent_can_capture_six(temp, opponent)


def is_cell_safe_from_six(board, r, c, color):
    opponent = "W" if color == "B" else "B"
    neighbors = board.orthogonal_neighbors(r, c)
    for (nr, nc) in neighbors:
        neighbor = board.grid[nr][nc]
        if neighbor and neighbor.color == opponent and neighbor.top_face == 5:
            return False
    return True


class SmartPositionalLookaheadStrategy:
    def choose_move(self, board, color):
        empty_cells = board.get_empty_cells()
        fallback = None
        safe_points = []

        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                move = (r, c, sum_pips, subset)
                if sum_pips == 6 or sum_pips <= 4:
                    if not is_candidate_position_dangerous(board, move, color):
                        return move
            else:
                move = (r, c, 1, [])
                if not is_candidate_position_dangerous(board, move, color):
                    if is_cell_safe_from_six(board, r, c, color):
                        safe_points.append(move)
                    elif fallback is None:
                        fallback = move

        if safe_points:
            return random.choice(safe_points)
        return fallback or (random.choice(empty_cells) + (1, []))