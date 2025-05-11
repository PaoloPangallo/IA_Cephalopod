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


def is_candidate_position_dangerous(original_board, candidate_move, my_color):
    board_copy = copy.deepcopy(original_board)
    r, c, top_face, captured = candidate_move
    for (rr, cc) in captured:
        board_copy.grid[rr][cc] = None
    board_copy.place_die(r, c, Die(my_color, top_face))
    opponent_color = "W" if my_color == "B" else "B"
    return opponent_can_capture_six(board_copy, opponent_color)


def is_border_cell(r, c, rows, cols):
    return r == 0 or c == 0 or r == rows - 1 or c == cols - 1


class Weird3Strategy:
    def choose_move(self, board, color):
        empty_cells = board.get_empty_cells()
        fallback_move = None

        if all(cell is None for row in board.grid for cell in row):
            r, c = random.choice(empty_cells)
            return (r, c, 1, [])

        # Cattura da 6
        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                if sum_pips == 6:
                    return (r, c, 6, subset)

        # Cattura tra 2 e 4 che non lascia 6 all’avversario
        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                if 2 <= sum_pips <= 4:
                    candidate_move = (r, c, sum_pips, subset)
                    if not is_candidate_position_dangerous(board, candidate_move, color):
                        return candidate_move

        # Cattura da 5 solo se non lascia un 6 all’avversario
        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                if sum_pips == 5:
                    candidate_move = (r, c, 5, subset)
                    if not is_candidate_position_dangerous(board, candidate_move, color):
                        return candidate_move

        # Posizionamento con 1 nei bordi che non lascia cattura da 6
        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if not capturing_options and is_border_cell(r, c, len(board.grid), len(board.grid[0])):
                candidate_move = (r, c, 1, [])
                if not is_candidate_position_dangerous(board, candidate_move, color):
                    return candidate_move

        # In caso estremo, 1 a caso
        if empty_cells:
            r, c = random.choice(empty_cells)
            return (r, c, 1, [])

        return None