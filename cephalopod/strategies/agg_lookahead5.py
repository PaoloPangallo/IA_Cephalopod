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

def simulate_board_after_move(original_board, move, color):
    board_copy = copy.deepcopy(original_board)
    r, c, top_face, captured = move
    for (rr, cc) in captured:
        board_copy.grid[rr][cc] = None
    board_copy.place_die(r, c, Die(color, top_face))
    return board_copy

class WeirdStrategy:
    def choose_move(self, board, color):
        empty_cells = board.get_empty_cells()
        opponent_color = "W" if color == "B" else "B"

        if all(cell is None for row in board.grid for cell in row):
            r, c = random.choice(empty_cells)
            return (r, c, 1, [])

        safe_captures = []
        fallback_captures = []

        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                move = (r, c, sum_pips, subset)

                if sum_pips == 6:
                    return move

                simulated_board = simulate_board_after_move(board, move, color)
                if not opponent_can_capture_six(simulated_board, opponent_color):
                    if sum_pips == 5:
                        safe_captures.append(move)
                    elif sum_pips <= 4:
                        fallback_captures.append(move)

        if safe_captures:
            return random.choice(safe_captures)

        # Posizionamenti con 1 che non lasciano catture da 6 all'avversario
        for (r, c) in empty_cells:
            candidate_move = (r, c, 1, [])
            simulated_board = simulate_board_after_move(board, candidate_move, color)
            if not opponent_can_capture_six(simulated_board, opponent_color):
                return candidate_move

        if fallback_captures:
            return random.choice(fallback_captures)

        # Ultima risorsa: posizionamento con 1 casuale
        r, c = random.choice(empty_cells)
        return (r, c, 1, [])
