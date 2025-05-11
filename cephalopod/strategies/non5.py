import copy
import random
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.core.board import Die


def opponent_can_capture_five(board, opponent_color):
    """
    Ritorna True se l'avversario può catturare con somma esattamente 5.
    """
    for (r, c) in board.get_empty_cells():
        capturing_options = find_capturing_subsets(board, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            if sum_pips == 5:
                return True
    return False


def is_position_dangerous(board, move, my_color):
    """
    Simula una mossa e controlla se dopo di essa l'avversario può catturare con somma 6.
    """
    board_copy = copy.deepcopy(board)
    r, c, top_face, captured = move

    # Applichiamo la mossa nella board simulata
    for (rr, cc) in captured:
        board_copy.grid[rr][cc] = None
    board_copy.place_die(r, c, Die(my_color, top_face))

    opponent_color = "W" if my_color == "B" else "B"
    return opponent_can_capture_five(board_copy, opponent_color)


class CautiousLookaheadStrategy:
    def choose_move(self, board, color):
        empty_cells = board.get_empty_cells()
        opponent_color = "W" if color == "B" else "B"
        opponent_threat_detected = opponent_can_capture_five(board, opponent_color)

        safe_non_capturing = []

        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                move = (r, c, sum_pips, subset)

                if sum_pips == 6:
                    if not is_position_dangerous(board, move, color):
                        return move

                if not opponent_threat_detected:
                    # Se non c'è pericolo da 5, possiamo prendere anche catture diverse da 6 (es. 4 o 3)
                    if not is_position_dangerous(board, move, color):
                        return move
                else:
                    # Se c'è rischio di 5, evitiamo catture diverse da 6
                    continue

        # Se nessuna cattura va bene, proviamo a fare mosse sicure non capturing
        for (r, c) in empty_cells:
            move = (r, c, 1, [])
            if not is_position_dangerous(board, move, color):
                safe_non_capturing.append(move)

        if safe_non_capturing:
            return random.choice(safe_non_capturing)

        # Fallback: mossa qualsiasi
        if empty_cells:
            r, c = random.choice(empty_cells)
            return r, c, 1, []

        return None
