import copy
import random
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.core.board import Die


def opponent_can_capture_six(board, opponent_color):
    """
    Data una board, controlla se in almeno una cella vuota l’avversario (opponent_color)
    può effettuare una cattura che porta a un dado con top_face = 6.
    """
    for (r, c) in board.get_empty_cells():
        capturing_options = find_capturing_subsets(board, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            if sum_pips == 6:
                return True
    return False


def is_candidate_position_danger   ous(original_board, candidate_move, my_color):
    """
    Simula la board dopo aver applicato candidate_move e verifica se l’avversario
    può effettuare una cattura con somma 6.

    candidate_move: (r, c, top_face, captured)
    my_color: colore del giocatore che sta piazzando il dado
    """
    board_copy = copy.deepcopy(original_board)
    r, c, top_face, captured = candidate_move

    # Rimuovi i dadi catturati nella simulazione
    for (rr, cc) in captured:
        board_copy.grid[rr][cc] = None

    # Applica il nuovo dado nella copia della board
    board_copy.place_die(r, c, Die(my_color, top_face))
    opponent_color = "W" if my_color == "B" else "B"
    return opponent_can_capture_six(board_copy, opponent_color)


class SmartLookaheadStrategy5:
    def choose_move(self, board, color):
        empty_cells = board.get_empty_cells()
        fallback_move = None

        if all(cell is None for row in board.grid for cell in row):
            r, c = random.choice(empty_cells)
            return (r, c, 1, [])  # Non-capturing obbligatoria: top_face = 1

        # Valuta le mosse capturing
        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                candidate_move = (r, c, sum_pips, subset)
                # Se possiamo fare una cattura con somma 6, lo facciamo sempre
                if sum_pips == 6:
                    if not is_candidate_position_dangerous(board, candidate_move, color):
                        return candidate_move
                elif sum_pips <= 5:
                    if not is_candidate_position_dangerous(board, candidate_move, color):
                        return candidate_move
                else:
                    continue

        # Valuta le mosse non capturing (top_face = 1)
        for (r, c) in empty_cells:
            candidate_move = (r, c, 1, [])
            if not is_candidate_position_dangerous(board, candidate_move, color):
                return candidate_move
            if fallback_move is None:
                fallback_move = candidate_move

        # Se tutte le mosse sono pericolose, usa il fallback (o un movimento casuale)
        if fallback_move is None and empty_cells:
            cell = random.choice(empty_cells)
            fallback_move = (cell[0], cell[1], 1, [])

        return fallback_move
