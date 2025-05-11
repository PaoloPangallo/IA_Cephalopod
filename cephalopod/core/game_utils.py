import copy
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.core.board import Board


def opponent_can_capture_six(board, opponent_color):
    """
    Data una board in uno stato simulato (dopo aver applicato una mossa candidata),
    verifica se l'avversario (opponent_color) ha almeno una mossa in cui può
    catturare dadi che portano ad ottenere un top_face = 6.
    """
    # Itera su tutte le celle vuote della board simulata
    for (r, c) in board.get_empty_cells():
        capturing_options = find_capturing_subsets(board, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            if sum_pips == 6:
                # L'avversario potrebbe ottenere un dado da 6 in questa cella
                return True
    return False


def is_candidate_position_dangerous(original_board, candidate_move, my_color):
    """
    Simula la board applicando la mossa candidata (candidate_move) e
    controlla se l'avversario può ottenere una cattura con somma 6.

    candidate_move: (r, c, top_face, captured)
    my_color: il colore del giocatore corrente
    """
    # Copia profonda della board per simulare la mossa
    board_copy = copy.deepcopy(original_board)
    r, c, top_face, captured = candidate_move

    # Rimuovi i dadi catturati (se presenti)
    for (rr, cc) in captured:
        board_copy.grid[rr][cc] = None

    # Applica il nuovo dado sulla board simulata
    # (Si assume che la mossa sia non-capturing e quindi top_face sia 1,
    #  oppure il dado acquisisce top_face = somma dei dadi catturati)
    from cephalopod.core.board import Die  # importa qui se necessario
    board_copy.place_die(r, c, Die(my_color, top_face))

    opponent_color = "W" if my_color == "B" else "B"
    return opponent_can_capture_six(board_copy, opponent_color)
