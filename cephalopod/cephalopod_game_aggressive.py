# cephalopod_game_aggressive.py

from .cephalopod_game_naive import (
    Board,
    Die,
    find_capturing_subsets,
    choose_capturing_subset
)

def choose_aggressive_move(board, color):
    """
    Strategia che massimizza immediatamente le catture.
    Per ogni cella vuota:
      - Trova se ci sono subset catturabili (>=2 dadi adiacenti con sum_pips <= 6)
      - Se c'è almeno un subset catturabile, sceglie quello più grande
    Alla fine sceglie la cella che permette di catturare più dadi.
    In caso di parità, sceglie la prima che incontra (o potresti randomizzare).
    Se non trova alcuna cattura, piazza un dado con top_face=1 in una cella "fallback".
    """

    empty_cells = board.get_empty_cells()
    if not empty_cells:
        return None  # Non ci sono mosse possibili

    best_move = None
    max_captured = -1
    fallback_move = None  # Se non troviamo catture, piazziamo qui

    for (r, c) in empty_cells:
        capturing_options = find_capturing_subsets(board, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            capture_size = len(subset)
            if capture_size > max_captured:
                max_captured = capture_size
                best_move = (r, c, sum_pips, subset)
        else:
            # Nessuna cattura in questa cella
            if fallback_move is None:
                fallback_move = (r, c, 1, [])

    if best_move is not None:
        return best_move
    return fallback_move
