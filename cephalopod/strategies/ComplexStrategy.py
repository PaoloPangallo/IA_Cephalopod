import random
import copy
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.core.board import Die

def opponent_can_capture_six(board, opponent_color):
    for (r, c) in board.get_empty_cells():
        options = find_capturing_subsets(board, r, c)
        if options:
            _, s = choose_capturing_subset(options)
            if s == 6:
                return True
    return False

def simulate_move(board, move, player):
    board_copy = copy.deepcopy(board)
    r, c, top_face, captured = move
    for rr, cc in captured:
        board_copy.grid[rr][cc] = None
    board_copy.place_die(r, c, Die(player, top_face))
    return board_copy

def is_resilient(board, move, player):
    simulated = simulate_move(board, move, player)
    opponent = "W" if player == "B" else "B"
    return not opponent_can_capture_six(simulated, opponent)

def is_border(r, c, size):
    return r == 0 or c == 0 or r == size - 1 or c == size - 1

class HybridResilientStrategy:
    def choose_move(self, board, color):
        empty_cells = board.get_empty_cells()
        board_size = len(board.grid)
        fallback = None
        safe_ones = []

        for r, c in empty_cells:
            options = find_capturing_subsets(board, r, c)
            if options:
                subset, sum_pips = choose_capturing_subset(options)
                move = (r, c, sum_pips, subset)

                # 1. Cattura da 6
                if sum_pips == 6:
                    return move

                # 2. Cattura che impedisce il 6 avversario
                sim_board = simulate_move(board, move, color)
                if not opponent_can_capture_six(sim_board, "W" if color == "B" else "B"):
                    return move

        # 3. Posizionamento 1 non catturabile (preferisci angoli)
        for r, c in empty_cells:
            move = (r, c, 1, [])
            if is_resilient(board, move, color):
                score = 2 if (r in [0, board_size - 1] and c in [0, board_size - 1]) else 1
                safe_ones.append((score, move))
        if safe_ones:
            safe_ones.sort(reverse=True)
            return safe_ones[0][1]

        # 4. Cattura 2-5 resiliente
        for r, c in empty_cells:
            options = find_capturing_subsets(board, r, c)
            if options:
                subset, sum_pips = choose_capturing_subset(options)
                if 2 <= sum_pips <= 5:
                    move = (r, c, sum_pips, subset)
                    if is_resilient(board, move, color):
                        return move

        # 5. Cattura non sicura, ma che non lascia 6 all'avversario
        for r, c in empty_cells:
            options = find_capturing_subsets(board, r, c)
            if options:
                subset, sum_pips = choose_capturing_subset(options)
                move = (r, c, sum_pips, subset)
                if not opponent_can_capture_six(simulate_move(board, move, color), "W" if color == "B" else "B"):
                    return move

        # 6. Uno ai bordi
        for r, c in empty_cells:
            if is_border(r, c, board_size):
                return (r, c, 1, [])

        # 7. Uno a caso
        if empty_cells:
            r, c = random.choice(empty_cells)
            return (r, c, 1, [])

        return None
