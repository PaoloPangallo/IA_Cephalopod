import copy
import random

from cephalopod.core.board import Die
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset


def evaluate_board(board, player):
    opponent = "W" if player == "B" else "B"
    score = 0
    for r in range(board.size):
        for c in range(board.size):
            die = board.grid[r][c]
            if die is not None:
                if die.color == player:
                    score += 1
                    if die.top_face == 6:
                        score += 3
                elif die.color == opponent:
                    score -= 1
                    if die.top_face == 6:
                        score -= 2
    return score


def opponent_can_capture_six(board, opponent_color):
    for (r, c) in board.get_empty_cells():
        options = find_capturing_subsets(board, r, c)
        if options:
            subset, sum_pips = choose_capturing_subset(options)
            if sum_pips == 6:
                return True
    return False


def is_dangerous_move(original_board, move, player):
    temp_board = copy.deepcopy(original_board)
    r, c, top_face, captured = move
    for (rr, cc) in captured:
        temp_board.grid[rr][cc] = None
    temp_board.place_die(r, c, Die(player, top_face))
    return opponent_can_capture_six(temp_board, "W" if player == "B" else "B")


class HybridSmartStrategy:
    def choose_move(self, board, color):
        empty_cells = board.get_empty_cells()
        best_score = -float("inf")
        best_move = None
        fallback = None

        for (r, c) in empty_cells:
            options = find_capturing_subsets(board, r, c)
            if options:
                subset, sum_pips = choose_capturing_subset(options)
                move = (r, c, sum_pips, subset)
                if sum_pips == 6 or sum_pips <= 4:
                    if not is_dangerous_move(board, move, color):
                        simulated = copy.deepcopy(board)
                        for (rr, cc) in subset:
                            simulated.grid[rr][cc] = None
                        simulated.place_die(r, c, Die(color, sum_pips))
                        score = evaluate_board(simulated, color)
                        if score > best_score:
                            best_score = score
                            best_move = move
            else:
                move = (r, c, 1, [])
                if not is_dangerous_move(board, move, color):
                    simulated = copy.deepcopy(board)
                    simulated.place_die(r, c, Die(color, 1))
                    score = evaluate_board(simulated, color)
                    if score > best_score:
                        best_score = score
                        best_move = move
                elif fallback is None:
                    fallback = move

        return best_move or fallback or (random.choice(empty_cells) + (1, []))
