import copy
import random
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.core.board import Die


def opponent_can_capture_six(board, opponent_color):
    for (r, c) in board.get_empty_cells():
        capturing_options = find_capturing_subsets(board, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            if sum_pips == 6:
                return True
    return False


def is_candidate_position_dangerous_deep(original_board, candidate_move, my_color):
    board_copy = copy.deepcopy(original_board)
    r, c, top_face, captured = candidate_move

    for (rr, cc) in captured:
        board_copy.grid[rr][cc] = None
    board_copy.place_die(r, c, Die(my_color, top_face))

    opponent_color = "W" if my_color == "B" else "B"

    for (or_, oc) in board_copy.get_empty_cells():
        capturing_options = find_capturing_subsets(board_copy, or_, oc)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            opp_move = (or_, oc, sum_pips, subset)
        else:
            opp_move = (or_, oc, 1, [])

        reply_board = copy.deepcopy(board_copy)
        for (rr, cc) in opp_move[3]:
            reply_board.grid[rr][cc] = None
        reply_board.place_die(opp_move[0], opp_move[1], Die(opponent_color, opp_move[2]))

        all_responses_lead_to_six = True
        for (mr, mc) in reply_board.get_empty_cells():
            sim_board = copy.deepcopy(reply_board)
            sim_board.place_die(mr, mc, Die(my_color, 1))
            if not opponent_can_capture_six(sim_board, opponent_color):
                all_responses_lead_to_six = False
                break

        if all_responses_lead_to_six:
            return True

    return False


class SmartLookaheadStrategy6:
    def __init__(self, weight_capture_sum=1.0, weight_captured_dice=10.0):
        self.weight_capture_sum = weight_capture_sum
        self.weight_captured_dice = weight_captured_dice

    def evaluate_move(self, move, board, color):
        r, c, top_face, captured = move
        score = 0
        if top_face == 1:
            score += 1  # bonus minimo per il posizionamento
        else:
            score -= self.weight_capture_sum * top_face
            score += self.weight_captured_dice * len(captured)
        return score

    def choose_move(self, board, color):
        empty_cells = board.get_empty_cells()
        fallback_move = None

        if all(cell is None for row in board.grid for cell in row):
            r, c = random.choice(empty_cells)
            return (r, c, 1, [])

        use_deep_check = len(empty_cells) <= 8
        check_fn = is_candidate_position_dangerous_deep if use_deep_check else is_candidate_position_dangerous

        safe_moves = []

        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                candidate_move = (r, c, sum_pips, subset)
                if not check_fn(board, candidate_move, color):
                    safe_moves.append(candidate_move)

        for (r, c) in empty_cells:
            candidate_move = (r, c, 1, [])
            if not check_fn(board, candidate_move, color):
                safe_moves.append(candidate_move)
            if fallback_move is None:
                fallback_move = candidate_move

        if safe_moves:
            best_move = max(safe_moves, key=lambda m: self.evaluate_move(m, board, color))
            return best_move

        if fallback_move is None and empty_cells:
            cell = random.choice(empty_cells)
            fallback_move = (cell[0], cell[1], 1, [])

        return fallback_move


def is_candidate_position_dangerous(original_board, candidate_move, my_color):
    board_copy = copy.deepcopy(original_board)
    r, c, top_face, captured = candidate_move
    for (rr, cc) in captured:
        board_copy.grid[rr][cc] = None
    board_copy.place_die(r, c, Die(my_color, top_face))
    opponent_color = "W" if my_color == "B" else "B"
    return opponent_can_capture_six(board_copy, opponent_color)
