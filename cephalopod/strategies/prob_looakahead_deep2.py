import copy
from cephalopod.core.mechanics import find_capturing_subsets
from cephalopod.core.board import Die


class ProbabilisticLookaheadStrategy:
    def __init__(self, config=None, debug=False):
        self.name = "ProbabilisticLookahead"
        self.debug = debug
        self.config = config or {
            "six_bonus": 20,
            "low_capture_opportunity_bonus": 2,
            "center_bonus_weight": 2.0,
            "exposure_penalty_weight": 0.5,
            "opponent_response_penalty_weight": 1.0
        }

    def choose_move(self, board, color):
        empty_cells = board.get_empty_cells()
        best_score = float("-inf")
        best_move = None

        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                for subset, sum_pips in capturing_options:
                    move = (r, c, sum_pips, subset)
                    score = self.evaluate_move_with_opponent_response(board, move, color)
                    if score > best_score:
                        best_score = score
                        best_move = move

            # Move without capture
            simple_move = (r, c, 1, [])
            score = self.evaluate_move_with_opponent_response(board, simple_move, color)
            if score > best_score:
                best_score = score
                best_move = simple_move

        return best_move

    def evaluate_move_with_opponent_response(self, board, move, my_color):
        board_copy = self.apply_move(copy.deepcopy(board), move, my_color)

        reward = self.get_reward(board, move)
        position_bonus = self.get_positional_bonus(move[0], move[1], board)
        exposure_penalty = self.get_exposure_penalty(board_copy, move[0], move[1])

        opponent_color = "W" if my_color == "B" else "B"
        opponent_best_gain = self.simulate_opponent_best_response(board_copy, opponent_color)
        opponent_penalty = self.config["opponent_response_penalty_weight"] * opponent_best_gain

        bonus_for_making_6 = self.config["six_bonus"] if self.check_if_move_is_six_capture(move) else 0
        bonus_if_opponent_captures_low = self.check_if_move_opens_small_capture_to_opponent(board_copy, opponent_color)

        total_score = (
            reward
            + bonus_for_making_6
            + bonus_if_opponent_captures_low
            + position_bonus
            - opponent_penalty
            - exposure_penalty
        )

        if self.debug:
            print(f"[DEBUG] Move at ({move[0]}, {move[1]}) face={move[2]}:")
            print(f"  reward={reward}")
            print(f"  six_bonus={bonus_for_making_6}")
            print(f"  opp_low_bonus={bonus_if_opponent_captures_low}")
            print(f"  position_bonus={position_bonus:.2f}")
            print(f"  opponent_penalty={opponent_penalty}")
            print(f"  exposure_penalty={exposure_penalty}")
            print(f"  TOTAL SCORE = {total_score}\n")

        return total_score

    def get_reward(self, board, move):
        _, _, _, captured = move
        return sum(board.grid[r][c].top_face for (r, c) in captured) if captured else 0

    def apply_move(self, board, move, color):
        r, c, top_face, captured = move
        for (rr, cc) in captured:
            board.grid[rr][cc] = None
        board.place_die(r, c, Die(color, top_face))
        return board

    def simulate_opponent_best_response(self, board, color):
        max_reward = 0
        for (r, c) in board.get_empty_cells():
            options = find_capturing_subsets(board, r, c)
            for subset, sum_pips in options:
                reward = sum(board.grid[rr][cc].top_face for (rr, cc) in subset)
                if reward > max_reward:
                    max_reward = reward
        return max_reward

    def get_positional_bonus(self, r, c, board):
        center_r = (board.rows - 1) / 2
        center_c = (board.cols - 1) / 2
        dist = abs(r - center_r) + abs(c - center_c)
        max_dist = center_r + center_c
        return (1 - dist / max_dist) * self.config["center_bonus_weight"]

    def get_exposure_penalty(self, board, r, c):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        exposure = 0
        for dr, dc in directions:
            rr, cc = r + dr, c + dc
            if 0 <= rr < board.rows and 0 <= cc < board.cols:
                if board.grid[rr][cc] is None:
                    exposure += 1
        return exposure * self.config["exposure_penalty_weight"]

    def check_if_move_is_six_capture(self, move):
        _, _, top_face, _ = move
        return top_face == 6

    def check_if_move_opens_small_capture_to_opponent(self, board, opponent_color):
        for (r, c) in board.get_empty_cells():
            options = find_capturing_subsets(board, r, c)
            for subset, sum_pips in options:
                if sum_pips < 6:
                    return self.config["low_capture_opportunity_bonus"]
        return 0
