import random
from typing import Optional, Tuple, Dict
from mAIN.CephalopodGame import CephalopodGame  # Solo se serve esplicitamente


class TunableResilient2MinimaxStrategy:
    def __init__(self, depth: int = 2, weights: Optional[Dict[str, float]] = None):
        self.depth = depth
        self.weights = {
            "piece_weight": 1.656,
            "bonus_six_weight": 1.349,
            "opponent_piece_weight": 1.227,
            "opponent_six_weight": 2.408,
            "risk_factor": 0.177,
            "dominance_weight": 0.909,
            "safe_placement_weight": 1.937,
            "opponent_threat_weight": 1.118,
        }

    def choose_move(self, game, state, player: str) -> Optional[Tuple]:
        legal_moves = game.actions(state)
        if not legal_moves:
            print(f"[DEBUG] No legal moves for {player}")
            return None

        if state.last_move is None:
            return random.choice(legal_moves)

        best_score = float("-inf")
        best_move = None

        for move in legal_moves:
            new_state = game.result(state, move)
            score = self.minimax(game, new_state, self.depth - 1, False, player)

            # Penalizza se la mossa lascia vulnerabili ≥ 3 pezzi
            if self.is_move_risky(game, state, move, player):
                penalty = self.weights["risk_factor"] * 5
                score -= penalty

            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None:
            print(f"[DEBUG] Strategy failed to select a move for {player}, returning random legal move")
            return random.choice(legal_moves)

        return best_move

    def minimax(self, game, state, depth: int, maximizing_player: bool, player: str) -> float:
        alpha = float("-inf")
        beta = float("inf")

        def ab_minimax(state, depth, maximizing, alpha, beta):
            if depth == 0 or game.is_terminal(state):
                return self.evaluate_board(state, player)

            legal_moves = game.actions(state)
            if not legal_moves:
                return self.evaluate_board(state, player)

            if maximizing:
                max_eval = float("-inf")
                for move in legal_moves:
                    result_state = game.result(state, move)
                    eval = ab_minimax(result_state, depth - 1, False, alpha, beta)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break  # Beta cut-off
                return max_eval
            else:
                min_eval = float("inf")
                for move in legal_moves:
                    result_state = game.result(state, move)
                    eval = ab_minimax(result_state, depth - 1, True, alpha, beta)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break  # Alpha cut-off
                return min_eval

        return ab_minimax(state, depth, maximizing_player, alpha, beta)

    def evaluate_board(self, state, player: str) -> float:
        opponent = "Red" if player == "Blue" else "Blue"
        player_score = 0
        opponent_score = 0
        dominance_score = 0
        safe_score = 0
        threat_score = 0

        for row in state.board:
            for cell in row:
                if cell is not None:
                    owner, pip = cell
                    if owner == player:
                        player_score += self.weights["piece_weight"]
                        if pip == 6:
                            player_score += self.weights["bonus_six_weight"]
                    elif owner == opponent:
                        opponent_score += self.weights["opponent_piece_weight"]
                        if pip == 6:
                            opponent_score += self.weights["opponent_six_weight"]

        my_pieces = sum(1 for row in state.board for cell in row if cell and cell[0] == player)
        opp_pieces = sum(1 for row in state.board for cell in row if cell and cell[0] == opponent)
        dominance_score = (my_pieces - opp_pieces) * self.weights.get("dominance_weight", 0)

        for r in range(len(state.board)):
            for c in range(len(state.board[r])):
                cell = state.board[r][c]
                if cell and cell[0] == player:
                    if not self.is_piece_vulnerable(state, r, c, player):
                        safe_score += self.weights.get("safe_placement_weight", 0)

        game = CephalopodGame(size=len(state.board))
        opponent_moves = game.actions(state)
        for move in opponent_moves:
            simulated = game.result(state, move)
            captured = self.count_captured_pieces(state, simulated, player)
            if captured >= 1:
                threat_score += self.weights.get("opponent_threat_weight", 0)

        total_score = (
            player_score - opponent_score +
            dominance_score +
            safe_score -
            threat_score
        )
        return total_score

    def is_move_risky(self, game, state, move, player):
        opponent = "Red" if player == "Blue" else "Blue"
        new_state = game.result(state, move)
        opp_moves = game.actions(new_state)

        for opp_move in opp_moves:
            simulated = game.result(new_state, opp_move)
            captured = self.count_captured_pieces(state, simulated, player)
            if captured >= 3:
                return True
        return False

    def count_captured_pieces(self, before_state, after_state, player):
        before = sum(1 for row in before_state.board for cell in row if cell and cell[0] == player)
        after = sum(1 for row in after_state.board for cell in row if cell and cell[0] == player)
        return before - after

    def is_piece_vulnerable(self, state, row, col, player):
        opponent = "Red" if player == "Blue" else "Blue"
        original_piece = state.board[row][col]
        simulated_state = state.copy()
        simulated_state.board[row][col] = None
        simulated_state.to_move = opponent

        game = CephalopodGame(size=state.size)
        for move in game.actions(simulated_state):
            test_state = game.result(simulated_state, move)
            if test_state.board[row][col] is None:
                return True
        return False

