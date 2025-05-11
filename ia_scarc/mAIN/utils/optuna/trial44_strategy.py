import random
import time
from typing import Optional, Tuple
from mAIN.CephalopodGame import CephalopodGame


class Trial44BestStrategyTimed:
    def __init__(self, debug: bool = False):
        self.eval_cache = {}
        self.minimax_cache = {}
        self.cache_hits = 0
        self.debug = debug
        self.weights = {
            'piece_weight': 1.6348501381211626,
            'bonus_six_weight': 2.590829894482983,
            'opponent_piece_weight': 1.8317548471057277,
            'opponent_six_weight': 3.154659219131491,
            'risk_factor': 0.482911642608139,
            'dominance_weight': 1.5121202509060507,
            'safe_placement_weight': 0.7019624166860627,
            'opponent_threat_weight': 0.19929086002522411
        }

    def choose_move(self, game, state, player: str) -> Optional[Tuple]:
        start = time.time()
        legal_moves = game.actions(state)
        if not legal_moves:
            return None
        if state.last_move is None:
            return random.choice(legal_moves)

        self.eval_cache.clear()
        self.minimax_cache.clear()
        self.cache_hits = 0

        best_move = None
        best_score = float('-inf')
        depth = 1
        last_completed_depth = 0

        while time.time() - start < 3.0:
            current_best = None
            current_best_score = float('-inf')

            for move in legal_moves:
                new_state = game.result(state, move)
                score = self.minimax(game, new_state, depth - 1, False, player)

                # Penalità per mosse rischiose
                if self.is_move_risky_for_self(game, state, move, player):
                    score -= self.weights["risk_factor"] * 5

                # Bonus per minacce future
                if self.is_move_dangerous_for_opponent(game, state, move, player):
                    score += self.weights["opponent_threat_weight"] * 2

                if score > current_best_score:
                    current_best_score = score
                    current_best = move

            if current_best is not None:
                best_move = current_best
                best_score = current_best_score
                last_completed_depth = depth

            depth += 1

        print(f"[Trial44BestStrategyTimed] Mossa scelta: {best_move} alla profondità {last_completed_depth}")
        print(f"[CACHE HITS] {self.cache_hits}")
        return best_move

    def minimax(self, game, state, depth: int, maximizing_player: bool, player: str) -> float:
        alpha = float('-inf')
        beta = float('inf')

        def ab_minimax(state, depth, maximizing, alpha, beta):
            key = (self.serialize_board(state.board), depth, maximizing, player)
            if key in self.minimax_cache:
                self.cache_hits += 1
                return self.minimax_cache[key]

            if depth == 0 or game.is_terminal(state):
                result = self.evaluate_board(state, player)
                self.minimax_cache[key] = result
                return result

            legal_moves = game.actions(state)
            if not legal_moves:
                result = self.evaluate_board(state, player)
                self.minimax_cache[key] = result
                return result

            legal_moves.sort(key=lambda move: self.heuristic_move_score(game, state, move, player), reverse=True)

            if maximizing:
                max_eval = float('-inf')
                for move in legal_moves:
                    result_state = game.result(state, move)
                    eval = ab_minimax(result_state, depth - 1, False, alpha, beta)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                self.minimax_cache[key] = max_eval
                return max_eval
            else:
                min_eval = float('inf')
                for move in legal_moves:
                    result_state = game.result(state, move)
                    eval = ab_minimax(result_state, depth - 1, True, alpha, beta)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                self.minimax_cache[key] = min_eval
                return min_eval

        return ab_minimax(state, depth, maximizing_player, alpha, beta)

    def evaluate_board(self, state, player: str) -> float:
        key = (self.serialize_board(state.board), player)
        if key in self.eval_cache:
            return self.eval_cache[key]

        opponent = "Red" if player == "Blue" else "Blue"
        turn = sum(1 for row in state.board for cell in row if cell is not None)
        player_score = 0
        opponent_score = 0
        safe_score = 0
        threat_score = 0

        for row in state.board:
            for cell in row:
                if cell:
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
        dominance_score = (my_pieces - opp_pieces) * self.weights["dominance_weight"]

        for r, row in enumerate(state.board):
            for c, cell in enumerate(row):
                if cell and cell[0] == player and not self.is_piece_vulnerable(state, r, c, player):
                    safe_score += self.weights["safe_placement_weight"]

        game = CephalopodGame(size=len(state.board))
        for move in game.actions(state):
            simulated = game.result(state, move)
            captured = self.count_captured_pieces(state, simulated, player)
            if captured >= 1:
                threat_score += self.weights["opponent_threat_weight"]

        if turn < 10:
            player_score *= 1.1
            safe_score *= 1.2
        else:
            dominance_score *= 1.2
            threat_score *= 1.5

        result = player_score - opponent_score + dominance_score + safe_score - threat_score
        self.eval_cache[key] = result
        return result

    def is_move_risky_for_self(self, game, state, move, player):
        new_state = game.result(state, move)
        for opp_move in game.actions(new_state):
            simulated = game.result(new_state, opp_move)
            if self.count_captured_pieces(state, simulated, player) >= 3:
                return True
        return False

    def is_move_dangerous_for_opponent(self, game, state, move, player):
        opponent = "Red" if player == "Blue" else "Blue"
        new_state = game.result(state, move)
        for my_next_move in game.actions(new_state):
            simulated = game.result(new_state, my_next_move)
            if self.count_captured_pieces(new_state, simulated, opponent) >= 2:
                return True
        return False

    def count_captured_pieces(self, before_state, after_state, player):
        before = sum(1 for row in before_state.board for cell in row if cell and cell[0] == player)
        after = sum(1 for row in after_state.board for cell in row if cell and cell[0] == player)
        return before - after

    def is_piece_vulnerable(self, state, row, col, player):
        opponent = "Red" if player == "Blue" else "Blue"
        simulated_state = state.copy()
        simulated_state.board[row][col] = None
        simulated_state.to_move = opponent

        game = CephalopodGame(size=state.size)
        for move in game.actions(simulated_state):
            result_state = game.result(simulated_state, move)
            if result_state.board[row][col] is None:
                return True
        return False

    def serialize_board(self, board):
        return tuple(tuple((cell[0], cell[1]) if cell else None for cell in row) for row in board)
