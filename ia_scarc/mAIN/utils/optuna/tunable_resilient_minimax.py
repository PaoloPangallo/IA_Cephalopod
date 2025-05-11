import random
from typing import Optional, Dict, Tuple


class TunableResilientMinimaxStrategy:
    def __init__(self, depth: int = 3, weights: Optional[Dict[str, float]] = None):
        self.depth = depth
        # Imposta i pesi di default se non vengono forniti, usando i valori di Trial 4
        self.weights = weights or {'piece_weight': 1.1255501891183788,
                                   'bonus_six_weight': 2.2180814249277154,
                                   'opponent_piece_weight': 1.4958250096024956,
                                   'opponent_six_weight': 1.9296143693747752}

    def choose_move(self, game, state, player: str) -> Optional[Tuple]:
        legal_moves = game.actions(state)
        if not legal_moves:
            print(f"[DEBUG] No legal moves for {player}")
            return None

        # Se è la prima mossa, restituisci una mossa casuale
        if state.last_move is None:
            print("[DEBUG] First move: choosing a random legal move")
            return random.choice(legal_moves)

        best_score = float("-inf")
        best_move = None

        for move in legal_moves:
            new_state = game.result(state, move)
            score = self.minimax(game, new_state, self.depth - 1, False, player)
            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None:
            print(f"[DEBUG] Strategy failed to select a move for {player}, returning random legal move")
            return random.choice(legal_moves)

        return best_move

    def minimax(self, game, state, depth: int, maximizing_player: bool, player: str) -> float:
        if depth == 0 or game.is_terminal(state):
            return self.evaluate_board(state, player)

        legal_moves = game.actions(state)
        if not legal_moves:
            return self.evaluate_board(state, player)

        if maximizing_player:
            best_score = float("-inf")
            for move in legal_moves:
                new_state = game.result(state, move)
                score = self.minimax(game, new_state, depth - 1, False, player)
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float("inf")
            for move in legal_moves:
                new_state = game.result(state, move)
                score = self.minimax(game, new_state, depth - 1, True, player)
                best_score = min(best_score, score)
            return best_score

    def evaluate_board(self, state, player: str) -> float:
        opponent = "Red" if player == "Blue" else "Blue"
        player_score = 0
        opponent_score = 0

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

        return player_score - opponent_score
