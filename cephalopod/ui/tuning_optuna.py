import json
import logging
from pathlib import Path

from cephalopod.core.board import Die
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.core.mechanics import get_opponent
from cephalopod.strategies import SmartLookaheadStrategy
from cephalopod.strategies.deep5lookahead import SmartLookaheadStrategy6
from cephalopod.strategies.deep_thinking import MinimaxStrategy
from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5
from cephalopod.strategies.variant2_lookahead2 import Variant2SmartLookahead
from game_modes.cephalopod_game_dynamic import CephalopodGameDynamic

STRATEGIES = {

    "SmartLookahead": SmartLookaheadStrategy,
    "Goat5": SmartLookaheadStrategy5,
    "Goat?": SmartLookaheadStrategy6,
    "v2": Variant2SmartLookahead,


}
# --- Logger setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- STRATEGIES dictionary: assicurati che sia visibile/importato dove serve ---

# --- MinimaxStrategy class ---
class MinimaxStrategy:
    def __init__(self, depth=2, weights=None):
        self.depth = depth
        self.weights = weights if weights is not None else {
            'piece': 1,
            'bonus_six': 3,
            'opponent_piece': 1,
            'opponent_six': 2
        }

    def choose_move(self, board, color):
        _, best_move = minimax(board, self.depth, color, True, color, self.weights)
        return best_move

# --- Funzioni di supporto ---
def get_all_legal_moves(board, player):
    moves = []
    for r, c in board.get_empty_cells():
        options = find_capturing_subsets(board, r, c)
        if options:
            subset, sum_pips = choose_capturing_subset(options)
            moves.append((r, c, sum_pips, subset))
        else:
            moves.append((r, c, 1, []))
    return moves

def simulate_move(board, move, player):
    import copy
    new_board = copy.deepcopy(board)
    r, c, top_face, captured = move
    for rr, cc in captured:
        new_board.grid[rr][cc] = None
    new_board.place_die(r, c, Die(player, top_face))
    return new_board

def evaluate_board(board, player, weights):
    opponent = get_opponent(player)
    score = 0
    for r in range(board.size):
        for c in range(board.size):
            die = board.grid[r][c]
            if die:
                if die.color == player:
                    score += weights['piece']
                    if die.top_face == 6:
                        score += weights['bonus_six']
                elif die.color == opponent:
                    score -= weights['opponent_piece']
                    if die.top_face == 6:
                        score -= weights['opponent_six']
    return score

def minimax(board, depth, player, maximizing, origin, weights, alpha=-float("inf"), beta=float("inf")):
    if depth == 0 or board.is_full():
        return evaluate_board(board, origin, weights), None

    moves = get_all_legal_moves(board, player)
    best_move = None

    if maximizing:
        max_score = -float("inf")
        for move in moves:
            next_board = simulate_move(board, move, player)
            score, _ = minimax(next_board, depth - 1, get_opponent(player), False, origin, weights, alpha, beta)
            if score > max_score:
                max_score = score
                best_move = move
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return max_score, best_move
    else:
        min_score = float("inf")
        for move in moves:
            next_board = simulate_move(board, move, player)
            score, _ = minimax(next_board, depth - 1, get_opponent(player), True, origin, weights, alpha, beta)
            if score < min_score:
                min_score = score
                best_move = move
            beta = min(beta, score)
            if beta <= alpha:
                break
        return min_score, best_move

# --- Carica i pesi da Optuna ---
def load_optuna_weights(filepath="logs/best_optuna_weights.json"):
    with open(filepath, "r") as f:
        return json.load(f)

# --- Simulazioni ---
def evaluate_against_strategies(optuna_weights, strategy_names, n_games=20, depth=2):
    results = {}
    for name in strategy_names:
        logger.info(f"\n▶ Inizio match contro: {name}")
        opponent_class = STRATEGIES[name]
        if callable(opponent_class):
            opponent_strategy = opponent_class()
        else:
            opponent_strategy = opponent_class()

        strategy_B = MinimaxStrategy(depth=depth, weights=optuna_weights)
        wins = 0

        for i in range(n_games):
            first_player = "B" if i % 2 == 0 else "W"
            if first_player == "B":
                game = CephalopodGameDynamic(strategy_B, opponent_strategy)
            else:
                game = CephalopodGameDynamic(opponent_strategy, strategy_B)

            winner = game.simulate_game2()
            if (first_player == "B" and winner == "B") or (first_player == "W" and winner == "W"):
                wins += 1

        win_rate = wins / n_games
        logger.info(f"✅ Win rate contro {name}: {win_rate * 100:.1f}%")
        results[name] = win_rate
    return results

# --- Main ---
# --- Main ---
def main():
    optuna_weights = load_optuna_weights()
    strategies_to_test = list(STRATEGIES.keys())  # <-- usa tutte le strategie

    results = evaluate_against_strategies(optuna_weights, strategies_to_test, n_games=20, depth=2)

    logger.info("\n🏁 Risultati finali:")
    for name, win_rate in results.items():
        logger.info(f"{name}: {win_rate * 100:.1f}%")

    # (Opzionale) salva in JSON
    Path("logs").mkdir(exist_ok=True)
    with open("logs/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Risultati salvati in logs/evaluation_results.json")

if __name__ == "__main__":
    main()


