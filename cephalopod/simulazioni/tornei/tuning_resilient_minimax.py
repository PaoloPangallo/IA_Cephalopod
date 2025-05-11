import os
import json
import copy
import optuna
from cephalopod.core.board import Die
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.game_modes.cephalopod_game_dynamic import CephalopodGameDynamic
from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5

from cephalopod.strategies.deep_thinking import MinimaxStrategy


def get_opponent(color):
    return "W" if color == "B" else "B"


def simulate_move(board, move, player):
    board_copy = copy.deepcopy(board)
    r, c, top_face, captured = move
    for rr, cc in captured:
        board_copy.grid[rr][cc] = None
    board_copy.place_die(r, c, Die(player, top_face))
    return board_copy


def get_all_legal_moves(board):
    moves = []
    for (r, c) in board.get_empty_cells():
        capturing_options = find_capturing_subsets(board, r, c)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            moves.append((r, c, sum_pips, subset))
        else:
            moves.append((r, c, 1, []))
    return moves


def evaluate_board(board, player, opponent, weights):
    score = 0
    for r in range(board.size):
        for c in range(board.size):
            die = board.grid[r][c]
            if die:
                if die.color == player:
                    score += weights["my_piece"]
                    if die.top_face == 6:
                        score += weights["my_six"]
                    elif die.top_face == 1:
                        score += weights["my_one"]
                elif die.color == opponent:
                    score += weights["opp_piece"]
                    if die.top_face == 6:
                        score += weights["opp_six"]
                    elif die.top_face == 1:
                        score += weights["opp_one"]
    return score


def simulate_opponent_response(board, player, move, weights):
    new_board = simulate_move(board, move, player)
    opponent = get_opponent(player)
    opponent_moves = get_all_legal_moves(new_board)

    penalty = 0
    for move in opponent_moves:
        _, _, sum_pips, subset = move
        captured_count = 0
        if subset:
            if sum_pips == 6:
                penalty += weights["allow_opp_6"]
            for (rr, cc) in subset:
                die = new_board.grid[rr][cc]
                if die and die.color == player and die.top_face != 6:
                    penalty += weights["lost_my_piece"]
                    captured_count += 1
        if captured_count >= 2:
            penalty += weights.get("lost_multi_my_piece", 0)
    return penalty


def reward_capture_gain(move, board, weights, player):
    r, c, sum_pips, subset = move
    reward = 0
    captured_count = 0
    for (rr, cc) in subset:
        die = board.grid[rr][cc]
        if die and die.color == get_opponent(player) and die.top_face != 6:
            reward += weights["captured_opp_piece"]
            captured_count += 1
    if captured_count >= 2:
        reward += weights.get("captured_multi_opp", 0)
    return reward


def minimax(board, depth, player, maximizing, original_player, weights, alpha=-float("inf"), beta=float("inf")):
    if depth == 0 or board.is_full():
        return evaluate_board(board, original_player, get_opponent(original_player), weights), None

    moves = get_all_legal_moves(board)
    best_move = None

    if maximizing:
        max_score = -float("inf")
        for move in moves:
            simulated = simulate_move(board, move, player)
            score, _ = minimax(simulated, depth - 1, get_opponent(player), False, original_player, weights, alpha, beta)
            score -= simulate_opponent_response(board, player, move, weights)
            score += reward_capture_gain(move, board, weights, player)
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
            simulated = simulate_move(board, move, player)
            score, _ = minimax(simulated, depth - 1, get_opponent(player), True, original_player, weights, alpha, beta)
            score += simulate_opponent_response(board, player, move, weights)
            score -= reward_capture_gain(move, board, weights, player)
            if score < min_score:
                min_score = score
                best_move = move
            beta = min(beta, score)
            if beta <= alpha:
                break
        return min_score, best_move


class ResilientMinimaxStrategyTunable:
    def __init__(self, depth=2, weights=None):
        self.depth = depth
        self.weights = weights or {
            "my_piece": 1,
            "my_six": 3,
            "my_one": 1,
            "opp_piece": -1,
            "opp_six": -2,
            "opp_one": -1,
            "allow_opp_6": 4,
            "captured_opp_piece": 1,
            "lost_my_piece": -1,
            "captured_multi_opp": 2,
            "lost_multi_my_piece": -2
        }

    def choose_move(self, board, color):
        _, best_move = minimax(board, self.depth, color, True, color, self.weights)
        return best_move


# === SALVATAGGIO CONFIG VINCENTI ===
RESULTS_DIR = "tuned_configs"
os.makedirs(RESULTS_DIR, exist_ok=True)
BEST_CONFIGS_PATH = os.path.join(RESULTS_DIR, "best_configs.json")
winning_configs = []


def save_winning_config(params, win_rate, threshold=0.7):
    if win_rate >= threshold:
        config_entry = {
            "win_rate": win_rate,
            "params": params
        }
        winning_configs.append(config_entry)
        with open(BEST_CONFIGS_PATH, "w") as f:
            json.dump(winning_configs, f, indent=2)
        print(f"✅ Config salvata con win rate {win_rate:.2f}")


# === FUNZIONE DI OBIETTIVO PER OPTUNA ===
def objective(trial):
    weights = {
        "my_piece": trial.suggest_float("my_piece", 0.5, 2.0),
        "my_six": trial.suggest_float("my_six", 1.0, 5.0),
        "my_one": trial.suggest_float("my_one", 0.5, 3.0),
        "opp_piece": trial.suggest_float("opp_piece", -2.0, -0.5),
        "opp_six": trial.suggest_float("opp_six", -4.0, -1.0),
        "opp_one": trial.suggest_float("opp_one", -3.0, -0.5),
        "allow_opp_6": trial.suggest_float("allow_opp_6", 2.0, 6.0),
        "captured_opp_piece": trial.suggest_float("captured_opp_piece", 0.5, 3.0),
        "lost_my_piece": trial.suggest_float("lost_my_piece", -3.0, -0.5),
        "captured_multi_opp": trial.suggest_float("captured_multi_opp", 0.5, 5.0),
        "lost_multi_my_piece": trial.suggest_float("lost_multi_my_piece", -5.0, -0.5),
    }

    wins = 0
    num_games_per_opponent = 5

    opponents = [
        SmartLookaheadStrategy5(),
        MinimaxStrategy()
    ]

    for opponent in opponents:
        for _ in range(num_games_per_opponent):
            player1 = ResilientMinimaxStrategyTunable(depth=3, weights=weights)
            player2 = opponent

            game = CephalopodGameDynamic(player1, player2)
            log = game.simulate_game()

            winner = next((m for m in log if m["player"] == "WINNER"), None)
            if winner and winner["captured"] == "B":
                wins += 1

    total_games = num_games_per_opponent * len(opponents)
    win_rate = wins / total_games
    print(f"🎯 Trial {trial.number} - Win Rate: {win_rate:.2f}")

    save_winning_config(weights, win_rate)
    return win_rate


# === AVVIO OTTIMIZZAZIONE OPTUNA ===
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("📊 Best weights trovati:")
    for key, value in study.best_params.items():
        print(f"{key}: {value:.2f}")
