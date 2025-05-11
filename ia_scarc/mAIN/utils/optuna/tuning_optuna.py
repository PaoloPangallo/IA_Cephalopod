import optuna
import json
import statistics
import logging
from optuna.samplers import CmaEsSampler
from statistics import harmonic_mean
from mAIN.CephalopodGame import CephalopodGame
from mAIN.strategies.minimax_strategies import MinimaxStrategy
from mAIN.strategies.smart_lookahead5 import SmartLookaheadStrategy5
from mAIN.utils.optuna.trial44_strategy import Trial44BestStrategy
from mAIN.utils.optuna.tunable2 import TunableResilient2MinimaxStrategy

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Baseline registry (usa lambda solo per strategie che vogliamo re-istanziate ad ogni run)
BASELINES = {
    "Minimax": MinimaxStrategy(depth=2),
    "SmartLookahead5": SmartLookaheadStrategy5(),
    "SelfDefault": TunableResilient2MinimaxStrategy(depth=2),
    "Trial44Best": lambda: Trial44BestStrategy(depth=2),
}


# Tournament match
def tournament_match(challenger, opponent, games_per_color=5, repetitions=1):
    results = []
    for rep in range(repetitions):
        for g in range(games_per_color):
            results.append(play_game(challenger, opponent))  # Challenger Blue
            results.append(play_game(opponent, challenger))  # Challenger Red
    return results


def play_game(blue_strategy, red_strategy):
    game = CephalopodGame(size=5)
    state = game.initial
    strategies = {"Blue": blue_strategy, "Red": red_strategy}
    current_player = state.to_move
    no_move_count = 0

    while not game.is_terminal(state):
        move = strategies[current_player].choose_move(game, state, current_player)
        if move is None:
            no_move_count += 1
            if no_move_count > 10:
                break
        else:
            state = game.result(state, move)
            no_move_count = 0
        current_player = "Red" if current_player == "Blue" else "Blue"

    result = state.count("Blue") - state.count("Red")
    return result


def score_trial(results):
    wins = sum(1 for r in results if r > 0)
    losses = sum(1 for r in results if r < 0)
    total = len(results)
    win_rate = wins / total
    avg_margin = statistics.mean(results)
    return win_rate, avg_margin, wins, losses


# Obiettivo multi-metrico
def objective(trial):
    weights = {
        "piece_weight": trial.suggest_float("piece_weight", 0.5, 2.0),
        "bonus_six_weight": trial.suggest_float("bonus_six_weight", 1.0, 4.0),
        "opponent_piece_weight": trial.suggest_float("opponent_piece_weight", 0.5, 2.0),
        "opponent_six_weight": trial.suggest_float("opponent_six_weight", 1.0, 4.0),
        "risk_factor": trial.suggest_float("risk_factor", 0.0, 1.0),
        "dominance_weight": trial.suggest_float("dominance_weight", 0.0, 2.0),
        "safe_placement_weight": trial.suggest_float("safe_placement_weight", 0.0, 2.0),
        "opponent_threat_weight": trial.suggest_float("opponent_threat_weight", 0.0, 2.0)
    }

    challenger = TunableResilient2MinimaxStrategy(depth=2, weights=weights)
    win_rates = []
    margins = []

    logger.info(f"Trial {trial.number}: Testing weights {weights}")

    for name, opponent_factory in BASELINES.items():
        opponent = opponent_factory() if callable(opponent_factory) else opponent_factory
        results = tournament_match(challenger, opponent, games_per_color=5, repetitions=1)
        win_rate, margin, wins, losses = score_trial(results)
        logger.info(f"{name}: {wins}W/{losses}L ({win_rate:.2%} win rate), Avg Margin = {margin:.2f}")
        win_rates.append(win_rate)
        margins.append(margin)

    composite_score = harmonic_mean(win_rates)
    avg_margin = statistics.mean(margins)

    logger.info(f"Trial {trial.number}: Composite Score = {composite_score:.4f}, Avg Margin = {avg_margin:.2f}")

    if composite_score >= 0.75:
        with open("supreme_configs.json", "a") as f:
            json.dump({"trial": trial.number, "weights": weights, "score": composite_score}, f)
            f.write("\n")

    trial.set_user_attr("win_rates", win_rates)
    trial.set_user_attr("margins", margins)
    return composite_score


if __name__ == "__main__":
    logger.info("Starting Optuna study")
    study = optuna.create_study(
        direction="maximize",
        study_name="cephalopod_tuning_v3",
        storage="sqlite:///cephalopod_v3.db",
        load_if_exists=True,
        sampler=CmaEsSampler(seed=42, sigma0=0.5),
        pruner=None
    )

    study.optimize(objective, n_trials=30, n_jobs=1)

    with open("best_supreme_config.json", "w") as f:
        json.dump(study.best_params, f, indent=4)

    logger.info("\n\U0001F3C1 Supreme tuning complete. Best configuration:")
    logger.info(study.best_params)
