import random

from mAIN.CephalopodGame import CephalopodGame
from mAIN.strategies.minimax_strategies import MinimaxStrategy
from mAIN.strategies.smart_lookahead5 import SmartLookaheadStrategy5
from mAIN.utils.mcts.mcts_strategy import MCTSStrategy
from mAIN.utils.optuna.trial44_strategy import Trial44BestStrategy

STRATEGIES = {
    "MCTS": MCTSStrategy(time_limit=2.5),  # Nessun rollout_policy da passare
    "SmartLookahead5": SmartLookaheadStrategy5(),
    "Minimax": MinimaxStrategy(depth=2),
    "Trial44Best": Trial44BestStrategy(depth=2)
}

def play_match(strategy_blue, strategy_red, verbose=False):
    game = CephalopodGame(size=5)
    state = game.initial
    strategies = {"Blue": strategy_blue, "Red": strategy_red}
    current_player = state.to_move

    print(f"Starting match: Blue vs Red")

    while not game.is_terminal(state):
        move = strategies[current_player].choose_move(game, state, current_player)
        if move is None:
            break  # No moves available
        state = game.result(state, move)
        current_player = "Red" if current_player == "Blue" else "Blue"

        # Log the move made
        print(f"Player {current_player} moved: {move}")

    count_blue = state.count("Blue")
    count_red = state.count("Red")
    winner = "Blue" if count_blue > count_red else "Red" if count_red > count_blue else "Draw"

    if verbose:
        print(f"Blue: {count_blue}, Red: {count_red}, Winner: {winner}")

    return winner

def run_tournament(strategy_name, games_per_opponent=10):
    mcts_strategy = STRATEGIES[strategy_name]
    results = {}

    print(f"\nRunning tournament for {strategy_name}...\n")

    for opponent_name, opponent_strategy in STRATEGIES.items():
        if opponent_name == strategy_name:
            continue

        blue_wins = 0
        red_wins = 0
        draws = 0

        print(f"Playing against {opponent_name}...")

        for _ in range(games_per_opponent // 2):
            # MCTS as Blue
            winner = play_match(mcts_strategy, opponent_strategy, verbose=False)
            if winner == "Blue":
                blue_wins += 1
            elif winner == "Red":
                red_wins += 1
            else:
                draws += 1

            # MCTS as Red
            winner = play_match(opponent_strategy, mcts_strategy, verbose=False)
            if winner == "Red":
                red_wins += 1
            elif winner == "Blue":
                blue_wins += 1
            else:
                draws += 1

        results[opponent_name] = {
            "Wins": blue_wins + red_wins,
            "Losses": games_per_opponent - blue_wins - red_wins - draws,
            "Draws": draws,
            "WinRate": (blue_wins + red_wins) / games_per_opponent
        }

    return results

if __name__ == "__main__":
    print("Testing MCTS Strategy against others...\n")
    summary = run_tournament("MCTS", games_per_opponent=10)
    for opponent, res in summary.items():
        print(f"vs {opponent}: {res['Wins']}W / {res['Losses']}L / {res['Draws']}D | WinRate: {res['WinRate']:.2%}")
