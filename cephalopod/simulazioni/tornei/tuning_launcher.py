import itertools
from cephalopod.game_modes.cephalopod_game_dynamic import CephalopodGameDynamic
from cephalopod.strategies.deep5lookahead import SmartLookaheadStrategy6

from main import main
from strategies import NaiveStrategy


def simulate_match(strategy1, strategy2, num_games=10):
    wins = {"strategy1": 0, "strategy2": 0}

    for i in range(num_games):
        game = CephalopodGameDynamic(strategy1, strategy2) if i % 2 == 0 else CephalopodGameDynamic(strategy2, strategy1)
        winner = game.simulate_game()
        if (i % 2 == 0 and winner == "B") or (i % 2 == 1 and winner == "W"):
            wins["strategy1"] += 1
        else:
            wins["strategy2"] += 1

    return wins


def train_strategy(param_grid, num_games=10):
    results = []
    for weight_capture_sum, weight_captured_dice in itertools.product(
        param_grid['weight_capture_sum'], param_grid['weight_captured_dice']
    ):
        strategy = SmartLookaheadStrategy6(
            weight_capture_sum=weight_capture_sum,
            weight_captured_dice=weight_captured_dice
        )
        opponent = NaiveStrategy()
        result = simulate_match(strategy, opponent, num_games)

        results.append({
            "weight_capture_sum": weight_capture_sum,
            "weight_captured_dice": weight_captured_dice,
            "wins_vs_naive": result["strategy1"],
            "losses": result["strategy2"]
        })
        print(f"Testato: sum={weight_capture_sum}, captured={weight_captured_dice} → W: {result['strategy1']}, L: {result['strategy2']}")

    return sorted(results, key=lambda x: x["wins_vs_naive"], reverse=True)


if __name__ == "__main__":
    param_grid = {
        "weight_capture_sum": [0.5, 1.0, 2.0],
        "weight_captured_dice": [5.0, 10.0, 15.0]
    }
    results = train_strategy(param_grid, num_games=20)
    print("\nTop risultati:")
    for res in results[:5]:
        print(res)



if __name__ == "__main__":
    main()
