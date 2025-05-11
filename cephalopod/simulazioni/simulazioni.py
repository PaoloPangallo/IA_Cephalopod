import os
import pandas as pd
from typing import Callable, Type
from cephalopod.game_modes.cephalopod_game_dynamic import CephalopodGameDynamic
from cephalopod.strategies import SmartLookaheadStrategy
from cephalopod.strategies.deep5lookahead import SmartLookaheadStrategy6
from cephalopod.strategies.mod_deep_5_lookahead import ModSmartLookaheadStrategy6
from cephalopod.strategies.mod_smart_lookahead import ModSmartLookaheadStrategy
from cephalopod.strategies.non5 import CautiousLookaheadStrategy
from cephalopod.strategies.smart_hybrid import HybridSmartStrategy
from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5
from cephalopod.strategies.smart_minimax import SmartMinimaxStrategy
from cephalopod.strategies.variant2_lookahead2 import Variant2SmartLookahead
from cephalopod.strategies.deep_thinking import ExpectimaxStrategy
from cephalopod.strategies.we3 import Weird3Strategy

from cephalopod.strategies.weird2 import Weird2Strategy


def simulate_match(
    strategy_A: Callable[[], object],
    strategy_B: Callable[[], object],
    name_A: str = "StrategyA",
    name_B: str = "StrategyB",
    num_games: int = 100
) -> None:
    results = []
    wins_A = 0
    wins_B = 0

    for game_num in range(1, num_games + 1):
        game = CephalopodGameDynamic(strategy_A(), strategy_B())
        log = game.simulate_game()

        moves = [entry for entry in log if entry["player"] not in ("END", "WINNER")]
        move_count = len(moves)

        final_score = next(entry for entry in log if entry["player"] == "END")["captured"]
        winner = next(entry for entry in log if entry["player"] == "WINNER")["captured"]

        if winner == "B":
            wins_A += 1
        else:
            wins_B += 1

        results.append({
            "Game": game_num,
            "Winner": winner,
            "Moves": move_count,
            "FinalScore": final_score
        })

    # Conversione in DataFrame
    df = pd.DataFrame(results)

    # Stampa riepilogo
    print(df)
    print("\n--- Summary ---")
    print(f"Total Games: {num_games}")
    print(f"{name_A} (B) Wins: {wins_A}  ({wins_A / num_games * 100:.1f}%)")
    print(f"{name_B} (W) Wins: {wins_B}  ({wins_B / num_games * 100:.1f}%)")

    # Salvataggio CSV
    os.makedirs("risultati", exist_ok=True)
    file_path = f"risultati/results_{name_A}_B_vs_{name_B}_W.csv"
    df.to_csv(file_path, index=False)
    print(f"\nResults saved to: {file_path}")


if __name__ == "__main__":
    simulate_match(
        strategy_B=lambda: SmartLookaheadStrategy5(),
        strategy_A=lambda: Weird2Strategy(),
        name_B="Mod",
        name_A="V2",
        num_games=1000
    )
