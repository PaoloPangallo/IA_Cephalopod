import random
from cephalopod.game_modes.cephalopod_game_dynamic import CephalopodGameDynamic
from cephalopod.simulazioni.tornei.round_robin import best_config, best_multi_config
from cephalopod.strategies import SmartLookaheadStrategy, SmartBlockAggressiveStrategy
from cephalopod.strategies.deep5lookahead import SmartLookaheadStrategy6
from cephalopod.strategies.deep_thinking import MinimaxStrategy, ExpectimaxStrategy
from cephalopod.strategies.mod_smart_lookahead import ModSmartLookaheadStrategy
from cephalopod.strategies.non5 import CautiousLookaheadStrategy
from cephalopod.strategies.prob_lookahead import ProbabilisticLookaheadStrategy
from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5
from cephalopod.strategies.smart_minimax import SmartMinimaxStrategy
from cephalopod.strategies.smart_position import SmartPositionalLookaheadStrategy
from strategies import NaiveStrategy, HeuristicStrategy, AggressiveStrategy

# === STRATEGIE REGISTRATE ===
STRATEGIES = {
    "Naive": NaiveStrategy,
    "Heuristic": HeuristicStrategy,
    "Aggressive": AggressiveStrategy,
    "SmartLookahead": SmartLookaheadStrategy,
    "Minimax (depth=3)": lambda: MinimaxStrategy(depth=3),
    "Expectimax (depth=3)": lambda: ExpectimaxStrategy(depth=3),
    "SmartMini": SmartMinimaxStrategy,
    "SmartPositional": SmartPositionalLookaheadStrategy,
    "ModifiedLookAhead": ModSmartLookaheadStrategy,
    "SmartBA": SmartBlockAggressiveStrategy,
    "SmartPos": SmartPositionalLookaheadStrategy,
    "Non5": CautiousLookaheadStrategy,
    # 🔥 Strategia probabilistica ottimizzata
    "Prob (tuned)": lambda: ProbabilisticLookaheadStrategy(config=best_config, debug=False),
    "Prob (multi)": lambda: ProbabilisticLookaheadStrategy(config=best_multi_config),
    "Goat5": SmartLookaheadStrategy5,
    "Goat?": SmartLookaheadStrategy6

}


# === SIMULAZIONE PARTITA ===
def simulate_match(strategy_B, strategy_W, max_dice_per_player=24):
    game = CephalopodGameDynamic(strategy_B, strategy_W, max_dice_per_player)
    moves_log = game.simulate_game()
    final_count = next((m["captured"] for m in moves_log if m["player"] == "END"), "B:?, W:?")
    winner_color = moves_log[-1].get("captured", None)

    if winner_color == "B":
        return strategy_B, strategy_W, final_count
    elif winner_color == "W":
        return strategy_W, strategy_B, final_count
    else:
        # Pareggio casuale
        winner = random.choice([strategy_B, strategy_W])
        loser = strategy_B if winner == strategy_W else strategy_W
        return winner, loser, final_count


# === TORNEO ROUND ROBIN CON RITORNO ===
def run_round_robin_home_away(strategy_names, max_dice_per_player=24):
    strategies = []
    for name in strategy_names:
        strat = STRATEGIES[name]
        instance = strat() if callable(strat) else strat
        instance.name = name
        strategies.append(instance)

    results = {s.name: {"points": 0, "matches": []} for s in strategies}

    print("\n=== TORNEO ANDATA E RITORNO ===")
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1 = strategies[i]
            s2 = strategies[j]

            # Andata: s1 come B, s2 come W
            winner1, loser1, score1 = simulate_match(s1, s2, max_dice_per_player)
            print(f"ANDATA: {s1.name} (B) vs {s2.name} (W) -> Vincitore: {winner1.name} | {score1}")
            results[s1.name]["points"] += 1
            results[s2.name]["points"] += 1
            if winner1.name == s1.name:
                results[s1.name]["points"] += 1
            elif winner1.name == s2.name:
                results[s2.name]["points"] += 1
            results[s1.name]["matches"].append(f"vs {s2.name} (B): {winner1.name} vince")
            results[s2.name]["matches"].append(f"vs {s1.name} (W): {winner1.name} vince")

            # Ritorno: s2 come B, s1 come W
            winner2, loser2, score2 = simulate_match(s2, s1, max_dice_per_player)
            print(f"RITORNO: {s2.name} (B) vs {s1.name} (W) -> Vincitore: {winner2.name} | {score2}")
            results[s1.name]["points"] += 1
            results[s2.name]["points"] += 1
            if winner2.name == s1.name:
                results[s1.name]["points"] += 1
            elif winner2.name == s2.name:
                results[s2.name]["points"] += 1
            results[s1.name]["matches"].append(f"vs {s2.name} (W): {winner2.name} vince")
            results[s2.name]["matches"].append(f"vs {s1.name} (B): {winner2.name} vince")

    return results


# === CLASSIFICA ===
def print_standings(results):
    print("\n=== CLASSIFICA FINALE ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["points"], reverse=True)
    for rank, (name, data) in enumerate(sorted_results, start=1):
        print(f"{rank}. {name}: {data['points']} punti")
        for match in data["matches"]:
            print(f"    - {match}")


# === MAIN ===
def main():
    strategy_names = [
        "SmartLookahead",
        "Minimax (depth=3)",
        "Expectimax (depth=3)",
        "SmartPositional",
        "SmartMini",
        "SmartBA",
        "Naive",
        "ModifiedLookAhead",
        "SmartPos",
        "Non5",
        "Prob (tuned)",
        "Prob (multi)",
        "Goat5",
        "Goat?"

    ]

    results = run_round_robin_home_away(strategy_names)
    print_standings(results)


if __name__ == '__main__':
    main()
