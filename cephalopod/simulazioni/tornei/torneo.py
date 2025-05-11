import random
from cephalopod.game_modes.cephalopod_game_dynamic import CephalopodGameDynamic
from cephalopod.strategies import SmartLookaheadStrategy, SmartBlockAggressiveStrategy
from cephalopod.strategies.deep_thinking import MinimaxStrategy, ExpectimaxStrategy
from cephalopod.strategies.mod_smart_lookahead import ModSmartLookaheadStrategy
from cephalopod.strategies.smart_minimax import SmartMinimaxStrategy
from cephalopod.strategies.smart_position import SmartPositionalLookaheadStrategy
from strategies import NaiveStrategy, HeuristicStrategy, AggressiveStrategy
from graphviz import Digraph

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
    "SmartBA": SmartBlockAggressiveStrategy  # ✅ aggiunta mancante

}


# === SIMULAZIONE PARTITA ===
def simulate_match(strategy_B, strategy_W, max_dice_per_player=24):
    """
    Simula una partita utilizzando CephalopodGameDynamic.
    - strategy_B gioca come "B", strategy_W come "W".
    La partita viene simulata fino a che la board è piena.
    Ritorna una tupla (winner, loser, final_score) dove final_score è una stringa ad es. "B:14,W:11".
    """
    game = CephalopodGameDynamic(strategy_B, strategy_W, max_dice_per_player)
    moves_log = game.simulate_game()

    final_count = next((m["captured"] for m in moves_log if m["player"] == "END"), "B:?, W:?")
    winner_color = moves_log[-1].get("captured", None)

    if winner_color == "B":
        return strategy_B, strategy_W, final_count
    elif winner_color == "W":
        return strategy_W, strategy_B, final_count
    else:
        winner = random.choice([strategy_B, strategy_W])
        loser = strategy_B if winner == strategy_W else strategy_W
        return winner, loser, final_count


# === SIMULAZIONE TORNEO LIVE ===
def run_tournament(strategy_names, max_dice_per_player=24):
    """
    Simula un torneo live tra le strategie indicate.
    - Istanzia e mescola le strategie.
    - Ogni round vengono simulati i match e i vincitori avanzano.
    - Viene costruita una struttura tournament_history, una lista di round in cui ogni match è rappresentato da
      (p1, p2, winner, score).

    Ritorna (tournament_history, champion, runner_up)
    """
    strategies = []
    for name in strategy_names:
        strategy_constructor = STRATEGIES[name]
        strategy_instance = strategy_constructor() if callable(strategy_constructor) else strategy_constructor
        strategy_instance.name = name
        strategies.append(strategy_instance)

    random.shuffle(strategies)
    tournament_history = []
    current_round = strategies.copy()
    round_number = 1

    # Simula round eliminatori finché non rimangono 2 finalisti
    while len(current_round) > 2:
        round_matches = []
        next_round = []
        print(f"\n=== Round {round_number} ===")
        for i in range(0, len(current_round), 2):
            if i + 1 < len(current_round):
                strat_B = current_round[i]
                strat_W = current_round[i + 1]
                winner, loser, score = simulate_match(strat_B, strat_W, max_dice_per_player)
                print(f"Match: {strat_B.name} (B) vs {strat_W.name} (W) -> Vincitore: {winner.name} | {score}")
                round_matches.append((strat_B.name, strat_W.name, winner.name, score))
                next_round.append(winner)
            else:
                print(f"{current_round[i].name} ottiene il bye.")
                next_round.append(current_round[i])
        tournament_history.append(round_matches)
        current_round = next_round
        round_number += 1

    # Finale (quando rimangono esattamente 2 strategie)
    if len(current_round) == 2:
        print("\n=== Finale ===")
        final_B = current_round[0]
        final_W = current_round[1]
        winner, loser, score = simulate_match(final_B, final_W, max_dice_per_player)
        print(f"Finale: {final_B.name} (B) vs {final_W.name} (W) -> Campione: {winner.name} | {score}")
        tournament_history.append([(final_B.name, final_W.name, winner.name, score)])
        champion = winner.name
        runner_up = loser.name
    else:
        champion = current_round[0].name
        runner_up = None
        print("Finale non disputata.")

    return tournament_history, champion, runner_up


# === VISUALIZZAZIONE BRACKET LIVE ===
def render_live_bracket(tournament_history, output_file="live_bracket"):
    """
    Disegna un bracket sportivo basato sulla struttura tournament_history.
    Assume un bracket perfetto con 8 partecipanti (3 round: 4 match, 2 match, 1 finale).

    Per ogni match, crea un nodo con l'etichetta:
       "p1 vs p2\nWinner: winner\nScore: score"
    e collega i match dei round precedenti ai match del round successivo.
    """
    dot = Digraph(comment="Torneo Live", format="png")
    dot.attr(rankdir="LR", splines="polyline")
    num_rounds = len(tournament_history)

    # Crea nodi per ogni match per ogni round
    for r, round_matches in enumerate(tournament_history):
        with dot.subgraph(name=f"cluster_round{r + 1}") as sub:
            sub.attr(rank="same", label=f"Round {r + 1}")
            for i, (p1, p2, winner, score) in enumerate(round_matches, start=1):
                node_id = f"R{r + 1}_M{i}"
                label = f"{p1} vs {p2}\nWinner: {winner}\nScore: {score}"
                sub.node(node_id, label, shape="box", style="filled", fillcolor="lightgrey")

    # Collegamenti: in un bracket perfetto con 8 partecipanti, il round r+1 ha metà match del round r.
    # Ad esempio, per r=0 (Round 1) abbiamo 4 match, per r=1 (Round 2) 2 match, per r=2 (Finale) 1 match.
    for r in range(num_rounds - 1):
        num_matches_current = len(tournament_history[r])
        num_matches_next = len(tournament_history[r + 1])
        # Assumiamo che ogni match del round successivo abbia 2 match antecedenti (in ordine)
        for j in range(num_matches_next):
            node_next = f"R{r + 2}_M{j + 1}"
            # I due match antecedenti sono: indice 2*j e 2*j+1
            node_prev1 = f"R{r + 1}_M{2 * j + 1}"
            node_prev2 = f"R{r + 1}_M{2 * j + 2}"
            dot.edge(node_prev1, node_next)
            dot.edge(node_prev2, node_next)

    dot.render(output_file, view=True)
    print(f"Torneo esportato graficamente in {output_file}.png")


# === MAIN ===
def main():
    # Lista dei nomi delle strategie partecipanti (esattamente 8 per questo esempio)
    strategy_names = [
        "SmartLookahead",
        "Minimax (depth=3)",
        "Expectimax (depth=3)",
        "SmartPositional",
        "SmartMini",
        "SmartBA",
        "Naive",
        "ModifiedLookAhead"







    ]

    tournament_history, champion, runner_up = run_tournament(strategy_names, max_dice_per_player=24)
    print("\n====================")
    print(f"Campione del torneo: {champion}")
    if runner_up:
        print(f"Secondo classificato: {runner_up}")
    print("====================")
    render_live_bracket(tournament_history, output_file="live_bracket")


if __name__ == '__main__':
    main()
