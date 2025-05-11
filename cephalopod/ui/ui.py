import random
from cephalopod.game_modes.cephalopod_game_dynamic import CephalopodGameDynamic
from cephalopod.strategies import SmartBlockAggressiveStrategy, SmartLookaheadStrategy
from cephalopod.strategies.deep_thinking import MinimaxStrategy, ExpectimaxStrategy
from cephalopod.strategies.mod_smart_lookahead import ModSmartLookaheadStrategy
from cephalopod.strategies.smart_minimax import SmartMinimaxStrategy
from cephalopod.strategies.smart_position import SmartPositionalLookaheadStrategy
from strategies import NaiveStrategy, HeuristicStrategy, AggressiveStrategy, Orthogonal1Strategy

# Dizionario delle strategie registrate.
STRATEGIES = {
    "Naive": NaiveStrategy,
    "Heuristic": HeuristicStrategy,
    "Aggressive": AggressiveStrategy,
    "Orthogonal1": Orthogonal1Strategy,
    "SmartBA": SmartBlockAggressiveStrategy,
    "SmartLookahead": SmartLookaheadStrategy,
    "Minimax (depth=3)": lambda: MinimaxStrategy(depth=3),
    "Expectimax (depth=3)": lambda: ExpectimaxStrategy(depth=3),
    "SmartMini": SmartMinimaxStrategy,
    "SmartPositional": SmartPositionalLookaheadStrategy,
    "ModifiedLookAhed": ModSmartLookaheadStrategy,
    "GOAT": SmartLookaheadStrategy

}


def simulate_match(strategy1, strategy2):
    """
    Simula una partita vera e propria tra due strategie:
      - strategy1 gioca come Black ("B")
      - strategy2 gioca come White ("W")

    Utilizza la classe CephalopodGameDynamic per simulare l'intera partita.
    Si assume che il log delle mosse (moves_log) termini con una mossa in cui
    il campo "winner" indichi il colore vincente ("B" o "W").

    Ritorna una tupla (winner, loser) con le istanze delle strategie.
    """
    # Istanzia il gioco con strategy1 come Black e strategy2 come White
    game = CephalopodGameDynamic(strategy1, strategy2)
    moves_log = game.simulate_game()

    # Supponiamo che l'ultimo elemento del log contenga il vincitore
    final_move = moves_log[-1]
    winner_color = final_move.get("winner", None)
    if winner_color == "B":
        return strategy1, strategy2
    elif winner_color == "W":
        return strategy2, strategy1
    else:
        # In caso di esito non chiaro, sceglie casualmente
        winner = random.choice([strategy1, strategy2])
        loser = strategy1 if winner == strategy2 else strategy2
        return winner, loser


def run_tournament(heuristic_names):
    """
    Esegue un torneo a bracket usando le strategie i cui nomi sono nella lista heuristic_names.
    Le strategie vengono istanziate usando il dizionario STRATEGIES e abbinate casualmente.

    Ritorna una lista dei round (ogni round è una lista di tuple match) e il campione finale.
    """
    # Istanzia le strategie e assegna un attributo "name" per una stampa più chiara
    strategies = []
    for name in heuristic_names:
        strategy_constructor = STRATEGIES[name]
        # Se è una lambda, la chiamiamo; altrimenti, assumiamo sia una classe e la istanziamo.
        strategy_instance = strategy_constructor() if callable(strategy_constructor) else strategy_constructor
        strategy_instance.name = name
        strategies.append(strategy_instance)

    # Shuffle casuale per il sorteggio iniziale
    random.shuffle(strategies)

    rounds = []  # Lista dei round (ogni round è una lista di match)
    current_round = strategies.copy()
    round_number = 1

    while len(current_round) > 1:
        round_matches = []
        next_round = []
        print(f"\n--- Round {round_number} ---")
        # Abbina le strategie a coppie
        for i in range(0, len(current_round), 2):
            if i + 1 < len(current_round):
                s1 = current_round[i]
                s2 = current_round[i + 1]
                # Simula una partita vera tra s1 (Black) e s2 (White)
                winner, loser = simulate_match(s1, s2)
                print(f"Match: {s1.name} (B) vs {s2.name} (W) -> Vincitore: {winner.name}")
                round_matches.append((s1, s2, winner, loser))
                next_round.append(winner)
            else:
                # Se dispari, l'ultimo ottiene un bye
                print(f"{current_round[i].name} ottiene il bye")
                next_round.append(current_round[i])
        rounds.append(round_matches)
        current_round = next_round
        round_number += 1

    champion = current_round[0]
    return rounds, champion


def simulate_third_place_and_challenge(rounds, champion):
    """
    Se il torneo ha avuto semifinale (almeno 4 partecipanti), simula:
      - la partita per il terzo e quarto posto (tra i perdenti delle semifinali)
      - una sfida speciale tra il campione e il quarto classificato.
    """
    if len(rounds) >= 2:
        semifinal_round = rounds[-2]
        if len(semifinal_round) == 2:
            semifinal_losers = [match[3] for match in semifinal_round]
            print("\n--- Match per il terzo posto ---")
            third_place_winner, fourth_place = simulate_match(semifinal_losers[0], semifinal_losers[1])
            print(f"Terzo posto: {third_place_winner.name} | Quarto posto: {fourth_place.name}")

            print("\n--- Sfida speciale: Campione vs Quarto posto ---")
            challenge_winner, _ = simulate_match(champion, fourth_place)
            print(f"Campione: {challenge_winner.name} vince la sfida speciale contro {fourth_place.name}")
        else:
            print("Le semifinali non sono state giocate in modo completo per determinare terzo e quarto posto.")
    else:
        print("Non ci sono abbastanza round per il match di terzo e quarto posto.")


def main():
    # Lista dei nomi delle strategie partecipanti (modifica a piacere)
    heuristic_names = [
        "SmartLookahead",
        "Minimax (depth=3)",
        "Expectimax (depth=3)",
        "SmartPositional",
        "SmartMini",
        "SmartBA",
        "Naive",
        "Heuristic",
        "GOAT"
    ]

    rounds, champion = run_tournament(heuristic_names)
    print(f"\nCampione del torneo: {champion.name}")
    simulate_third_place_and_challenge(rounds, champion)


if __name__ == '__main__':
    main()
