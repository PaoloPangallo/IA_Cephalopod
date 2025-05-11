import os
import random
import tkinter as tk
from tkinter import ttk, messagebox
from cephalopod.game_modes.cephalopod_game_dynamic import CephalopodGameDynamic
from cephalopod.strategies import SmartBlockAggressiveStrategy, SmartLookaheadStrategy
from cephalopod.strategies.deep_thinking import MinimaxStrategy, ExpectimaxStrategy
from cephalopod.strategies.mod_smart_lookahead import ModSmartLookaheadStrategy
from cephalopod.strategies.smart_minimax import SmartMinimaxStrategy
from cephalopod.strategies.smart_position import SmartPositionalLookaheadStrategy
from strategies import NaiveStrategy, HeuristicStrategy, AggressiveStrategy, Orthogonal1Strategy
from graphviz import Digraph

# === STRATEGIE REGISTRATE ===
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
    "ModifiedLookAhed": ModSmartLookaheadStrategy
}


# === SIMULAZIONE PARTITA ===
def simulate_match(strategy_B, strategy_W, max_dice_per_player=24):
    """
    Simula una partita utilizzando CephalopodGameDynamic.
    - strategy_B gioca come "B", strategy_W come "W".
    La partita viene simulata fino a che la board è piena.
    Ritorna (winner, loser, final_score) dove final_score è una stringa es. "B:14,W:11".
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
    Simula un torneo live:
      - Istanzia e mescola le strategie.
      - Ogni round vengono simulati i match e i vincitori avanzano.
      - Costruisce una struttura tournament_history (lista di round)
        in cui ogni match è rappresentato da (p1, p2, winner, score).

    Ritorna (tournament_history, champion, runner_up).
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

    while len(current_round) > 2:
        round_matches = []
        next_round = []
        for i in range(0, len(current_round), 2):
            if i + 1 < len(current_round):
                strat_B = current_round[i]
                strat_W = current_round[i + 1]
                winner, loser, score = simulate_match(strat_B, strat_W, max_dice_per_player)
                round_matches.append((strat_B.name, strat_W.name, winner.name, score))
                next_round.append(winner)
            else:
                next_round.append(current_round[i])
        tournament_history.append(round_matches)
        current_round = next_round
        round_number += 1

    if len(current_round) == 2:
        final_B = current_round[0]
        final_W = current_round[1]
        winner, loser, score = simulate_match(final_B, final_W, max_dice_per_player)
        tournament_history.append([(final_B.name, final_W.name, winner.name, score)])
        champion = winner.name
        runner_up = loser.name
    else:
        champion = current_round[0].name
        runner_up = None
    return tournament_history, champion, runner_up


# === VISUALIZZAZIONE BRACKET LIVE ===
def render_live_bracket(tournament_history, output_file="live_bracket"):
    """
    Disegna un bracket "sportivo" basato sulla struttura tournament_history.
    Assume un bracket perfetto per 8 partecipanti (3 round: 4 match, 2 match, 1 finale).
    """
    dot = Digraph(comment="Torneo Live", format="png")
    dot.attr(rankdir="LR", splines="polyline")
    num_rounds = len(tournament_history)
    # Crea i nodi per ogni match in ogni round
    for r, round_matches in enumerate(tournament_history):
        with dot.subgraph(name=f"cluster_round{r + 1}") as sub:
            sub.attr(rank="same", label=f"Round {r + 1}")
            for i, (p1, p2, winner, score) in enumerate(round_matches, start=1):
                node_id = f"R{r + 1}_M{i}"
                label = f"{p1} vs {p2}\nWinner: {winner}\nScore: {score}"
                sub.node(node_id, label, shape="box", style="filled", fillcolor="lightgrey")
    # Collegamenti tra match: ogni match del round successivo collega 2 match del round precedente
    for r in range(num_rounds - 1):
        num_matches_current = len(tournament_history[r])
        num_matches_next = len(tournament_history[r + 1])
        for j in range(num_matches_next):
            node_next = f"R{r + 2}_M{j + 1}"
            node_prev1 = f"R{r + 1}_M{2 * j + 1}"
            node_prev2 = f"R{r + 1}_M{2 * j + 2}"
            dot.edge(node_prev1, node_next)
            dot.edge(node_prev2, node_next)
    dot.render(output_file, view=False)
    return output_file + ".png"


# === INTERFACCIA UTENTE (UI) CON TKINTER ===
class TournamentUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulazione Torneo Cephalopod")
        self.geometry("600x400")
        self.create_widgets()

    def create_widgets(self):
        # Titolo
        title = ttk.Label(self, text="Simulazione Torneo Cephalopod", font=("Helvetica", 16))
        title.pack(pady=10)

        # Pulsante per simulare il torneo
        self.simulate_button = ttk.Button(self, text="Simula Torneo", command=self.simula_torneo)
        self.simulate_button.pack(pady=5)

        # Area di testo per mostrare i risultati
        self.result_text = tk.Text(self, height=10, width=70)
        self.result_text.pack(pady=5)

        # Pulsante per visualizzare il bracket
        self.view_button = ttk.Button(self, text="Visualizza Bracket", command=self.visualizza_bracket,
                                      state="disabled")
        self.view_button.pack(pady=5)

    def simula_torneo(self):
        self.result_text.delete("1.0", tk.END)
        strategy_names = [
            "SmartLookahead",
            "Minimax (depth=3)",
            "Expectimax (depth=3)",
            "SmartPositional",
            "SmartMini",
            "SmartBA",
            "Naive",
            "Heuristic"
        ]
        history, champion, runner_up = run_tournament(strategy_names, max_dice_per_player=24)
        result_str = f"Campione: {champion}\n"
        if runner_up:
            result_str += f"Secondo classificato: {runner_up}\n"
        result_str += "\nDettaglio del torneo:\n"
        for i, round_matches in enumerate(history, start=1):
            result_str += f"Round {i}:\n"
            for match in round_matches:
                p1, p2, winner, score = match
                result_str += f"  {p1} vs {p2} -> Winner: {winner} | Score: {score}\n"
        self.result_text.insert(tk.END, result_str)
        self.tournament_history = history
        self.view_button.config(state="normal")

    def visualizza_bracket(self):
        bracket_file = render_live_bracket(self.tournament_history, output_file="live_bracket")
        try:
            os.startfile(bracket_file)
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile aprire il file: {e}")


def main():
    app = TournamentUI()
    app.mainloop()


if __name__ == '__main__':
    main()
