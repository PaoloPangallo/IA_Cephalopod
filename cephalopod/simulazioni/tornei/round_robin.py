import tkinter as tk
from tkinter import ttk
import random
import json
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from cephalopod.core.board import Board, Die
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.strategies import (
    NaiveStrategy, HeuristicStrategy, AggressiveStrategy, SmartLookaheadStrategy,

)
from cephalopod.strategies.smart_minimax import SmartMinimaxStrategy

from cephalopod.strategies.smart_position import SmartPositionalLookaheadStrategy

from cephalopod.strategies.mod_smart_lookahead import ModSmartLookaheadStrategy

from cephalopod.strategies.non5 import CautiousLookaheadStrategy

from cephalopod.strategies.deep5lookahead import SmartLookaheadStrategy6

from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5

from cephalopod.strategies import SmartBlockAggressiveStrategy
from cephalopod.strategies.variant2_lookahead2 import Variant2SmartLookahead

from cephalopod.strategies.weird2 import Weird2Strategy
from cephalopod.strategies.tunMinMax import TunableMinimaxStrategy


STRATEGIES = {

    "SmartLookahead": SmartLookaheadStrategy,

    "Goat5": SmartLookaheadStrategy5,
    "Goat?": SmartLookaheadStrategy6,
    "V1": Variant2SmartLookahead,
    "we2": Weird2Strategy,
}

BOARD_SIZE = 5

class TrainingViewer(tk.Tk):
    def __init__(self, configs, strategies_dict):
        super().__init__()
        self.title("Minimax Tuning Viewer")
        self.configs = configs
        self.strategies = list(strategies_dict.items())
        self.strategy_index = 0
        self.current_config_index = 0
        self.scores = []
        self.board = Board()
        self.cell_labels = []
        self.current_player = "B"
        self.tunable_strategy = None
        self.opponent_strategy = None
        self.opponent_name = None
        self.strategies_dict = strategies_dict

        self.setup_widgets()
        self.after(1000, self.start_next_match)

    def setup_widgets(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.info_label = ttk.Label(top_frame, text="Training in corso...")
        self.info_label.pack(side=tk.LEFT)

        self.weights_label = ttk.Label(top_frame, text="Pesi attuali: ")
        self.weights_label.pack(side=tk.RIGHT)

        self.save_button = ttk.Button(top_frame, text="💾 Salva miglior config", command=self.save_best_config)
        self.save_button.pack(side=tk.RIGHT, padx=10)

        self.board_frame = ttk.Frame(self)
        self.board_frame.pack(padx=10, pady=10)

        for r in range(BOARD_SIZE):
            row = []
            for c in range(BOARD_SIZE):
                lbl = tk.Label(self.board_frame, text=" ", width=4, height=2,
                               bg="#f0f0f0", relief=tk.RAISED, borderwidth=2,
                               font=("Helvetica", 12, "bold"))
                lbl.grid(row=r, column=c, padx=2, pady=2)
                row.append(lbl)
            self.cell_labels.append(row)

        self.graph_frame = ttk.Frame(self)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(6, 2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.log_frame = ttk.Frame(self)
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = tk.Text(self.log_frame, height=8, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def start_next_match(self):
        if self.current_config_index >= len(self.configs):
            self.strategy_index += 1
            self.current_config_index = 0

            if self.strategy_index >= len(self.strategies):
                self.finish_training()
                return

        weights = self.configs[self.current_config_index]
        self.tunable_strategy = TunableMinimaxStrategy(weights=weights)

        self.opponent_name, opponent_cls = self.strategies[self.strategy_index]
        self.opponent_strategy = opponent_cls() if callable(opponent_cls) else opponent_cls()

        self.board = Board()
        self.current_player = "B"
        self.clear_board()
        self.update_weights_display(weights)
        self.append_log(f"⚔️ VS {self.opponent_name} — Config #{self.current_config_index + 1}")
        self.after(1, self.play_turn)

    def play_turn(self):
        if self.board.is_full():
            self.evaluate_result()
            return

        strat = self.tunable_strategy if self.current_player == "B" else self.opponent_strategy

        legal_moves = []
        for (r, c) in self.board.get_empty_cells():
            capturing_options = find_capturing_subsets(self.board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                legal_moves.append((r, c, sum_pips, subset))
            else:
                legal_moves.append((r, c, 1, []))

        move = strat.choose_move(self.board, self.current_player)
        if move not in legal_moves:
            move = random.choice(legal_moves)

        r, c, top_face, captured = move
        for rr, cc in captured:
            self.board.grid[rr][cc] = None
            self.update_cell(rr, cc)

        self.board.place_die(r, c, Die(self.current_player, top_face))
        self.update_cell(r, c, self.current_player, top_face)
        self.current_player = "W" if self.current_player == "B" else "B"
        self.after(1, self.play_turn)  # 10 ms delay

    def evaluate_result(self):
        b_score = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                      if self.board.grid[r][c] and self.board.grid[r][c].color == "B")
        w_score = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                      if self.board.grid[r][c] and self.board.grid[r][c].color == "W")

        result = "Pareggio"
        win = 0
        if b_score > w_score:
            result = "✅ Tunable (B) vince"
            win = 1
        elif w_score > b_score:
            result = "❌ Opponent (W) vince"

        self.info_label.config(
            text=f"[{self.opponent_name}] Config {self.current_config_index + 1}/{len(self.configs)}: {result} (B:{b_score}, W:{w_score})"
        )
        self.append_log(f"[{self.opponent_name}] Config #{self.current_config_index + 1}: {result} - B:{b_score}, W:{w_score}")
        self.scores.append((self.opponent_name, self.current_config_index + 1, win))
        self.current_config_index += 1
        self.update_graph()
        self.after(2000, self.start_next_match)

    def update_cell(self, r, c, player=None, top_face=None):
        lbl = self.cell_labels[r][c]
        if player is None:
            lbl.config(text=" ", bg="#f0f0f0")
        else:
            lbl.config(text=f"{player}{top_face}", bg="black" if player == "B" else "white",
                       fg="white" if player == "B" else "black")

    def clear_board(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.update_cell(r, c)

    def update_weights_display(self, weights):
        text = ", ".join(f"{k}:{v}" for k, v in weights.items())
        self.weights_label.config(text=f"Pesi: {text}")

    def update_graph(self):
        self.ax.clear()
        opponent_groups = {}
        for name, idx, res in self.scores:
            opponent_groups.setdefault(name, []).append((idx, res))
        for name, values in opponent_groups.items():
            x = [i for i, _ in values]
            y = [r for _, r in values]
            self.ax.plot(x, y, marker='o', linestyle='-', label=name)
        self.ax.set_title("Performance TunableMinimax (B)")
        self.ax.set_xlabel("Configurazione")
        self.ax.set_ylabel("Vittoria (1 = sì)")
        self.ax.set_ylim(0, 1.1)
        self.ax.legend()
        self.canvas.draw()

    def append_log(self, text):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def save_best_config(self):
        if hasattr(self, 'best_config'):
            with open("best_config.json", "w") as f:
                json.dump(self.best_config, f, indent=4)
            self.append_log("✅ Configurazione migliore salvata in 'best_config.json'")
        else:
            self.append_log("⚠️ Nessuna configurazione salvata: termina prima l'allenamento.")

    def finish_training(self):
        # Calcola le vittorie per configurazione
        from collections import defaultdict

        win_counter = defaultdict(int)  # config_index → numero di vittorie

        for _, config_index, win in self.scores:
            if win:
                win_counter[config_index] += 1

        if not win_counter:
            self.append_log("❌ Nessuna configurazione ha vinto almeno una partita.")
            return

        # Trova la configurazione con più vittorie
        best_config_index = max(win_counter.items(), key=lambda x: x[1])[0] - 1  # da 1-based a 0-based
        self.best_config = self.configs[best_config_index]
        best_wins = win_counter[best_config_index + 1]

        self.append_log(f"🏁 Allenamento completato!")
        self.append_log(f"🥇 Migliore: config #{best_config_index + 1} con {best_wins} vittorie complessive")
        self.save_performance_log()

    def save_performance_log(self):
        with open("performance_log.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Strategia", "Configurazione", "Vittoria (1=Si, 0=No)"])
            for row in self.scores:
                writer.writerow(row)
        self.append_log("📄 Log performance salvato in 'performance_log.csv'")


if __name__ == "__main__":
    with open("../../ui/tuning_configs_60k.json") as f:
        configs = json.load(f)

    app = TrainingViewer(configs=configs[:100], strategies_dict=STRATEGIES)
    app.mainloop()
