import tkinter as tk
from tkinter import ttk
import random
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from cephalopod.core.board import Board, Die
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.strategies import SmartLookaheadStrategy
from cephalopod.strategies.tunMinMax import TunableMinimaxStrategy

BOARD_SIZE = 5

class TrainingViewer(tk.Tk):
    def __init__(self, configs, opponent_cls):
        super().__init__()
        self.title("Minimax Tuning Viewer")
        self.configs = configs
        self.opponent_cls = opponent_cls
        self.current_config_index = 0
        self.scores = []
        self.board = Board()
        self.cell_labels = []
        self.current_player = "B"
        self.tunable_strategy = None
        self.opponent_strategy = None

        self.setup_widgets()
        self.after(1000, self.start_next_match)

    def setup_widgets(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.info_label = ttk.Label(top_frame, text="Training in corso...")
        self.info_label.pack(side=tk.LEFT)

        self.weights_label = ttk.Label(top_frame, text="Pesi attuali: ")
        self.weights_label.pack(side=tk.RIGHT)

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

        # Frame grafico
        self.graph_frame = ttk.Frame(self)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(6, 2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_title("Performance TunableMinimax (B)")
        self.ax.set_xlabel("Configurazione")
        self.ax.set_ylabel("Vittoria")

    def start_next_match(self):
        if self.current_config_index >= len(self.configs):
            self.finish_training()
            return

        weights = self.configs[self.current_config_index]
        self.tunable_strategy = TunableMinimaxStrategy(weights=weights)
        self.opponent_strategy = self.opponent_cls()
        self.board = Board()
        self.current_player = "B"
        self.clear_board()
        self.update_weights_display(weights)
        self.after(500, self.play_turn)

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
        self.after(300, self.play_turn)

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

        weights = self.configs[self.current_config_index]
        self.info_label.config(text=f"Config {self.current_config_index + 1}/{len(self.configs)}: {result} (B:{b_score}, W:{w_score})")
        self.scores.append((self.current_config_index + 1, win))
        self.current_config_index += 1
        self.update_graph()
        self.after(1200, self.start_next_match)

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
        x = [idx for idx, _ in self.scores]
        y = [res for _, res in self.scores]
        self.ax.clear()
        self.ax.plot(x, y, marker='o', linestyle='-', label='Vittorie Tunable (B)')
        self.ax.set_title("Performance TunableMinimax (B)")
        self.ax.set_xlabel("Configurazione")
        self.ax.set_ylabel("Vittoria (1 = sì)")
        self.ax.set_ylim(0, 1.1)
        self.ax.legend()
        self.canvas.draw()

    def finish_training(self):
        best_config = max(self.scores, key=lambda x: x[1])
        self.info_label.config(text=f"🏁 Allenamento completato! Best config: #{best_config[0]}")


if __name__ == "__main__":
    with open("tuning_configs_60k.json") as f:
        configs = json.load(f)

    app = TrainingViewer(configs=configs[:200], opponent_cls=SmartLookaheadStrategy)
    app.mainloop()
