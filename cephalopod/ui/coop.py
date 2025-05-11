import json
import tkinter as tk
from tkinter import ttk, messagebox

import copy
import random
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.core.board import Die, Board
from cephalopod.strategies import SmartLookaheadStrategy, NaiveStrategy, HeuristicStrategy, AggressiveStrategy, \
    SmartBlockAggressiveStrategy
from cephalopod.strategies.deep_thinking import MinimaxStrategy
from cephalopod.strategies.smart_minimax import SmartMinimaxStrategy
from cephalopod.strategies.smart_position import SmartPositionalLookaheadStrategy
from cephalopod.strategies.mod_smart_lookahead import ModSmartLookaheadStrategy
from cephalopod.strategies.non5 import CautiousLookaheadStrategy
from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5
from cephalopod.strategies.deep5lookahead import SmartLookaheadStrategy6
from clon.bc_policy_player import BCPolicyPlayer

from cephalopod.strategies.variant2_lookahead2 import Variant2SmartLookahead
from cephalopod.strategies.agg_lookahead5 import WeirdStrategy
from cephalopod.strategies.weird2 import Weird2Strategy
from cephalopod.strategies.we3 import Weird3Strategy
from cephalopod.strategies.tunMinMax import TunableMinimaxStrategy, EliteStrategy
from cephalopod.strategies.weigh import WeightedTacticalStrategy
from cephalopod.strategies.hybridMinimax import HybridMinimaxStrategy
from cephalopod.strategies.ComplexStrategy import HybridResilientStrategy

from cephalopod.strategies.complex2strategy import ResilientMinimaxStrategy
from simulazioni.tornei.tuning_resilient_minimax import ResilientMinimaxStrategyTunable
from cephalopod.strategies.tuna import Minimax2Strategy

BOARD_SIZE = 5

# Carica il file JSON con i pesi ottimizzati
with open("logs/best_optuna_weights.json", "r") as f:
    best_optuna_config = json.load(f)

STRATEGIES = {
    "Naive": NaiveStrategy,
    "Heuristic": HeuristicStrategy,
    "Aggressive": AggressiveStrategy,
    "SmartLookahead": SmartLookaheadStrategy,
    "Minimax (depth=3)": lambda: MinimaxStrategy(depth=5),
    "SmartMini": SmartMinimaxStrategy,
    "SmartPositional": SmartPositionalLookaheadStrategy,
    "ModifiedLookAhead": ModSmartLookaheadStrategy,
    "SmartBA": SmartBlockAggressiveStrategy,
    "SmartPos": SmartPositionalLookaheadStrategy,
    "Non5": CautiousLookaheadStrategy,
    "Goat5": SmartLookaheadStrategy5,
    "Goat?": SmartLookaheadStrategy6,
    "BCPlayer": lambda: BCPolicyPlayer(),
    "v2": Variant2SmartLookahead,
    "We": WeirdStrategy,
    "We2": Weird2Strategy,
    "we3": Weird3Strategy,
    "Elite": lambda: EliteStrategy(),
    "weig": WeightedTacticalStrategy,
    "Hyb": HybridMinimaxStrategy,
    "compl": HybridResilientStrategy,
    "MinimaxFull": ResilientMinimaxStrategy,
    "prova": ResilientMinimaxStrategyTunable,
    "OptunaBest": lambda: Minimax2Strategy(depth=6),

}


class CephalopodMatchUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cephalopod - Strategy Match Viewer")

        self.board = Board()
        self.cell_labels = []
        self.strategy1 = None
        self.strategy2 = None
        self.current_player = "B"
        self.move_log = []
        self.step_mode = False

        self.setup_widgets()

    def setup_widgets(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(pady=5)

        ttk.Label(control_frame, text="Strategy B:").pack(side=tk.LEFT, padx=5)
        self.strat1_var = tk.StringVar(value="SmartLookahead")
        ttk.Combobox(control_frame, textvariable=self.strat1_var, values=list(STRATEGIES.keys()),
                     state="readonly", width=20).pack(side=tk.LEFT)

        ttk.Label(control_frame, text="Strategy W:").pack(side=tk.LEFT, padx=5)
        self.strat2_var = tk.StringVar(value="Goat5")
        ttk.Combobox(control_frame, textvariable=self.strat2_var, values=list(STRATEGIES.keys()),
                     state="readonly", width=20).pack(side=tk.LEFT)

        ttk.Button(control_frame, text="Start Match", command=self.reset_game).pack(side=tk.LEFT, padx=10)

        self.info_label = ttk.Label(self, text="Match not started")
        self.info_label.pack(pady=5)

        board_frame = ttk.Frame(self)
        board_frame.pack(padx=10, pady=10)

        for r in range(BOARD_SIZE):
            row = []
            for c in range(BOARD_SIZE):
                lbl = tk.Label(board_frame, text=" ", width=4, height=2, bg="#f0f0f0", relief=tk.RAISED,
                               borderwidth=2, font=("Helvetica", 12, "bold"))
                lbl.grid(row=r, column=c, padx=2, pady=2)
                row.append(lbl)
            self.cell_labels.append(row)

    def reset_game(self):
        self.board = Board()
        self.current_player = "B"
        self.strategy1 = STRATEGIES[self.strat1_var.get()]()
        self.strategy2 = STRATEGIES[self.strat2_var.get()]()
        self.move_log.clear()
        self.step_mode = False

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.cell_labels[r][c].config(text=" ", bg="#f0f0f0")

        self.info_label.config(text="Match in progress...")
        self.after(500, self.play_turn)

    def play_turn(self):
        if self.board.is_full():
            self.end_game()
            return

        strat = self.strategy1 if self.current_player == "B" else self.strategy2

        legal_moves = []
        for (r, c) in self.board.get_empty_cells():
            capturing_options = find_capturing_subsets(self.board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                legal_moves.append((r, c, sum_pips, subset))
            else:
                legal_moves.append((r, c, 1, []))

        move = strat.choose_move(self.board, self.current_player)

        if move is None or move not in legal_moves:
            move = random.choice(legal_moves)

        r, c, top_face, captured = move
        for (rr, cc) in captured:
            self.board.grid[rr][cc] = None
            self.update_cell(rr, cc)
        self.board.place_die(r, c, Die(self.current_player, top_face))
        self.update_cell(r, c, self.current_player, top_face)

        self.current_player = "W" if self.current_player == "B" else "B"
        self.after(500, self.play_turn)

    def update_cell(self, r, c, player=None, top_face=None):
        lbl = self.cell_labels[r][c]
        if player is None:
            lbl.config(text=" ", bg="#f0f0f0")
        else:
            lbl.config(text=f"{player}{top_face}", bg="black" if player == "B" else "white",
                       fg="white" if player == "B" else "black")

    def end_game(self):
        b_count = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                      if self.board.grid[r][c] and self.board.grid[r][c].color == "B")
        w_count = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                      if self.board.grid[r][c] and self.board.grid[r][c].color == "W")
        winner = "Draw"
        if b_count > w_count:
            winner = f"{self.strat1_var.get()} wins (B)"
        elif w_count > b_count:
            winner = f"{self.strat2_var.get()} wins (W)"

        self.info_label.config(text=f"Game Over. {winner} (B: {b_count}, W: {w_count})")
        messagebox.showinfo("Match Result", f"{winner}\nFinal Score -> B: {b_count}, W: {w_count}")


def main():
    app = CephalopodMatchUI()
    app.mainloop()


if __name__ == "__main__":
    main()
