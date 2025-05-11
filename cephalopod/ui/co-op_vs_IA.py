import tkinter as tk
from tkinter import ttk, messagebox


from cephalopod.RL.base_rl_player import RLPlayer
from cephalopod.RL.reward_shaping.rewardshaping import AdvancedBoardShaper

# Strategie tradizionali
from cephalopod.strategies import SmartLookaheadStrategy, SmartBlockAggressiveStrategy
from cephalopod.strategies.deep5lookahead import SmartLookaheadStrategy6
from cephalopod.strategies.deep_thinking import MinimaxStrategy, ExpectimaxStrategy
from cephalopod.strategies.mod_smart_lookahead import ModSmartLookaheadStrategy
from cephalopod.strategies.non5 import CautiousLookaheadStrategy
from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5
from cephalopod.strategies.smart_minimax import SmartMinimaxStrategy
from cephalopod.strategies.smart_position import SmartPositionalLookaheadStrategy
from cephalopod.strategies import NaiveStrategy, HeuristicStrategy, AggressiveStrategy
from clon.bc_policy_player import BCPolicyPlayer

# 👇 Import della tua strategia behavior cloning
from core.board import Board, Die
from core.mechanics import choose_capturing_subset, find_capturing_subsets
from cephalopod.strategies.weird2 import Weird2Strategy

BOARD_SIZE = 5





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

    "Goat5": SmartLookaheadStrategy5,
    "Goat?": SmartLookaheadStrategy6,

    "RLPlayer": lambda: RLPlayer(
        name="RL",
        exp_rate=0.0,
        policy_path="policies/policy_RL_advanced.pkl",
        reward_shaper=AdvancedBoardShaper()
    ),

    "BCPlayer": lambda: BCPolicyPlayer(),
     "We2": Weird2Strategy

}


class CephalopodPlayUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cephalopod - Play Against AI")

        self.board = Board()
        self.cell_labels = []
        self.current_player = "B"  # Human is always B
        self.ai_player = "W"
        self.ai_strategy = NaiveStrategy()

        self.setup_widgets()

    def setup_widgets(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(pady=5)

        ttk.Label(top_frame, text="Select AI Strategy:").pack(side=tk.LEFT, padx=5)
        self.strategy_var = tk.StringVar(value="Naive")
        self.strategy_menu = ttk.Combobox(top_frame, textvariable=self.strategy_var, values=list(STRATEGIES.keys()),
                                          state="readonly")
        self.strategy_menu.pack(side=tk.LEFT, padx=5)

        start_button = ttk.Button(top_frame, text="Start New Game", command=self.reset_game)
        start_button.pack(side=tk.LEFT, padx=5)

        self.info_label = ttk.Label(self, text="Click a cell to play as B (You)")
        self.info_label.pack(pady=5)

        board_frame = ttk.Frame(self)
        board_frame.pack(padx=10, pady=10)

        for r in range(BOARD_SIZE):
            row = []
            for c in range(BOARD_SIZE):
                lbl = tk.Label(board_frame, text=" ", width=4, height=2, bg="#dcdcdc", relief=tk.RAISED, borderwidth=2,
                               font=("Helvetica", 12, "bold"))
                lbl.grid(row=r, column=c, padx=2, pady=2)
                lbl.bind("<Button-1>", lambda e, row=r, col=c: self.on_cell_click(row, col))
                row.append(lbl)
            self.cell_labels.append(row)

    def reset_game(self):
        self.board = Board()
        self.current_player = "B"
        self.ai_strategy = STRATEGIES[self.strategy_var.get()]()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.cell_labels[r][c].config(text=" ", bg="#dcdcdc")
        self.info_label.config(text="Game reset. Your turn (B).")

    def on_cell_click(self, row, col):
        if self.current_player != "B":
            return
        if self.board.grid[row][col] is not None:
            return

        # Check for capturing move
        capturing_options = find_capturing_subsets(self.board, row, col)
        if capturing_options:
            subset, sum_pips = choose_capturing_subset(capturing_options)
            for (rr, cc) in subset:
                self.board.grid[rr][cc] = None
                self.update_cell(rr, cc)
            top_face = sum_pips
        else:
            subset = []
            top_face = 1

        self.board.place_die(row, col, Die("B", top_face))
        self.update_cell(row, col, "B", top_face)
        self.current_player = "W"
        self.after(500, self.play_ai_turn)

    def play_ai_turn(self):
        if self.board.is_full():
            self.end_game()
            return

        move = self.ai_strategy.choose_move(self.board, "W")
        if move is None:
            self.end_game()
            return

        r, c, top_face, captured = move
        for (rr, cc) in captured:
            self.board.grid[rr][cc] = None
            self.update_cell(rr, cc)
        self.board.place_die(r, c, Die("W", top_face))
        self.update_cell(r, c, "W", top_face)
        self.current_player = "B"

        if self.board.is_full():
            self.end_game()

    def update_cell(self, r, c, player=None, top_face=None):
        lbl = self.cell_labels[r][c]
        if player is None:
            lbl.config(text=" ", bg="#dcdcdc")
        else:
            lbl.config(text=f"{player}{top_face}", bg="black" if player == "B" else "white",
                       fg="white" if player == "B" else "black")

    def end_game(self):
        b_count = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                      if self.board.grid[r][c] is not None and self.board.grid[r][c].color == "B")
        w_count = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                      if self.board.grid[r][c] is not None and self.board.grid[r][c].color == "W")
        winner = "Draw"
        if b_count > w_count:
            winner = "You win (B)!"
        elif w_count > b_count:
            winner = "AI wins (W)!"

        self.info_label.config(text=f"Game Over. {winner} (B: {b_count}, W: {w_count})")
        messagebox.showinfo("Game Over", f"{winner}\nFinal Score -> B: {b_count}  W: {w_count}")


def main():
    app = CephalopodPlayUI()
    app.mainloop()


if __name__ == "__main__":
    main()
