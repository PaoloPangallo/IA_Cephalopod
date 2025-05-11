# 📁 File: cephalopod/alphazero/ui.py

import tkinter as tk
import numpy as np
import torch
from cephalopod.alphazero.neural_network import NeuralNetwork
from cephalopod.alphazero.cephalopod_zero_dynamic import CephalopodZero
from cephalopod.core.board import Board, Die
from cephalopod.core.mechanics import get_opponent, find_capturing_subsets, choose_capturing_subset

CELL_SIZE = 80
BOARD_SIZE = 5
DELAY = 1000  # in millisecondi

class CephalopodUI:
    def __init__(self, agent):
        self.agent = agent
        self.board = Board()
        self.dice_left = {"B": 24, "W": 24}
        self.current_player = "B"

        self.root = tk.Tk()
        self.root.title("CephalopodZero - Autoplay")
        self.canvas = tk.Canvas(self.root, width=CELL_SIZE*BOARD_SIZE, height=CELL_SIZE*BOARD_SIZE)
        self.canvas.pack()
        self.cell_text = {}

        self.draw_board()
        self.root.after(1000, self.play_step)
        self.root.mainloop()

    def draw_board(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x0, y0 = c * CELL_SIZE, r * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="black")
                self.cell_text[(r, c)] = self.canvas.create_text(
                    x0 + CELL_SIZE//2, y0 + CELL_SIZE//2,
                    text="", font=("Arial", 16, "bold")
                )

    def update_board(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                die = self.board.grid[r][c]
                fill = "white"
                text = ""
                if die:
                    fill = "blue" if die.color == "B" else "red"
                    text = str(die.top_face)
                x0, y0 = c * CELL_SIZE, r * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="black")
                self.canvas.itemconfig(self.cell_text[(r, c)], text=text, fill="white" if text else "black")

    def play_step(self):
        if self.board.is_full():
            print("🏁 Partita finita")
            return

        board_tensor = self.agent.encode_board(self.board, self.current_player)
        board_clone = self.board.clone() if hasattr(self.board, "clone") else self.board
        policy = self.agent.mcts.run(board_clone, self.current_player)
        legal_moves = self.board.get_empty_cells()
        probs = np.array([
            policy[r * 5 + c] if (r, c) in legal_moves else 0 for r in range(5) for c in range(5)
        ])
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(25) / 25
        move_index = np.random.choice(25, p=probs)
        r, c = divmod(move_index, 5)

        if self.board.grid[r][c] is not None or self.dice_left[self.current_player] <= 0:
            print("⚠️ Mossa non valida, fine gioco")
            return

        captured, sum_pips = choose_capturing_subset(find_capturing_subsets(self.board, r, c))
        top_face = 6 - sum_pips if captured else 1

        for rr, cc in (captured or []):
            self.board.grid[rr][cc] = None
        self.board.place_die(r, c, Die(self.current_player, top_face))
        self.dice_left[self.current_player] -= 1
        self.current_player = get_opponent(self.current_player)
        self.update_board()
        self.root.after(DELAY, self.play_step)

if __name__ == "__main__":
    model = NeuralNetwork()
    model.load_state_dict(torch.load("cephalopod/alphazero/cephalopod_zero.pth", map_location="cpu"))
    model.eval()
    agent = CephalopodZero(model, mcts_simulations=25)
    CephalopodUI(agent)
