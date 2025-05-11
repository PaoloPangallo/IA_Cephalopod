import threading
import torch
from torch.utils.data import DataLoader

from RL.base_rl_player import RLPlayer
from RL.reward_shaping.rewardshaping import AdvancedBoardShaper

import random
from cephalopod.core.board import Die, Board
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
import pickle
import copy
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from cephalopod.RL.base_rl_player import RLPlayer
from cephalopod.RL.reward_shaping.rewardshaping import AdvancedBoardShaper
from cephalopod.game_modes.cephalopod_game_dynamic import CephalopodGameDynamic
from cephalopod.strategies import SmartLookaheadStrategy, SmartBlockAggressiveStrategy
from cephalopod.strategies.deep5lookahead import SmartLookaheadStrategy6
from cephalopod.strategies.deep_thinking import MinimaxStrategy, ExpectimaxStrategy
from cephalopod.strategies.mod_smart_lookahead import ModSmartLookaheadStrategy
from cephalopod.strategies.non5 import CautiousLookaheadStrategy
from cephalopod.strategies.prob_lookahead import ProbabilisticLookaheadStrategy
from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5
from cephalopod.strategies.smart_minimax import SmartMinimaxStrategy
from cephalopod.strategies.smart_position import SmartPositionalLookaheadStrategy
from clon.evaluate_bc_vs_expert import BCPolicyPlayer
from strategies import NaiveStrategy, HeuristicStrategy, AggressiveStrategy

# Placeholder per le configurazioni della strategia probabilistica
best_config = {"param": "value"}
best_multi_config = {"param": "value"}

# Dizionario delle strategie/euristiche disponibili
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
    "Goat?": SmartLookaheadStrategy6,
    "RLPlayer": lambda: RLPlayer(
        name="RL",
        exp_rate=0.0,
        policy_path="policies/policy_RL_advanced.pkl",
        reward_shaper=AdvancedBoardShaper()
    ),
    "BCPlayer": lambda: BCPolicyPlayer(),
}

##########################################################
# TRAINING & DATASET GENERATION
##########################################################

def train_behavior_model(epochs=20, model_save_path="policy_bc.pt", optimizer_path="optimizer_state.pt",
                         log_path="training_log.csv", patience=5, verbose=True, resume=True, log_callback=None):
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)


    start_epoch = 0
    best_loss = float('inf')
    epochs_without_improvement = 0

    if resume and os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location="cpu"))
        log(f"[✓] Modello pre-addestrato caricato da '{model_save_path}'")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
            log(f"[✓] Stato dell'optimizer caricato da '{optimizer_path}'")

    criterion = torch.nn.MSELoss()

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "batch", "batch_loss", "total_loss", "timestamp"])

    log("[INFO] Inizio training...")
    all_losses = []
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        for batch_idx, (states, actions) in enumerate(dataloader):
            try:
                preds = model(states)
                loss = criterion(preds.float(), actions.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss

                if verbose and batch_idx % 10 == 0:
                    log(f"[Ep {epoch + 1} | Batch {batch_idx}/{total_batches}] Loss: {batch_loss:.4f}")

                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, batch_idx, batch_loss, total_loss, datetime.now().isoformat()])

            except Exception as e:
                log(f"[ERRORE] Ep {epoch+1} | Batch {batch_idx} — {e}")

        all_losses.append(total_loss)
        log(f"[✓] Epoca {epoch + 1} completata. Loss totale: {total_loss:.2f}")

        if total_loss < best_loss:
            best_loss = total_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            log(f"[✓] Nuovo best model salvato in '{model_save_path}'")
        else:
            epochs_without_improvement += 1
            log(f"[INFO] Nessun miglioramento. ({epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                log("[STOP] Early stopping attivato.")
                break

    # Creazione del grafico di loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(all_losses) + 1), all_losses, marker='o')
    plt.title("Loss per Epoca")
    plt.xlabel("Epoca")
    plt.ylabel("Loss Totale")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_plot.png")
    log("[✓] Grafico salvato in 'training_loss_plot.png'")
    return all_losses


def generate_expert_dataset(num_games=1000, save_path="expert_dataset.pkl", log_callback=None):
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5
    expert = SmartLookaheadStrategy5()
    data = []

    for i in range(num_games):
        board = Board()
        color = "B"

        # Inizializzazione con 4 mosse casuali
        for _ in range(4):
            empty = board.get_empty_cells()
            if not empty:
                break
            r, c = random.choice(empty)
            state = str([[str(cell) if cell else "" for cell in row] for row in board.grid])
            data.append((state, (r, c, 1, [])))
            board.place_die(r, c, Die(color, 1))
            color = "W" if color == "B" else "B"

        # Proseguimento della partita fino al riempimento della board
        while not board.is_full():
            move = expert.choose_move(board, color)
            state = str([[str(cell) if cell else "" for cell in row] for row in board.grid])
            data.append((state, move))
            r, c, top_face, captured = move
            for (rr, cc) in captured:
                board.grid[rr][cc] = None
            board.place_die(r, c, Die(color, top_face))
            color = "W" if color == "B" else "B"

        log(f"[INFO] Gioco {i + 1}/{num_games} completato.")

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    log(f"[✓] Dataset salvato in '{save_path}'")

##########################################################
# GUI PER IL GIOCO CON MODALITÀ CONTROLLO
##########################################################

class GameFrame(ttk.Frame):
    def __init__(self, master, player1, player2, controlled_mode=False, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.player1 = player1
        self.player2 = player2
        self.controlled_mode = controlled_mode
        self.history = []    # Lista di stati (board, current_color, descrizione mossa)
        self.current_step = 0
        self.reset_game()
        self.init_ui()
        if self.controlled_mode:
            self.create_control_buttons()
            self.save_state("Stato iniziale")
        else:
            self.after(1000, self.play_game)  # Modalità automatica

    def reset_game(self):
        self.board = Board()
        self.current_color = "B"
        self.cells = []

    def init_ui(self):
        self.board_frame = ttk.Frame(self)
        self.board_frame.pack(padx=10, pady=10)
        for r in range(5):
            row = []
            for c in range(5):
                lbl = tk.Label(self.board_frame, text="", width=4, height=2, relief=tk.RAISED,
                               font=("Arial", 12), bg="lightgray")
                lbl.grid(row=r, column=c, padx=2, pady=2)
                row.append(lbl)
            self.cells.append(row)
        self.status_label = ttk.Label(self, text="Gioco in corso...", font=("Arial", 12))
        self.status_label.pack(pady=5)

    def create_control_buttons(self):
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(pady=5)
        self.prev_btn = ttk.Button(self.control_frame, text="Precedente", command=self.prev_move)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.next_btn = ttk.Button(self.control_frame, text="Successivo", command=self.next_move)
        self.next_btn.pack(side=tk.LEFT, padx=5)

    def save_state(self, move_description):
        # Salva una copia profonda dello stato attuale della board, il turno corrente e la descrizione della mossa
        board_copy = copy.deepcopy(self.board)
        state = {
            "board": board_copy,
            "current_color": self.current_color,
            "move": move_description
        }
        self.history.append(state)
        self.current_step = len(self.history) - 1

    def load_state(self, state):
        self.board = copy.deepcopy(state["board"])
        self.current_color = state["current_color"]
        self.update_ui()
        self.status_label.config(text=f"Stato: {state['move']}")

    def next_move(self):
        # Se esiste uno stato successivo già salvato, ci si sposta avanti; altrimenti simula la mossa successiva
        if self.current_step < len(self.history) - 1:
            self.current_step += 1
            state = self.history[self.current_step]
            self.load_state(state)
        else:
            if self.board.is_full():
                self.declare_winner()
                return
            player = self.player1 if self.current_color == "B" else self.player2
            move = player.choose_move(self.board, self.current_color)
            r, c, top_face, captured = move
            for (rr, cc) in captured:
                self.board.grid[rr][cc] = None
            self.board.place_die(r, c, Die(self.current_color, top_face))
            move_desc = f"{self.current_color} -> ({r},{c}) Face:{top_face}, Catturati:{captured}"
            self.current_color = "W" if self.current_color == "B" else "B"
            self.save_state(move_desc)
            self.load_state(self.history[self.current_step])

    def prev_move(self):
        if self.current_step > 0:
            self.current_step -= 1
            state = self.history[self.current_step]
            self.load_state(state)

    def play_game(self):
        # Modalità automatica: esegue le mosse in loop finché la board non è piena
        if self.board.is_full():
            self.declare_winner()
            return
        player = self.player1 if self.current_color == "B" else self.player2
        move = player.choose_move(self.board, self.current_color)
        r, c, top_face, captured = move
        for (rr, cc) in captured:
            self.board.grid[rr][cc] = None
        self.board.place_die(r, c, Die(self.current_color, top_face))
        self.update_ui()
        self.current_color = "W" if self.current_color == "B" else "B"
        self.after(300, self.play_game)

    def update_ui(self):
        for r in range(5):
            for c in range(5):
                die = self.board.grid[r][c]
                if die:
                    self.cells[r][c]["text"] = f"{die.color}{int(die.top_face)}"
                    self.cells[r][c]["bg"] = "black" if die.color == "B" else "white"
                    self.cells[r][c]["fg"] = "white" if die.color == "B" else "black"
                else:
                    self.cells[r][c]["text"] = ""
                    self.cells[r][c]["bg"] = "lightgray"

    def declare_winner(self):
        b_count = sum(1 for r in range(5) for c in range(5)
                      if self.board.grid[r][c] and self.board.grid[r][c].color == "B")
        w_count = sum(1 for r in range(5) for c in range(5)
                      if self.board.grid[r][c] and self.board.grid[r][c].color == "W")
        if b_count > w_count:
            result = f"Player B (Nero) vince! ({b_count} a {w_count})"
        elif w_count > b_count:
            result = f"Player W (Bianco) vince! ({w_count} a {b_count})"
        else:
            result = f"Pareggio! ({b_count} a {w_count})"
        self.status_label.config(text=result)

##########################################################
# APPLICAZIONE PRINCIPALE CON INTERFACCIA A TAB
##########################################################

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cephalopod Interactive Suite")
        self.geometry("800x600")
        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Training
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="Training")
        controls_frame = ttk.Frame(self.train_frame)
        controls_frame.pack(pady=10)
        ttk.Label(controls_frame, text="Epoche:").grid(row=0, column=0, padx=5, pady=5)
        self.epochs_entry = ttk.Entry(controls_frame, width=10)
        self.epochs_entry.insert(0, "20")
        self.epochs_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(controls_frame, text="Patience:").grid(row=0, column=2, padx=5, pady=5)
        self.patience_entry = ttk.Entry(controls_frame, width=10)
        self.patience_entry.insert(0, "5")
        self.patience_entry.grid(row=0, column=3, padx=5, pady=5)
        self.resume_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="Resume", variable=self.resume_var).grid(row=0, column=4, padx=5, pady=5)
        self.start_train_btn = ttk.Button(controls_frame, text="Start Training", command=self.start_training)
        self.start_train_btn.grid(row=0, column=5, padx=5, pady=5)
        self.train_log = tk.Text(self.train_frame, height=15)
        self.train_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 2: Generazione Expert Dataset
        self.dataset_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_frame, text="Expert Dataset")
        ds_controls = ttk.Frame(self.dataset_frame)
        ds_controls.pack(pady=10)
        ttk.Label(ds_controls, text="Numero di giochi:").grid(row=0, column=0, padx=5, pady=5)
        self.num_games_entry = ttk.Entry(ds_controls, width=10)
        self.num_games_entry.insert(0, "1000")
        self.num_games_entry.grid(row=0, column=1, padx=5, pady=5)
        self.generate_ds_btn = ttk.Button(ds_controls, text="Genera Dataset", command=self.generate_dataset)
        self.generate_ds_btn.grid(row=0, column=2, padx=5, pady=5)
        self.ds_log = tk.Text(self.dataset_frame, height=10)
        self.ds_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 3: Game – Selezione strategie e modalità di simulazione
        self.game_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.game_tab, text="Game")
        game_controls = ttk.Frame(self.game_tab)
        game_controls.pack(pady=10)
        ttk.Label(game_controls, text="Giocatore 1:").grid(row=0, column=0, padx=5, pady=5)
        self.player1_combo = ttk.Combobox(game_controls, values=list(STRATEGIES.keys()))
        self.player1_combo.current(0)
        self.player1_combo.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(game_controls, text="Giocatore 2:").grid(row=0, column=2, padx=5, pady=5)
        self.player2_combo = ttk.Combobox(game_controls, values=list(STRATEGIES.keys()))
        self.player2_combo.current(1)
        self.player2_combo.grid(row=0, column=3, padx=5, pady=5)
        # Selezione modalità di simulazione
        self.controlled_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(game_controls, text="Modalità Controllata", variable=self.controlled_mode_var).grid(row=0, column=4, padx=5, pady=5)
        self.start_game_btn = ttk.Button(game_controls, text="Start Game", command=self.start_game)
        self.start_game_btn.grid(row=0, column=5, padx=5, pady=5)

    def log_train(self, msg):
        self.train_log.insert(tk.END, msg + "\n")
        self.train_log.see(tk.END)

    def log_dataset(self, msg):
        self.ds_log.insert(tk.END, msg + "\n")
        self.ds_log.see(tk.END)

    def start_training(self):
        try:
            epochs = int(self.epochs_entry.get())
            patience = int(self.patience_entry.get())
        except ValueError:
            self.log_train("[ERRORE] Controlla i valori inseriti per epoche e patience.")
            return
        self.start_train_btn.config(state=tk.DISABLED)
        self.log_train("[INFO] Avvio training in background...")

        def run_training():
            train_behavior_model(epochs=epochs, patience=patience,
                                 resume=self.resume_var.get(), log_callback=self.log_train)
            self.log_train("[INFO] Training completato.")
            self.start_train_btn.config(state=tk.NORMAL)
        threading.Thread(target=run_training, daemon=True).start()

    def generate_dataset(self):
        try:
            num_games = int(self.num_games_entry.get())
        except ValueError:
            self.log_dataset("[ERRORE] Inserisci un numero valido di giochi.")
            return
        self.generate_ds_btn.config(state=tk.DISABLED)
        self.log_dataset("[INFO] Generazione dataset in background...")

        def run_generation():
            generate_expert_dataset(num_games=num_games, log_callback=self.log_dataset)
            self.log_dataset("[INFO] Generazione completata.")
            self.generate_ds_btn.config(state=tk.NORMAL)
        threading.Thread(target=run_generation, daemon=True).start()

    def start_game(self):
        strat1_key = self.player1_combo.get()
        strat2_key = self.player2_combo.get()
        try:
            player1 = STRATEGIES[strat1_key]()
            player2 = STRATEGIES[strat2_key]()
        except Exception as e:
            print(f"[ERRORE] Istanziazione delle strategie: {e}")
            return

        game_window = tk.Toplevel(self)
        game_window.title("Cephalopod Battle Game")
        controlled = self.controlled_mode_var.get()
        game_frame = GameFrame(game_window, player1, player2, controlled_mode=controlled)
        game_frame.pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    app = App()
    app.mainloop()
