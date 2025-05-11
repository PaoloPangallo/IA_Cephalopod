import concurrent.futures
import itertools
import tkinter as tk
from tkinter import ttk

# Variabili globali per i player AI, verranno impostate dinamicamente
playerBmodule = None
playerRmodule = None


# ============================
# CLASSI DI BASE E LOGICA DI GIOCO
# ============================

class Game:
    """
    Classe base per definire un gioco.
    Sottoclasse questa classe e implementa: actions, result, is_terminal e utility.
    L'attributo .initial deve contenere lo stato iniziale.
    """

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, move):
        raise NotImplementedError

    def is_terminal(self, state):
        return not self.actions(state)

    def utility(self, state, player):
        raise NotImplementedError


class Board:
    def __init__(self, size, board=None, to_move="Blue", last_move=None):
        self.size = size
        if board is None:
            self.board = [[None for _ in range(size)] for _ in range(size)]
        else:
            self.board = board
        self.to_move = to_move      # "Blue" o "Red"
        self.last_move = last_move  # (cella_inserimento, celle_catturate)

    def copy(self):
        new_board = [row[:] for row in self.board]
        return Board(self.size, new_board, self.to_move, self.last_move)

    def is_full(self):
        for row in self.board:
            if any(cell is None for cell in row):
                return False
        return True

    def count(self, player):
        cnt = 0
        for row in self.board:
            for cell in row:
                if cell is not None and cell[0] == player:
                    cnt += 1
        return cnt

# Funzione ausiliaria che genera tutti i sottoinsiemi (delle celle adiacenti) con dimensione minima min_size.
def get_subsets(adjacent, min_size=2):
    subsets = []
    n = len(adjacent)
    for r in range(min_size, n+1):
        for comb in itertools.combinations(adjacent, r):
            subsets.append(list(comb))
    return subsets



class CephalopodGame(Game):
    """
    Classe che definisce le regole del gioco Cephalopod.
    """

    def __init__(self, size=5, first_player="Blue"):
        self.size = size
        self.first_player = first_player
        self.initial = Board(size, to_move=first_player)

    def actions(self, state):
        moves = []
        for r in range(state.size):
            for c in range(state.size):
                if state.board[r][c] is None:  # cella vuota
                    adjacent = []
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < state.size and 0 <= nc < state.size:
                            if state.board[nr][nc] is not None:
                                adjacent.append(((nr, nc), state.board[nr][nc][1]))
                    capture_moves = []
                    if len(adjacent) >= 2:
                        subsets = get_subsets(adjacent, 2)
                        for subset in subsets:
                            s = sum(pip for pos, pip in subset)
                            if 2 <= s <= 6:
                                positions = tuple(pos for pos, pip in subset)
                                capture_moves.append(((r, c), s, positions))
                    if capture_moves:
                        moves.extend(capture_moves)
                    else:
                        moves.append(((r, c), 1, ()))
        return moves

    def result(self, state, move):
        new_state = state.copy()
        (r, c), pip, captured = move
        current_player = state.to_move
        new_state.board[r][c] = (current_player, pip)
        for pos in captured:
            rr, cc = pos
            new_state.board[rr][cc] = None
        new_state.last_move = ((r, c), captured)
        new_state.to_move = "Red" if current_player == "Blue" else "Blue"
        return new_state

    def is_terminal(self, state):
        return state.is_full()

    def utility(self, state, player="Blue"):
        countBlue = state.count("Blue")
        countRed = state.count("Red")
        return 1 if player == "Blue" and countBlue > countRed else -1


def random_player(game, state, timeout=3):
    """
    Giocatore artificiale che sceglie una mossa casuale.
    """
    moves = game.actions(state)
    return random.choice(moves)


# ============================
# DIALOGO DI SETUP DELLA PARTITA
# ============================

class GameSetupDialog(tk.Toplevel):
    """
    Finestra di setup per selezionare il tipo di giocatore per Blue e Red.
    Ogni giocatore può essere "human" oppure una strategia AI scelta da un menu a tendina.
    """

    def __init__(self, parent, available_players):
        super().__init__(parent)
        self.title("Setup Partita")
        self.resizable(False, False)
        self.result = None

        tk.Label(self, text="Seleziona tipo per Blue:").grid(row=0, column=0, padx=10, pady=10)
        self.blue_var = tk.StringVar(value="human")
        self.blue_combo = ttk.Combobox(self, textvariable=self.blue_var, state="readonly",
                                       values=["human"] + available_players)
        self.blue_combo.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(self, text="Seleziona tipo per Red:").grid(row=1, column=0, padx=10, pady=10)
        self.red_var = tk.StringVar(value="human")
        self.red_combo = ttk.Combobox(self, textvariable=self.red_var, state="readonly",
                                      values=["human"] + available_players)
        self.red_combo.grid(row=1, column=1, padx=10, pady=10)

        self.start_button = tk.Button(self, text="Avvia Partita", command=self.on_start)
        self.start_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.grab_set()
        self.wait_window(self)

    def on_start(self):
        self.result = {"Blue": self.blue_var.get(), "Red": self.red_var.get()}
        self.destroy()


# ============================
# INTERFACCIA GRAFICA
# ============================

import tkinter as tk
from tkinter import simpledialog, messagebox
import random, time, threading, importlib
import concurrent.futures


class CephalopodGUI:
    def __init__(self, game, player_types, ai_names, time_out=3):
        self.game = game
        self.player_types = player_types
        self.ai_names = ai_names  # E.g., {"Blue": "player.playerMinimax", "Red": "Human"}
        self.state_history = [game.initial]
        self.current_index = 0
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.waiting_for_human = False
        self.human_move = None
        self.time_out = time_out

        self.capture_selection_mode = False
        self.pending_placement = None
        self.pending_candidate_moves = []
        self.selected_capture_cells = set()

        self.auto_mode = False
        self.show_auto = (self.player_types.get("Blue") == "ai" and self.player_types.get("Red") == "ai")

        self.root = tk.Tk()
        self.root.title("Cephalopod")
        self.root.geometry("500x500")
        self.root.configure(bg="white")

        self.info_label = tk.Label(self.root, text="", font=("Helvetica", 12), bg="white")
        self.info_label.pack(pady=5)

        self.board_frame = tk.Frame(self.root, bg="white")
        self.board_frame.pack(pady=10)

        self.controls_frame = tk.Frame(self.root, bg="white")
        self.controls_frame.pack(pady=10)

        self.cells = [[None for _ in range(self.game.size)] for _ in range(self.game.size)]
        for r in range(self.game.size):
            for c in range(self.game.size):
                lbl = tk.Label(self.board_frame, text="", width=4, height=2,
                               borderwidth=2, relief="ridge", font=("Helvetica", 16), bg="white", anchor="center")
                lbl.grid(row=r, column=c, padx=3, pady=3)
                lbl.bind("<Button-1>", lambda e, row=r, col=c: self.cell_clicked(row, col))
                self.cells[r][c] = lbl

        self.prev_button = tk.Button(self.controls_frame, text="Precedente", command=self.prev_move,
                                     font=("Helvetica", 12), padx=10, pady=5)
        self.prev_button.grid(row=0, column=0, padx=5)
        self.next_button = tk.Button(self.controls_frame, text="Successivo", command=lambda: self.next_move(),
                                     font=("Helvetica", 12), padx=10, pady=5)
        self.next_button.grid(row=0, column=1, padx=5)

        if self.show_auto:
            self.auto_button = tk.Button(self.controls_frame, text="Auto", highlightbackground="red",
                                         command=self.toggle_auto,
                                         font=("Helvetica", 12), padx=10, pady=5)
            self.auto_button.grid(row=0, column=2, padx=5)

        self.confirm_button = tk.Button(self.controls_frame, text="Conferma", highlightbackground="green",
                                        command=self.confirm_capture,
                                        font=("Helvetica", 12), padx=10, pady=5)
        self.confirm_button.grid_forget()

        self.status_label = tk.Label(self.controls_frame, text="", bg="white", font=("Helvetica", 12))
        self.status_label.grid(row=1, column=0, columnspan=4)

        self.update_board()

    def current_state(self):
        return self.state_history[self.current_index]

    def update_board(self):
        state = self.current_state()
        for r in range(state.size):
            for c in range(state.size):
                cell = state.board[r][c]
                lbl = self.cells[r][c]
                if cell is None:
                    lbl.config(text="", bg="white")
                else:
                    player, pip = cell
                    color = "lightblue" if player == "Blue" else "lightcoral"
                    lbl.config(text=str(pip), bg=color)
                lbl.config(relief="ridge", borderwidth=4)
        if state.last_move:
            (r, c), captured = state.last_move
            self.cells[r][c].config(relief="solid", borderwidth=4)
            for (rr, cc) in captured:
                self.cells[rr][cc].config(relief="solid", borderwidth=4)
        if self.game.is_terminal(state):
            blue_count = state.count("Blue")
            red_count = state.count("Red")
            winner = "Blue" if blue_count > red_count else "Red"
            self.status_label.config(text=f"Vincitore: {winner} (Blue: {blue_count}, Red: {red_count})")
        else:
            self.status_label.config(text="Turno: " + state.to_move)
            self.info_label.config(
                text=f"Blue: {self.ai_names.get('Blue', 'Human')} | Red: {self.ai_names.get('Red', 'Human')}"
            )

    def cell_clicked(self, r, c):
        if self.waiting_for_human:
            if self.capture_selection_mode:
                allowed = set()
                for move in self.pending_candidate_moves:
                    allowed.update(move[2])
                if (r, c) in allowed:
                    if (r, c) in self.selected_capture_cells:
                        self.selected_capture_cells.remove((r, c))
                    else:
                        self.selected_capture_cells.add((r, c))
                    self.update_board()
            else:
                state = self.current_state()
                legal_moves = self.game.actions(state)
                candidate_moves = [move for move in legal_moves if move[0] == (r, c)]
                if not candidate_moves:
                    return
                if len(candidate_moves) == 1:
                    self.human_move = candidate_moves[0]
                    self.waiting_for_human = False
                else:
                    self.capture_selection_mode = True
                    self.pending_placement = (r, c)
                    self.pending_candidate_moves = candidate_moves
                    self.selected_capture_cells = set()
                    self.update_board()
                    self.confirm_button.grid(row=0, column=3, padx=5)

    def confirm_capture(self):
        for move in self.pending_candidate_moves:
            if set(move[2]) == self.selected_capture_cells:
                self.human_move = move
                self.waiting_for_human = False
                break
        if self.human_move is None:
            messagebox.showerror("Errore", "Selezione non valida. Riprova.")
            return
        self.capture_selection_mode = False
        self.pending_placement = None
        self.pending_candidate_moves = []
        self.selected_capture_cells = set()
        self.confirm_button.grid_forget()
        self.update_board()

    def prev_move(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_board()

    def next_move(self):
        if self.current_index < len(self.state_history) - 1:
            self.current_index += 1
            self.update_board()
        else:
            if (self.player_types.get("Blue") == "ai" and self.player_types.get("Red") == "ai") and not self.auto_mode:
                if not self.game.is_terminal(self.current_state()):
                    self.play_turn()
                    self.update_board()

    def toggle_auto(self):
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            self.auto_button.config(text="Auto ON", highlightbackground="green")
            threading.Thread(target=self.auto_play, daemon=True).start()
        else:
            self.auto_button.config(text="Auto", highlightbackground="red")

    def auto_play(self):
        while self.auto_mode and not self.game.is_terminal(self.state_history[-1]):
            self.play_turn()
            time.sleep(0.5)
        if self.game.is_terminal(self.state_history[-1]):
            self.show_game_over()

    def play_turn(self):
        state = self.state_history[-1]
        if self.game.is_terminal(state):
            return
        current_player = state.to_move
        legal_moves = self.game.actions(state)
        move = None
        if self.player_types[current_player] == "ai":
            if current_player == "Blue":
                future = self.executor.submit(playerBmodule.playerStrategy, self.game, state)
            else:
                future = self.executor.submit(playerRmodule.playerStrategy, self.game, state)
            try:
                move = future.result(timeout=self.time_out)
            except concurrent.futures.TimeoutError:
                print(f"[TIMEOUT] {current_player} ha sforato i {self.time_out} secondi. Esegue mossa random.")
                future.cancel()
                move = random.choice(legal_moves)

                future.cancel()
                move = random.choice(legal_moves)
        else:
            self.waiting_for_human = True
            self.human_move = None
            while self.waiting_for_human:
                self.root.update()
                time.sleep(0.1)
            move = self.human_move
            self.update_board()
        new_state = self.game.result(state, move)
        self.state_history.append(new_state)
        self.current_index = len(self.state_history) - 1
        self.update_board()
        if self.game.is_terminal(new_state):
            self.show_game_over()

    def restart_game(self, dialog):
        dialog.destroy()
        from CephalopodGame import GameSetupDialog  # evita circular import
        available_players = [

         "player.playerMinimax",
         "player.playerSmartLookahead5",
         "player.playerResilientMinimax",
         "player.playerTunableResilient",
         "player.44player",
         "player.mcts_player",
         "player.advPlayer",
         "player.alphabeta_player",
         "player.merge_player",
         "player.self_minimax",
         "player.zob"
        ]


        setup = GameSetupDialog(self.root, available_players)
        config = setup.result
        first = simpledialog.askstring("Chi inizia?", "Blue o Red?", parent=self.root)
        first = first.capitalize() if first in ["Blue", "Red"] else "Blue"

        self.player_types = {color: ("ai" if config[color] != "human" else "human") for color in ["Blue", "Red"]}
        self.ai_names = {color: config[color] if config[color] != "human" else "Human" for color in ["Blue", "Red"]}

        global playerBmodule, playerRmodule
        playerBmodule = importlib.import_module(config["Blue"]) if config["Blue"] != "human" else None
        playerRmodule = importlib.import_module(config["Red"]) if config["Red"] != "human" else None

        self.game = CephalopodGame(size=5, first_player=first)
        self.state_history = [self.game.initial]
        self.current_index = 0
        self.update_board()

    def show_game_over(self):
        state = self.state_history[-1]
        blue_count = state.count("Blue")
        red_count = state.count("Red")
        winner = "Blue" if blue_count > red_count else "Red"

        dialog = tk.Toplevel(self.root)
        dialog.title("Partita Terminata")
        dialog.geometry("300x200")

        msg = f"Vincitore: {winner}\nBlue: {blue_count} dadi\nRed: {red_count} dadi"
        tk.Label(dialog, text=msg, font=("Helvetica", 12), pady=20).pack()

        restart_btn = tk.Button(dialog, text="Restart", font=("Helvetica", 12),
                                command=lambda: self.restart_game(dialog))
        restart_btn.pack(pady=5)

        exit_btn = tk.Button(dialog, text="Esci", font=("Helvetica", 12),
                             command=self.root.destroy)
        exit_btn.pack(pady=5)

        dialog.transient(self.root)
        dialog.grab_set()
        self.root.wait_window(dialog)

    def run_game_loop(self):
        if not (self.player_types.get("Blue") == "ai" and self.player_types.get("Red") == "ai" and not self.auto_mode):
            def loop():
                while not self.game.is_terminal(self.state_history[-1]):
                    self.play_turn()
                    time.sleep(0.1)

            threading.Thread(target=loop, daemon=True).start()
        self.root.mainloop()


def main():
    root = tk.Tk()
    root.withdraw()
    available_players = [
        "playerExampleRandom",
        "player.playerMinimax",
        "player.playerSmartLookahead5",
        "player.playerResilientMinimax",
        "player.playerTunableResilient",
        "player.44player",
        "player.mcts_player",
        "player.advPlayer",
        "player.alphabeta_player",
        "player.merge_player",
        "player.self_minimax",
        "player.zob",
        "player.alphabeta_player2",
        "player.alphabeta_player3"

    ]

    setup_dialog = GameSetupDialog(root, available_players)
    config = setup_dialog.result

    first = simpledialog.askstring("Primo giocatore", "Chi inizia? (Blue o Red):", parent=root)
    first = first.capitalize() if first else "Blue"
    if first not in ["Red", "Blue"]:
        first = "Blue"

    player_types = {}
    ai_modules = {}
    ai_names = {}

    for color in ["Blue", "Red"]:
        if config[color] == "human":
            player_types[color] = "human"
            ai_names[color] = "Human"
        else:
            player_types[color] = "ai"
            ai_names[color] = config[color]
            ai_modules[color] = importlib.import_module(config[color])

    root.destroy()

    global playerBmodule, playerRmodule
    playerBmodule = ai_modules.get("Blue", None)
    playerRmodule = ai_modules.get("Red", None)

    game = CephalopodGame(size=5, first_player=first)
    gui = CephalopodGUI(game, player_types, ai_names, time_out=3)
    gui.run_game_loop()


# Alla fine del file
if __name__ == '__main__':
    main()
