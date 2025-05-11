import random

from cephalopod.core.board import Board, Die
from cephalopod.core.mechanics import get_opponent


class CephalopodGameDynamic:
    def __init__(self, strategy_B, strategy_W, max_dice_per_player=24):

        self.board = Board()
        self.strategy = {"B": strategy_B, "W": strategy_W}
        self.max_dice_per_player = max_dice_per_player
        self.dice_remaining = {"B": max_dice_per_player, "W": max_dice_per_player}
        self.current_player = "B"
        self.moves_log = []
        self.move_num = 1

    def simulate_move(self):
        empty_cells = self.board.get_empty_cells()
        if not empty_cells:
            return False

        # Seleziona la strategia in base al giocatore corrente
        strategy = self.strategy[self.current_player]
        move = strategy.choose_move(self.board, self.current_player)
        if move is None:
            return False
        r, c, top_face, captured = move

        # Rimuove eventuali dadi catturati
        for (rr, cc) in captured:
            self.board.grid[rr][cc] = None

        # Posiziona il nuovo dado
        new_die = Die(self.current_player, top_face)
        self.board.place_die(r, c, new_die)
        self.dice_remaining[self.current_player] -= 1

        self.moves_log.append({
            "move_num": self.move_num,
            "player": self.current_player,
            "row": r,
            "col": c,
            "top_face": top_face,
            "captured": captured
        })
        self.move_num += 1
        self.current_player = get_opponent(self.current_player)
        return True

    def simulate_game(self):
        while not self.board.is_full():
            if not self.simulate_move():
                break

        # Conta i dadi presenti per determinare il vincitore
        b_count = sum(1 for r in range(self.board.size) for c in range(self.board.size)
                      if self.board.grid[r][c] is not None and self.board.grid[r][c].color == "B")
        w_count = sum(1 for r in range(self.board.size) for c in range(self.board.size)
                      if self.board.grid[r][c] is not None and self.board.grid[r][c].color == "W")
        winner = "B" if b_count > w_count else "W"
        self.moves_log.append({
            "move_num": self.move_num,
            "player": "END",
            "row": -1,
            "col": -1,
            "top_face": -1,
            "captured": f"FinalCount => B:{b_count}, W:{w_count}"
        })
        self.moves_log.append({
            "move_num": self.move_num + 1,
            "player": "WINNER",
            "row": -1,
            "col": -1,
            "top_face": -1,
            "captured": winner
        })
        return self.moves_log

    def simulate_game2(self):
        while not self.board.is_full():
            if not self.simulate_move():
                break

        # Conta i dadi presenti per determinare il vincitore
        b_count = sum(1 for r in range(self.board.size) for c in range(self.board.size)
                      if self.board.grid[r][c] is not None and self.board.grid[r][c].color == "B")
        w_count = sum(1 for r in range(self.board.size) for c in range(self.board.size)
                      if self.board.grid[r][c] is not None and self.board.grid[r][c].color == "W")
        winner = "B" if b_count > w_count else "W"
        self.moves_log.append({
            "move_num": self.move_num,
            "player": "END",
            "row": -1,
            "col": -1,
            "top_face": -1,
            "captured": f"FinalCount => B:{b_count}, W:{w_count}"
        })
        self.moves_log.append({
            "move_num": self.move_num + 1,
            "player": "WINNER",
            "row": -1,
            "col": -1,
            "top_face": -1,
            "captured": winner
        })
        return winner


if __name__ == "__main__":
    # Per test, chiediamo all'utente quale strategia usare per ciascun giocatore
    print("Scegli strategia per il giocatore B:")
    print("1 - Naive")
    print("2 - Heuristic")
    choice_B = input("Inserisci scelta per B (1 o 2): ").strip()

    # Import delle strategie dal package strategies
    from strategies import NaiveStrategy, HeuristicStrategy

    strategy_B = HeuristicStrategy() if choice_B == "2" else NaiveStrategy()

    print("Scegli strategia per il giocatore W:")
    print("1 - Naive")
    print("2 - Heuristic")
    choice_W = input("Inserisci scelta per W (1 o 2): ").strip()
    strategy_W = HeuristicStrategy() if choice_W == "2" else NaiveStrategy()

    game = CephalopodGameDynamic(strategy_B, strategy_W, max_dice_per_player=24)
    log = game.simulate_game()
    for move in log:
        print(move)
