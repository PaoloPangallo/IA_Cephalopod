import pickle
import copy
from cephalopod.core.board import Board, Die
from cephalopod.game_modes.cephalopod_game_dynamic import CephalopodGameDynamic
from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5 as TacticalExpert


def encode_board(board):
    return str([[str(cell) if cell else "" for cell in row] for row in board.grid])


def generate_expert_dataset(num_games=100, save_path="expert_dataset.pkl"):
    expert = TacticalExpert()
    data = []

    for i in range(num_games):
        game = CephalopodGameDynamic(expert, expert)
        board = game.board
        color = "B"

        while not board.is_full():
            move = expert.choose_move(board, color)
            state = encode_board(copy.deepcopy(board))
            data.append((state, move))
            r, c, top_face, captured = move
            for (rr, cc) in captured:
                board.grid[rr][cc] = None
            board.place_die(r, c, Die(color, top_face))
            color = "W" if color == "B" else "B"


    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"[✓] Dataset salvato in '{save_path}'")


if __name__ == "__main__":
    generate_expert_dataset(num_games=4000)
