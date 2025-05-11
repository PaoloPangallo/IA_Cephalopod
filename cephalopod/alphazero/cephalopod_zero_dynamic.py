import numpy as np
import torch

from cephalopod.alphazero.mcts import MCTS

class CephalopodZero:
    def __init__(self, model, mcts_simulations=50):
        self.model = model
        self.mcts = MCTS(self, num_simulations=mcts_simulations)

    def encode_board(self, board, player):
        import numpy as np
        planes = np.zeros((3, 5, 5), dtype=np.float32)
        for r in range(5):
            for c in range(5):
                die = board.grid[r][c]
                if die:
                    if die.color == player:
                        planes[0][r][c] = 1
                        planes[1][r][c] = die.top_face / 6
                    else:
                        planes[2][r][c] = die.top_face / 6
        return torch.tensor(planes, dtype=torch.float32)

    def generate_self_play_data(self, num_games=10):
        data = []
        for _ in range(num_games):
            data.extend(self.play_game())
        return data

    def play_game(self):
        from cephalopod.core.board import Board
        from cephalopod.core.mechanics import get_opponent
        import copy

        board = Board()
        data = []
        current_player = "B"
        while not board.is_full():
            board_input = self.encode_board(board, current_player)
            policy = self.mcts.run(copy.deepcopy(board), current_player)
            data.append((board_input.numpy(), policy, 0))  # target value sarà corretto a fine partita

            legal_moves = board.get_empty_cells()
            flat_probs = [policy[r * 5 + c] if (r, c) in legal_moves else 0 for r in range(5) for c in range(5)]
            move_index = np.random.choice(25, p=np.array(flat_probs) / sum(flat_probs))
            r, c = divmod(move_index, 5)

            from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
            from cephalopod.core.board import Die

            captured, sum_pips = choose_capturing_subset(find_capturing_subsets(board, r, c))
            top_face = 6 - sum_pips if captured else 1
            for rr, cc in (captured or []):
                board.grid[rr][cc] = None
            board.place_die(r, c, Die(current_player, top_face))
            current_player = get_opponent(current_player)

        final_value = self.evaluate_winner(board)
        for i in range(len(data)):
            state, policy, _ = data[i]
            data[i] = (state, policy, final_value if i % 2 == 0 else -final_value)
        return data

    def evaluate_winner(self, board):
        b_count = sum(1 for row in board.grid for die in row if die and die.color == "B")
        w_count = sum(1 for row in board.grid for die in row if die and die.color == "W")
        return 1 if b_count > w_count else -1
