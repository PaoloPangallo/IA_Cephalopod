import time
from cephalopod.game_modes.cephalopod_game_dynamic import CephalopodGameDynamic
from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5
import torch
from torch import nn
import os


class BehaviorCloningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class BCPolicyPlayer:
    def __init__(self, model_path="policy_bc.pt", device=None):
        self.model = BehaviorCloningModel()
        abs_path = os.path.join(os.path.dirname(__file__), model_path)
        self.model.load_state_dict(torch.load(abs_path, map_location=device or "cpu"))
        self.model.eval()
        self.device = device or "cpu"

    def encode_board(self, board):
        board_tensor = torch.zeros((5, 5, 2), dtype=torch.float32)
        for r in range(5):
            for c in range(5):
                cell = board.grid[r][c]
                if cell:
                    board_tensor[r, c, 0] = 1 if cell.color == "B" else 2
                    board_tensor[r, c, 1] = cell.top_face
        return board_tensor

    def choose_move(self, board, color):
        with torch.no_grad():
            encoded = self.encode_board(board).unsqueeze(0).to(self.device)
            pred = self.model(encoded)[0]
            r, c, top_face = pred.round().int().tolist()

            if not board.in_bounds(r, c) or board.grid[r][c] is not None:
                for (rr, cc) in board.get_empty_cells():
                    return (rr, cc, 1, [])

            from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
            captured = []
            capturing_options = find_capturing_subsets(board, r, c)
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                if sum_pips == top_face:
                    captured = subset

            return (r, c, top_face, captured)


def evaluate(num_matches=20, alternate_colors=True):
    wins = {"BC": 0, "Expert": 0, "Draws": 0}

    bc_player = BCPolicyPlayer()
    expert = SmartLookaheadStrategy5()

    for i in range(num_matches):
        print(f"\n[INFO] Match {i + 1}/{num_matches} in corso...")

        if alternate_colors and i % 2 == 0:
            player_B = bc_player
            player_W = expert
            bc_color = "B"
        else:
            player_B = expert
            player_W = bc_player
            bc_color = "W"

        game = CephalopodGameDynamic(player_B, player_W)

        start = time.time()
        winner = game.simulate_game2()
        duration = time.time() - start


        if winner == bc_color:
            wins["BC"] += 1
        elif winner == ("W" if bc_color == "B" else "B"):
            wins["Expert"] += 1
        else:
            wins["Draws"] += 1

    print("\n=== RISULTATI FINALI ===")
    print(f"Behavior Cloning wins: {wins['BC']}")
    print(f"Expert Strategy wins: {wins['Expert']}")


if __name__ == "__main__":
    evaluate(num_matches=2000)
