import torch
from cephalopod.core.board import Die
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.clon.behavior import BehaviorCloningModel
from cephalopod.clon.expert_dataset import ExpertDataset


class BCPolicyPlayer:
    def __init__(self, model_path="policy_bc.pt", device=None):
        self.device = device or "cpu"
        self.model = BehaviorCloningModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.dataset = ExpertDataset("expert_dataset.pkl")

    def encode_board(self, board):
        board_str = str([[str(cell) if cell else "" for cell in row] for row in board.grid])
        return self.dataset.encode_board(board_str)

    def is_move_legal(self, board, move, color):
        r, c, top_face, captured = move
        if not board.in_bounds(r, c):
            return False
        if board.grid[r][c] is not None:
            return False
        if captured:
            sum_pips = sum(board.grid[rr][cc].top_face for rr, cc in captured)
            if sum_pips != top_face:
                return False
        return True

    def choose_move(self, board, color):
        with torch.no_grad():
            encoded = self.encode_board(board).unsqueeze(0).to(self.device)
            pred = self.model(encoded)[0]
            r, c, top_face = pred.round().int().tolist()
            top_face = max(1, min(6, top_face))  # assicurati che sia in [1,6]

            # Prova mossa predetta
            capturing_options = find_capturing_subsets(board, r, c)
            captured = []
            if capturing_options:
                subset, sum_pips = choose_capturing_subset(capturing_options)
                if sum_pips == top_face:
                    captured = subset

            move = (r, c, top_face, captured)
            if self.is_move_legal(board, move, color):
                return move

            # Se non è legale, prova tutte le mosse possibili
            for rr, cc in board.get_empty_cells():
                # 1. Prova cattura
                options = find_capturing_subsets(board, rr, cc)
                if options:
                    subset, sum_pips = choose_capturing_subset(options)
                    move = (rr, cc, sum_pips, subset)
                    if self.is_move_legal(board, move, color):
                        return move
                # 2. Altrimenti mossa base
                move = (rr, cc, 1, [])
                if self.is_move_legal(board, move, color):
                    return move

            # Se non trova nulla (edge case raro)
            return (r, c, 1, [])
