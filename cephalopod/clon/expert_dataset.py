import torch
from torch.utils.data import Dataset
import pickle
import os
import ast
import time


class ExpertDataset(Dataset):
    def __init__(self, path):
        start = time.time()

        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), path)

        with open(path, "rb") as f:
            self.raw_data = pickle.load(f)

        print(f"[✓] Dataset caricato da '{path}' in {time.time() - start:.2f} secondi.")
        print(f"[✓] Numero esempi: {len(self.raw_data)}")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        board_str, move = self.raw_data[idx]
        encoded_board = self.encode_board(board_str)
        encoded_move = self.encode_move(move)
        return encoded_board, encoded_move

    def encode_board(self, board_str):
        board_tensor = torch.zeros((5, 5, 2), dtype=torch.float32)
        try:
            raw = ast.literal_eval(board_str)
        except Exception as e:
            print(f"[ERRORE] Parsing board fallito: {e}")
            return board_tensor

        for r in range(5):
            for c in range(5):
                val = raw[r][c]
                if val:
                    try:
                        color, face = val.strip("() ").split(",")
                        color = color.strip()
                        face = int(face)
                        board_tensor[r, c, 0] = 1 if color == "B" else 2
                        board_tensor[r, c, 1] = min(max(face, 1), 6)  # Clamp pips tra 1 e 6
                    except Exception as e:
                        print(f"[ERRORE] Encoding cella ({r},{c}): {val} — {e}")
                        continue

        return board_tensor

    def encode_move(self, move):
        r, c, face, _ = move
        # Clamp coordinate e top_face in range accettabili
        r = max(0, min(r, 4))
        c = max(0, min(c, 4))
        face = max(1, min(face, 6))
        return torch.tensor([r, c, face], dtype=torch.long)
