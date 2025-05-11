import torch
from torch import nn


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
            nn.Linear(128, 3),
            nn.Sigmoid()  # ⬅️ Converte output in range (0, 1)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [B, 2, 5, 5]
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        # ⬇️ Mapping in range desiderato
        r = x[:, 0] * 4  # row: 0–4
        c = x[:, 1] * 4  # col: 0–4
        f = x[:, 2] * 5 + 1  # top_face: 1–6
        return torch.stack([r, c, f], dim=1)
