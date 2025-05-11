import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc_policy = nn.Linear(64 * 5 * 5, 25)
        self.fc_value = nn.Linear(64 * 5 * 5, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        policy = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))
        return policy, value

    def predict(self, x, legal_moves):
        self.eval()
        with torch.no_grad():
            policy_logits, value = self(x.unsqueeze(0))
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

            masked_policy = np.array([
                policy[r * 5 + c] if (r, c) in legal_moves else 0 for r in range(5) for c in range(5)
            ], dtype=np.float32)
            masked_policy_sum = masked_policy.sum()
            if masked_policy_sum > 0:
                masked_policy /= masked_policy_sum
            else:
                masked_policy = np.ones(25, dtype=np.float32) / 25

            return masked_policy, value.item()
