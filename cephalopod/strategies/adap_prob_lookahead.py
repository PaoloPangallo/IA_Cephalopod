import copy
import random
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import json
from cephalopod.core.board import Die
from cephalopod.core.mechanics import find_capturing_subsets
from cephalopod.game_modes.cephalopod_game_dynamic import CephalopodGameDynamic
from cephalopod.strategies import SmartLookaheadStrategy, HeuristicStrategy, NaiveStrategy
from cephalopod.strategies.smart_minimax import SmartMinimaxStrategy

MIN_WEIGHTS = {
    "six_bonus": 100,
    "low_capture_opportunity_bonus": 0.1,
    "center_bonus_weight": 0.1,
    "exposure_penalty_weight": 0.01,
    "opponent_response_penalty_weight": 0.01
}

class AdaptiveProbabilisticLookaheadStrategy:
    def __init__(self, initial_config=None, learning_rate=0.1):
        self.name = "AdaptiveProb"
        self.learning_rate = learning_rate
        self.config = initial_config or {
            "six_bonus": 200,
            "low_capture_opportunity_bonus": 2,
            "center_bonus_weight": 1.0,
            "exposure_penalty_weight": 0.2,
            "opponent_response_penalty_weight": 100
        }

    def choose_move(self, board, color):
        empty_cells = board.get_empty_cells()
        best_score = float("-inf")
        best_move = None

        for (r, c) in empty_cells:
            capturing_options = find_capturing_subsets(board, r, c)
            for subset, sum_pips in capturing_options or [([], 1)]:
                move = (r, c, sum_pips, subset)
                score = self.evaluate_move(board, move, color)
                if score > best_score:
                    best_score = score
                    best_move = move

        return best_move

    def evaluate_move(self, board, move, color):
        r, c, top_face, captured = move
        reward = sum(board.grid[rr][cc].top_face for (rr, cc) in captured) if captured else 0

        bonus_six = self.config["six_bonus"] if top_face == 6 else 0
        bonus_low = self.config["low_capture_opportunity_bonus"] if 0 < top_face < 6 else 0

        rows = len(board.grid)
        cols = len(board.grid[0]) if board.grid else 0

        center_r, center_c = (rows - 1) / 2, (cols - 1) / 2
        dist = abs(r - center_r) + abs(c - center_c)
        max_dist = center_r + center_c
        center_bonus = (1 - dist / max_dist) * self.config["center_bonus_weight"]

        exposure = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if board.grid[nr][nc] is None:
                    exposure += 1
        exposure_penalty = exposure * self.config["exposure_penalty_weight"]

        return reward + bonus_six + bonus_low + center_bonus - exposure_penalty

    def update_weights(self, reward):
        if reward == 0:
            return
        for k in self.config:
            noise = np.random.uniform(-0.01, 0.01)
            if reward > 0:
                self.config[k] *= (1 + self.learning_rate + noise)
            else:
                self.config[k] *= (1 - self.learning_rate + noise)
            self.config[k] = max(self.config[k], MIN_WEIGHTS[k])


def train_adaptive_strategy(episodes_per_opponent=200):
    opponents = [
        ("SmartLookahead", SmartLookaheadStrategy),
        ("SmartMinimax", SmartMinimaxStrategy),
        ("Heuristic", HeuristicStrategy),
        ("Naive", NaiveStrategy)
    ]

    strategy = AdaptiveProbabilisticLookaheadStrategy()
    history = []
    best_cumulative = float("-inf")
    cumulative_reward = 0
    best_config = strategy.config.copy()
    consecutive_losses = 0
    reset_threshold = 15

    with open("adaptive_log.csv", "w", newline="") as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["Episode", "Reward", "CumulativeReward", "Opponent"] + list(strategy.config.keys()))

        episode_count = 0
        for opp_name, OpponentClass in opponents:
            print(f"\n🧠 Inizio allenamento contro: {opp_name}")
            for ep in range(episodes_per_opponent):
                opponent = OpponentClass()

                if ep % 2 == 0:
                    winner = simulate_match(copy.deepcopy(strategy), copy.deepcopy(opponent))
                    reward = 1 if winner == "B" else -1
                else:
                    winner = simulate_match(copy.deepcopy(opponent), copy.deepcopy(strategy))
                    reward = 1 if winner == "W" else -1

                strategy.update_weights(reward)
                cumulative_reward += reward
                history.append((episode_count + 1, reward))

                print(f"Episode {episode_count+1}: {'WIN' if reward > 0 else 'LOSS'} | Reward: {reward} | Cumulative: {cumulative_reward:.1f}")
                print("  Config:", {k: round(v, 2) for k, v in strategy.config.items()})

                if cumulative_reward > best_cumulative:
                    print("  🔺 Nuovo miglioramento cumulativo!")
                    best_cumulative = cumulative_reward
                    best_config = strategy.config.copy()
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1

                if consecutive_losses >= reset_threshold:
                    print("  🔄 Troppe sconfitte di fila. Reset della strategia.")
                    strategy = AdaptiveProbabilisticLookaheadStrategy()
                    consecutive_losses = 0

                row = [episode_count + 1, reward, cumulative_reward, opp_name] + [round(strategy.config[k], 4) for k in strategy.config]
                writer.writerow(row)
                episode_count += 1

    plot_learning_curve(history)

    with open("adaptive_best_config.json", "w") as f:
        json.dump(best_config, f, indent=4)
    print("\n🧠 Configurazione migliore salvata in adaptive_best_config.json")

    return best_config


def simulate_match(strategy_B, strategy_W):
    game = CephalopodGameDynamic(strategy_B, strategy_W)
    moves_log = game.simulate_game()
    winner_color = moves_log[-1].get("captured", None)
    return winner_color if winner_color in ["B", "W"] else random.choice(["B", "W"])


def plot_learning_curve(history):
    episodes, rewards = zip(*history)
    cum_rewards = [sum(rewards[:i+1]) for i in range(len(rewards))]
    df = pd.DataFrame({"Episode": episodes, "Reward": rewards, "CumulativeReward": cum_rewards})
    df.to_csv("adaptive_curve.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["Episode"], df["CumulativeReward"], marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Adaptive Strategy Learning Curve")
    plt.grid(True)

    plot_path = "adaptive_curve.png"
    plt.savefig(plot_path)
    print(f"\n📈 Learning curve salvata in: {plot_path}")



if __name__ == "__main__":
    final_config = train_adaptive_strategy(episodes_per_opponent=200)
    print("\n🎯 Configurazione finale appresa:")
    for k, v in final_config.items():
        print(f"{k}: {round(v, 2)}")