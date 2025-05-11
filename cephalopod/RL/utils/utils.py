import csv
import os
from datetime import datetime


def log_training(path, episode, win, final_reward):
    header = ["Episode", "Win", "FinalReward"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([episode, win, final_reward])


def maybe_decay_exploration(agent, min_exp=0.05, decay_factor=0.9):
    if agent.exp_rate > min_exp:
        agent.exp_rate *= decay_factor


def get_policy_name(name, reward_shaper):
    base = reward_shaper.__class__.__name__.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"policy_{name}_{base}_{timestamp}.pkl"
