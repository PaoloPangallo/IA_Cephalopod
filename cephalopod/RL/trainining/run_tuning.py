import os
from itertools import product

from cephalopod.RL.base_rl_player import RLPlayer
from cephalopod.RL.reward_shaping.rewardshaping import BasicShaper, RiskAwareShaper
from cephalopod.RL.trainining.training import train_rl_agent
from cephalopod.strategies.smart_position import SmartPositionalLookaheadStrategy


def get_shaper_by_name(name):
    if name == "basic":
        return BasicShaper()
    elif name == "riskaware":
        return RiskAwareShaper()
    else:
        raise ValueError(f"Reward shaper '{name}' non valido")


# === CONFIGURAZIONE ESPERIMENTI ===
learning_rates = [0.1, 0.2]
gammas = [0.9, 0.95]
shapers = ["basic", "riskaware"]
episodes = 3000
opponent = SmartPositionalLookaheadStrategy()
save_dir = "policies/tuning_runs"
log_dir = "logs/tuning_runs"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


# === LOOP ESPERIMENTALE ===
for lr, gamma, shaper_name in product(learning_rates, gammas, shapers):
    shaper = get_shaper_by_name(shaper_name)
    agent_name = f"RL_lr{lr}_g{gamma}_{shaper_name}"

    rl = RLPlayer(
        name=agent_name,
        lr=lr,
        gamma=gamma,
        reward_shaper=shaper,
    )

    log_path = os.path.join(log_dir, f"{agent_name}.csv")

    print(f"\n🚀 Inizio tuning: {agent_name}")

    train_rl_agent(
        rl_agent=rl,
        opponent_agent=opponent,
        episodes=episodes,
        save_dir=save_dir,
        log_path=log_path,
        decay_exploration=True,
        verbose=True,
    )
