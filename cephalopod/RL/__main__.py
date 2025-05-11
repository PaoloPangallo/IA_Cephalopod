import argparse
import os

from cephalopod.RL.base_rl_player import RLPlayer
from cephalopod.RL.reward_shaping.rewardshaping import BasicShaper, RiskAwareShaper, AggressiveBoardShaper, \
    AggressiveBoardShaper2, AdvancedBoardShaper, OpponentManager
from cephalopod.RL.trainining.training import train_rl_agent
from cephalopod.strategies import AggressiveStrategy
from cephalopod.strategies.deep_thinking import ExpectimaxStrategy
from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5
from cephalopod.strategies.smart_position import SmartPositionalLookaheadStrategy
from strategies import NaiveStrategy


def get_shaper(name):
    if name == "basic":
        return BasicShaper()
    elif name == "riskaware":
        return RiskAwareShaper()
    elif name == "aggressive":
        return AggressiveBoardShaper()
    elif name == "strategic":
        return AggressiveBoardShaper2()
    elif name == "advanced":
        return AdvancedBoardShaper()
    else:
        raise ValueError(f"Reward shaper '{name}' non valido.")


def smart_decay_exploration(agent, win_rate):
    if win_rate >= 0.6:
        agent.exp_rate = max(agent.exp_rate * 0.9, 0.05)  # Riduci ε se l'agente è bravo
    elif win_rate < 0.3:
        agent.exp_rate = min(agent.exp_rate * 10, 0.3)   # Ri-aumenta ε se performa male


def main():
    parser = argparse.ArgumentParser(description="Training RLPlayer su Cephalopod")
    parser.add_argument("--name", type=str, default="RL")
    parser.add_argument("--episodes", type=int, default=25000)
    parser.add_argument("--opponent", type=str, default="smartpos")
    parser.add_argument("--save-dir", type=str, default="policies")
    parser.add_argument("--log-path", type=str, default="training_log.csv")
    parser.add_argument("--shaper", type=str, default="basic",
                        choices=["basic", "riskaware", "aggressive", "strategic", "advanced"])

    if len(os.sys.argv) == 1:
        print("\n🔧 Nessun argomento passato. Scegli uno shaper:")
        print("1. basic\n2. riskaware\n3. aggressive\n4. strategic\n5. advanced")
        choice = input("Inserisci il numero: ").strip()
        shaper_map = {
            "1": "basic",
            "2": "riskaware",
            "3": "aggressive",
            "4": "strategic",
            "5": "advanced"
        }
        selected = shaper_map.get(choice, "basic")
        args = parser.parse_args(["--shaper", selected])

    else:
        args = parser.parse_args()

    shaper = get_shaper(args.shaper)
    policy_path = os.path.join(args.save_dir, f"policy_{args.name}_{args.shaper}.pkl")

    rl = RLPlayer(name=args.name, reward_shaper=shaper, policy_path=policy_path)

    try:
        rl.load_policy()
        print(f"[LOAD] Policy caricata da {rl.policy_path}")
        # ε non viene modificato qui, sarà gestito dinamicamente nel training
    except FileNotFoundError:
        print(f"[WARN] Nessuna policy trovata, training da zero.")

    opponent = SmartLookaheadStrategy5()

    train_rl_agent(
        rl_agent=rl,
        opponent_agent=opponent,
        episodes=args.episodes,
        save_dir=args.save_dir,
        log_path=args.log_path,
        use_fixed_save=True,
    )


if __name__ == "__main__":
    main()
