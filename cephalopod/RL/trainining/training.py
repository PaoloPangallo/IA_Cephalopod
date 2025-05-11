import os
import matplotlib.pyplot as plt

from cephalopod.RL.utils.utils import maybe_decay_exploration, get_policy_name, log_training
from cephalopod.game_modes.cephalopod_game_dynamic import CephalopodGameDynamic


def train_rl_agent(
        rl_agent,
        opponent_agent,
        episodes=10000,
        save_every=1000,
        log_path="training_log.csv",
        save_dir="policies",
        decay_exploration=True,
        verbose=True,
        use_fixed_save=False,
        smart_decay_callback=None  # 🧠 se vuoi passare una funzione di decay intelligente
):
    os.makedirs(save_dir, exist_ok=True)
    win_rate_over_time = []
    avg_reward_over_time = []
    log_full = []
    episode_rewards = []  # 🧠 lista reward shaping episodio per episodio

    for i in range(episodes):
        rl_color = "B" if i % 2 == 0 else "W"
        # Se opponent_agent è dinamico, scegli l'avversario giusto e scala la difficoltà
        if hasattr(opponent_agent, "choose_opponent"):
            opponent_agent.scale_difficulty(i)
            current_opponent = opponent_agent.choose_opponent()
        else:
            current_opponent = opponent_agent

        game = CephalopodGameDynamic(
            strategy_B=rl_agent if rl_color == "B" else current_opponent,
            strategy_W=rl_agent if rl_color == "W" else current_opponent
        )

        log = game.simulate_game()
        winner = log[-1]["captured"]

        b_count = sum(1 for r in range(game.board.size) for c in range(game.board.size)
                      if game.board.grid[r][c] and game.board.grid[r][c].color == "B")
        w_count = sum(1 for r in range(game.board.size) for c in range(game.board.size)
                      if game.board.grid[r][c] and game.board.grid[r][c].color == "W")
        total = b_count + w_count
        final_reward = b_count / total if rl_color == "B" else w_count / total
        if winner != rl_color:
            final_reward -= 1.0

        win = 1 if winner == rl_color else 0
        rl_agent.feed_reward(final_reward)
        shaped_total = sum(r for _, r in rl_agent.states)

        rl_agent.reset()

        log_training(log_path, i + 1, win, final_reward)
        log_full.append(win)

        # 🧠 Salva la somma dei reward shaping dell’episodio corrente
        episode_rewards.append(shaped_total)

        if (i + 1) % 50 == 0:
            win_rate = sum(log_full[-50:]) / 50
            avg_shaped_reward = sum(episode_rewards[-50:]) / 50
            win_rate_over_time.append(win_rate)
            avg_reward_over_time.append(avg_shaped_reward)

            if verbose:
                print(f"[{i + 1}/{episodes}] Win rate ultimi 50: {win_rate:.2f} | Avg reward: {avg_shaped_reward:.2f} | ε={rl_agent.exp_rate:.2f}")

            if smart_decay_callback:
                smart_decay_callback(rl_agent, win_rate)

        if (i + 1) % save_every == 0:
            if decay_exploration and not smart_decay_callback:
                maybe_decay_exploration(rl_agent)
            if use_fixed_save:
                rl_agent.save_policy()
            else:
                filename = get_policy_name(rl_agent.name, rl_agent.reward_shaper)
                rl_agent.save_policy(os.path.join(save_dir, filename))

    # Save finale
    if use_fixed_save:
        rl_agent.save_policy()
    else:
        rl_agent.save_policy(os.path.join(save_dir, get_policy_name(rl_agent.name, rl_agent.reward_shaper)))

    print(f"\n🌝 Training completato.")

    step = 50
    x_axis = range(step, episodes + 1, step)

    plt.figure()
    plt.plot(x_axis, win_rate_over_time, label="Win Rate", marker='o')
    plt.plot(x_axis, avg_reward_over_time, label="Avg Reward", marker='x')
    plt.xlabel("Partite giocate")
    plt.ylabel("Valore")
    plt.title(f"Performance RL: {rl_agent.name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"[WARN] Impossibile visualizzare il grafico: {e}")
