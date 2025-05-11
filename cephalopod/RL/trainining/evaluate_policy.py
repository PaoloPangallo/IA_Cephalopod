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
        use_fixed_save=False
):
    os.makedirs(save_dir, exist_ok=True)
    win_rate_over_time = []
    log_full = []

    for i in range(episodes):
        rl_color = "B" if i % 2 == 0 else "W"
        game = CephalopodGameDynamic(
            strategy_B=rl_agent if rl_color == "B" else opponent_agent,
            strategy_W=rl_agent if rl_color == "W" else opponent_agent
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
        rl_agent.reset()

        log_training(log_path, i + 1, win, final_reward)
        log_full.append(win)

        if (i + 1) % 50 == 0:
            win_rate = sum(log_full[-50:]) / 50
            win_rate_over_time.append(win_rate)
            if verbose:
                print(f"[{i + 1}/{episodes}] Win rate ultimi 50: {win_rate:.2f} | ε={rl_agent.exp_rate:.2f}")

        if (i + 1) % save_every == 0:
            if decay_exploration:
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

    step = 50  # ogni quanto salvi il win rate
    plt.plot(range(step, episodes + 1, step), win_rate_over_time)
    plt.xlabel("Partite giocate")
    plt.ylabel("Win rate (ultime 100)")
    plt.title(f"Performance RL: {rl_agent.name}")
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"[WARN] Impossibile visualizzare il grafico: {e}")