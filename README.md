# Cephalopod AI — Search, Reinforcement Learning & Neural Agents

**A multi-paradigm AI project for the Cephalopod strategy game**

This repository explores several approaches to game-playing artificial intelligence on the same environment, ranging from handcrafted heuristics and adversarial search to reinforcement learning, imitation learning and an AlphaZero-style neural agent.

The project is useful as a comparison of **how different AI paradigms represent, search and learn strategies for the same discrete decision problem**.

## What is implemented

### Classical game AI
The repository contains multiple search- and heuristic-based players, including greedy heuristics, lookahead strategies, Minimax, Alpha-Beta variants, Monte Carlo Tree Search and hybrid strategies.

### Hyperparameter optimization
Search-based agents include tunable evaluation functions. **Optuna** is used to optimize heuristic weights against baseline opponents through repeated matches.

### Reinforcement Learning
The `cephalopod/RL/` package contains a reinforcement-learning player with configurable reward shaping, including basic, risk-aware and aggressive variants. Large learned policy tables are generated locally and intentionally excluded from Git.

### Behavior Cloning
The `cephalopod/clon/` experiments implement imitation learning from an expert strategy: expert-game generation, state/action encoding, PyTorch training and evaluation against search-based opponents.

The trained inference model `policy_bc.pt` is retained, while the large generated expert dataset is not.

### AlphaZero-style agent
The `cephalopod/alphazero/` implementation combines a convolutional policy/value network, Monte Carlo Tree Search, self-play and neural training.

## Repository structure

```text
IA_Cephalopod/
├── cephalopod/
│   ├── core/
│   ├── strategies/
│   ├── game_modes/
│   ├── simulazioni/
│   ├── RL/
│   ├── clon/
│   ├── alphazero/
│   └── ui/
├── ia_scarc/
│   └── mAIN/
├── requirements.txt
└── README.md
```

The repository contains two complementary implementations developed during the project. `cephalopod/` is the main experimental framework, while `ia_scarc/` contains additional search, player and Optuna-tuning experiments.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The graphical interfaces use **Tkinter**, included with standard Python installations on Windows. Some Linux distributions require the system `python3-tk` package.

## Example entry points

AlphaZero-style demo:

```bash
python -m cephalopod.alphazero.main
```

AlphaZero-style self-play training:

```bash
python -m cephalopod.alphazero.train_cephalopod_zero
```

The reinforcement-learning, tournament and behavior-cloning packages contain additional training and evaluation scripts.

## Repository hygiene

Large generated artifacts are deliberately not tracked: RL policy tables, expert datasets, training logs, optimizer state, Optuna SQLite studies and Python/IDE caches.

Small inference checkpoints used by the neural-agent demos are retained so the trained agents can still be inspected without committing the much larger training datasets.

## Why this project matters

The project shows a progression from **explicit search and heuristics** to **learning-based agents** within one common environment.

It makes it possible to compare trade-offs between handcrafted evaluation and learned policies, search depth and computational cost, exploration and exploitation, imitation from a strong expert, and self-play with neural policy/value estimation.

## Author

**Paolo Pangallo**  
M.Sc. Computer Engineering — Artificial Intelligence  
University of Calabria
