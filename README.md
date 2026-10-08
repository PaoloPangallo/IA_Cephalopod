# Cephalopod AI — Search, Reinforcement Learning & Neural Agents

**A multi-paradigm AI project for the Cephalopod strategy game**

This repository explores several approaches to game-playing artificial intelligence on the same environment, ranging from handcrafted heuristics and adversarial search to reinforcement learning, imitation learning and an AlphaZero-style neural agent.

**Motivation:** how does action selection change when the same strategic decision is approached through rules, adversarial search, experience and expert demonstrations? The repository explores that question through multiple AI agent families; it does not claim a single controlled performance ranking across all of them.

### Architecture and project walkthrough

- [System design: motivation, components, algorithms and two editable Mermaid diagrams](docs/SYSTEM_DESIGN.md)
- [Guida in italiano: spiegazione semplice, presentazione da colloquio e domande tecniche](docs/PROJECT_WALKTHROUGH_IT.md)

The diagrams document the main `cephalopod/` experiments while distinguishing the separate `ia_scarc/` implementation. All original algorithms, experiments and saved results remain available.

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

## Core-game and AlphaZero correctness checks

This project includes research explorations and prototype agents. Recent fixes
introduced regression coverage for the core engine, corrected package imports
and updated parts of the AlphaZero-style pipeline. The main research
implementations and stored artifacts are preserved; **existing checkpoint results
are not claimed to have been retrained or revalidated**.

Run the deterministic rule and search smoke tests from the repository root:

```bash
python -m unittest discover -s tests -v
python -m cephalopod.game_modes.cephalopod_game_dynamic
```

Other entry points (desktop GUI required for Tkinter):

```bash
python -m cephalopod.ui.ui
python -m cephalopod.alphazero.main
```

The graphical bracket renderer requires both the `graphviz` Python package and
the Graphviz system executable. The comparative simulation utilities use `pandas`.
Not all historical scripts, experiments, or pretrained weights are covered by
these tests. In particular, the `ia_scarc/` tree is kept as a separate original
implementation, not silently merged with the main game engine.

### Rules and evaluation caveats

- Capturing uses the **sum of the captured dice pips**, up to six; the old
  AlphaZero-style path mistakenly used `6 - sum`.
- MCTS backs up values from the current player's perspective; selection adjusts
  the child's perspective, and terminal nodes use the actual game score.
- Neural training now uses the full MCTS target distribution instead of an
  argmax-only target. Historical weights may reflect the old objective.
- A stopped game with equal piece counts is reported as `DRAW`, not
  automatically won by White. A tournament may still have its own tie-break
  policy for advancing a player.
- The originally reported evaluation numbers are **historical** until
  rerunning controlled matches with a fixed seed and reproducible checkpoints.
