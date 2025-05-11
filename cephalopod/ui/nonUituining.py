import csv
import json
import random
from pathlib import Path

from cephalopod.core.board import Board, Die
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.strategies import SmartLookaheadStrategy
from cephalopod.strategies.deep5lookahead import SmartLookaheadStrategy6
from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5
from cephalopod.strategies.tunMinMax import TunableMinimaxStrategy
from cephalopod.strategies.variant2_lookahead2 import Variant2SmartLookahead
from cephalopod.strategies.weird2 import Weird2Strategy

STRATEGIES = {
    "SmartLookahead": SmartLookaheadStrategy,
    "Goat5": SmartLookaheadStrategy5,
    "Goat?": SmartLookaheadStrategy6,
    "V1": Variant2SmartLookahead,
    "we2": Weird2Strategy,
}

BOARD_SIZE = 5

def evaluate_match(tunable, opponent):
    board = Board()
    current_player = "B"

    while not board.is_full():
        strategy = tunable if current_player == "B" else opponent

        legal_moves = []
        for (r, c) in board.get_empty_cells():
            capturing = find_capturing_subsets(board, r, c)
            if capturing:
                subset, sum_pips = choose_capturing_subset(capturing)
                legal_moves.append((r, c, sum_pips, subset))
            else:
                legal_moves.append((r, c, 1, []))

        move = strategy.choose_move(board, current_player)
        if move not in legal_moves:
            move = random.choice(legal_moves)

        r, c, top, captured = move
        for rr, cc in captured:
            board.grid[rr][cc] = None
        board.place_die(r, c, Die(current_player, top))
        current_player = "W" if current_player == "B" else "B"

    # Calcola punteggi
    b_score = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                  if board.grid[r][c] and board.grid[r][c].color == "B")
    w_score = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                  if board.grid[r][c] and board.grid[r][c].color == "W")

    return 1 if b_score > w_score else 0  # 1 se Tunable vince, altrimenti 0

def main():
    # Carica configurazioni
    config_path = Path("tuning_configs_60k.json")
    with open(config_path) as f:
        configs = json.load(f)

    block_size = 100
    total_configs = len(configs)
    perfect_configs_all = []
    results = []
    block_number = 1

    for start in range(0, total_configs, block_size):
        end = min(start + block_size, total_configs)
        results_block = []
        perfect_configs = []

        print(f"\n🔄 Valutazione config {start + 1}–{end}...\n")

        for config_idx, weights in enumerate(configs[start:end], start=start + 1):
            tunable = TunableMinimaxStrategy(weights=weights)
            win_count = 0

            for name, strat_cls in STRATEGIES.items():
                opponent = strat_cls() if callable(strat_cls) else strat_cls()
                win = evaluate_match(tunable, opponent)
                results.append((config_idx, name, win))
                results_block.append((config_idx, name, win))
                if win:
                    win_count += 1

                print(f"Config {config_idx:03} vs {name:<20}: {'✅ Win' if win else '❌ Loss'}")

            if win_count == len(STRATEGIES):
                perfect_configs.append({
                    "index": config_idx,
                    "wins": win_count,
                    "config": weights
                })

        # Salva risultati per blocco
        csv_path = f"performance_block_{block_number:03}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Config_Index", "Strategy", "Win"])
            writer.writerows(results_block)

        if perfect_configs:
            json_path = f"perfect_configs_{block_number:03}.json"
            with open(json_path, "w") as f:
                json.dump(perfect_configs, f, indent=4)
            print(f"🎯 Salvate {len(perfect_configs)} config perfette in {json_path}")

        perfect_configs_all.extend(perfect_configs)
        block_number += 1

    # Salva configurazioni perfette globali
    with open("perfect_configs_all.json", "w") as f:
        json.dump(perfect_configs_all, f, indent=4)

    print(f"\n🏁 Completato! Configurazioni perfette totali: {len(perfect_configs_all)}")
    print("📄 File salvati: performance_block_*.csv, perfect_configs_*.json, perfect_configs_all.json")

if __name__ == "__main__":
    main()
