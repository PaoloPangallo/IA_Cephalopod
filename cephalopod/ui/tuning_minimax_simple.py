import copy
import json
import logging
import random
from pathlib import Path

import numpy as np

from game_modes.cephalopod_game_dynamic import CephalopodGameDynamic

# ---- Logger setup ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Import delle componenti ----
from cephalopod.core.board import Die
from cephalopod.core.mechanics import get_opponent, find_capturing_subsets, choose_capturing_subset


# ----- Param bounds -----
PARAM_BOUNDS = {
    'piece': (0.5, 2.0),
    'bonus_six': (1.0, 5.0),
    'opponent_piece': (0.5, 2.0),
    'opponent_six': (1.0, 4.0)
}


# ----- Strategy & Evaluation -----
class MinimaxStrategy:
    def __init__(self, depth=3, weights=None):
        self.depth = depth
        self.weights = weights if weights else {
            'piece': 1,
            'bonus_six': 3,
            'opponent_piece': 1,
            'opponent_six': 2
        }

    def choose_move(self, board, color):
        _, best_move = minimax(board, self.depth, color, True, color, self.weights)
        return best_move


def minimax(board, depth, player, maximizing, origin, weights, alpha=-float("inf"), beta=float("inf")):
    if depth == 0 or board.is_full():
        return evaluate_board(board, origin, weights), None

    moves = get_all_legal_moves(board, player)
    best_move = None

    if maximizing:
        max_score = -float("inf")
        for move in moves:
            next_board = simulate_move(board, move, player)
            score, _ = minimax(next_board, depth - 1, get_opponent(player), False, origin, weights, alpha, beta)
            if score > max_score:
                max_score = score
                best_move = move
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return max_score, best_move
    else:
        min_score = float("inf")
        for move in moves:
            next_board = simulate_move(board, move, player)
            score, _ = minimax(next_board, depth - 1, get_opponent(player), True, origin, weights, alpha, beta)
            if score < min_score:
                min_score = score
                best_move = move
            beta = min(beta, score)
            if beta <= alpha:
                break
        return min_score, best_move


def evaluate_board(board, player, weights):
    opponent = get_opponent(player)
    score = 0
    for r in range(board.size):
        for c in range(board.size):
            die = board.grid[r][c]
            if die:
                if die.color == player:
                    score += weights['piece']
                    if die.top_face == 6:
                        score += weights['bonus_six']
                elif die.color == opponent:
                    score -= weights['opponent_piece']
                    if die.top_face == 6:
                        score -= weights['opponent_six']
    return score


def get_all_legal_moves(board, player):
    moves = []
    for r, c in board.get_empty_cells():
        options = find_capturing_subsets(board, r, c)
        if options:
            subset, sum_pips = choose_capturing_subset(options)
            moves.append((r, c, sum_pips, subset))
        else:
            moves.append((r, c, 1, []))
    return moves


def simulate_move(board, move, player):
    new_board = copy.deepcopy(board)
    r, c, top_face, captured = move
    for rr, cc in captured:
        new_board.grid[rr][cc] = None
    new_board.place_die(r, c, Die(player, top_face))
    return new_board


def simulate_cephalopod_game(weights, depth=3):
    baseline = {'piece': 1, 'bonus_six': 3, 'opponent_piece': 1, 'opponent_six': 2}
    strategy_B = MinimaxStrategy(depth, weights)
    strategy_W = MinimaxStrategy(depth, baseline)
    game = CephalopodGameDynamic(strategy_B, strategy_W)
    winner = game.simulate_game2()
    return 1 if winner == "B" else -1


# ----- Genetic Optimizer -----
def random_weights():
    return {k: random.uniform(*v) for k, v in PARAM_BOUNDS.items()}


def fitness(weights, n_games=5, depth=3):
    results = [simulate_cephalopod_game(weights, depth) for _ in range(n_games)]
    return np.mean(results)


def crossover(p1, p2):
    return {k: p1[k] if random.random() < 0.5 else p2[k] for k in p1}


def mutate(weights, rate=0.1):
    mutated = weights.copy()
    for k, (low, high) in PARAM_BOUNDS.items():
        if random.random() < rate:
            delta = (high - low) * 0.1
            mutated[k] += random.uniform(-delta, delta)
            mutated[k] = max(low, min(high, mutated[k]))
    return mutated


def genetic_algorithm(gens=30, pop_size=20, depth=3, n_games=5, out_file="best_weights.json"):
    history = []
    population = [random_weights() for _ in range(pop_size)]

    for gen in range(gens):
        logger.info(f"GEN {gen}")
        scored = [(w, fitness(w, n_games, depth)) for w in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        best = scored[0]
        logger.info(f"MIGLIOR FITNESS: {best[1]:.3f} | PESI: {best[0]}")
        history.append({'generation': gen, 'fitness': best[1], 'weights': best[0]})

        # Salva su file ogni generazione
        Path("logs").mkdir(exist_ok=True)
        with open(f"logs/{out_file}", "w") as f:
            json.dump(history, f, indent=2)

        # Riproduzione
        survivors = [w for w, _ in scored[:pop_size // 2]]
        new_pop = survivors.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = mutate(crossover(p1, p2))
            new_pop.append(child)
        population = new_pop

    return history[-1]


# ----- MAIN -----
if __name__ == "__main__":
    logger.info("Inizio tuning avanzato con algoritmo genetico!")
    final_best = genetic_algorithm()
    logger.info("Tuning completato.")
    logger.info(f"MIGLIORI PESI FINALI: {final_best['weights']} con fitness: {final_best['fitness']}")
