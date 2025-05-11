import copy
import random
import time
from cephalopod.core.board import Board, Die
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5

class TunableMinimaxStrategy:
    def __init__(self, weights=None, depth=1):
        self.weights = weights or {
            "capture_6": 6,
            "avoid_enemy_6": -12,
            "safe_capture": 0,
            "border_position": 0,
            "random_1": 0
        }
        self.depth = depth

    def choose_move(self, board, color):
        _, move = self.minimax(board, self.depth, color, True, color)
        return move

    def minimax(self, board, depth, player, maximizing_player, original_player, alpha=float("-inf"), beta=float("inf")):
        if depth == 0 or board.is_full():
            return self.evaluate_board(board, original_player), None

        possible_moves = self.generate_moves(board, player)
        if not possible_moves:
            return self.evaluate_board(board, original_player), None

        best_move = None

        if maximizing_player:
            best_score = -float("inf")
            for move in possible_moves:
                new_board = self.simulate_move(board, move, player)
                score, _ = self.minimax(new_board, depth - 1, self.get_opponent(player), False, original_player, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return best_score, best_move
        else:
            best_score = float("inf")
            for move in possible_moves:
                new_board = self.simulate_move(board, move, player)
                score, _ = self.minimax(new_board, depth - 1, self.get_opponent(player), True, original_player, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return best_score, best_move

    def generate_moves(self, board, color):
        moves = []
        for (r, c) in board.get_empty_cells():
            capturing = find_capturing_subsets(board, r, c)
            if capturing:
                subset, sum_pips = choose_capturing_subset(capturing)
                if sum_pips == 6 or sum_pips <= 4:
                    moves.append((r, c, sum_pips, subset))
                elif sum_pips == 5:
                    continue
            else:
                moves.append((r, c, 1, []))
        return moves

    def simulate_move(self, board, move, color):
        board_copy = copy.deepcopy(board)
        r, c, top, captured = move
        for rr, cc in captured:
            board_copy.grid[rr][cc] = None
        board_copy.place_die(r, c, Die(color, top))
        return board_copy

    def evaluate_board(self, board, my_color):
        opponent = self.get_opponent(my_color)
        score = 0
        for r in range(len(board.grid)):
            for c in range(len(board.grid[0])):
                die = board.grid[r][c]
                if die is not None:
                    if die.color == my_color:
                        score += 1
                        if die.top_face == 6:
                            score += self.weights["capture_6"]
                    elif die.color == opponent:
                        score -= 1
                        if die.top_face == 6:
                            score += self.weights["avoid_enemy_6"]
        return score

    def get_opponent(self, color):
        return "W" if color == "B" else "B"

def evaluate_match(strategy1, strategy2, n_matches=10):
    wins = 0
    total_margin = 0
    for i in range(n_matches):
        board = Board()
        current_player = "B" if i % 2 == 0 else "W"
        s1, s2 = (strategy1, strategy2) if current_player == "B" else (strategy2, strategy1)

        while not board.is_full():
            strat = s1 if current_player == "B" else s2
            move = strat.choose_move(board, current_player)
            if move is None:
                break
            r, c, top, captured = move
            for rr, cc in captured:
                board.grid[rr][cc] = None
            board.place_die(r, c, Die(current_player, top))
            current_player = "W" if current_player == "B" else "B"

        b_score = sum(1 for r in board.grid for d in r if d and d.color == "B")
        w_score = sum(1 for r in board.grid for d in r if d and d.color == "W")
        if (b_score > w_score and strategy1 == s1) or (w_score > b_score and strategy2 == s1):
            wins += 1
        total_margin += abs(b_score - w_score)

    return wins, total_margin / n_matches

def main():
    test_configs = [
        {
            "name": f"Config_{i}",
            "weights": {
                "capture_6": random.randint(0, 10),
                "avoid_enemy_6": random.randint(-20, 0),
                "safe_capture": random.randint(0, 5),
                "border_position": random.randint(0, 10),
                "random_1": random.randint(-3, 3)
            }
        }
        for i in range(1000)
    ]

    best = None
    most_wins = -1

    for config in test_configs:
        print(f"\n⚔️ Testing {config['name']}")
        strat = TunableMinimaxStrategy(weights=config["weights"], depth=2)
        opponent = SmartLookaheadStrategy5()
        start = time.time()
        wins, avg_margin = evaluate_match(strat, opponent, n_matches=10)
        end = time.time()
        print(f"🏆 {config['name']} won {wins}/10 matches | Avg margin: {avg_margin:.2f} | Duration: {round(end - start, 2)}s")
        if wins > most_wins:
            most_wins = wins
            best = config

    print("\n🔥 Best configuration:")
    print(best)

if __name__ == "__main__":
    main()