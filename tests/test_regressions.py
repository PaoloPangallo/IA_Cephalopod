"""Regression tests for rules, match summaries, and the AlphaZero MCTS adapter.

Run with: python -m unittest discover -s tests -v
Only NumPy is needed; PyTorch-dependent checks are optional.
"""
import unittest
import numpy as np

from cephalopod.core.board import Board, Die
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
from cephalopod.game_modes.cephalopod_game_dynamic import CephalopodGameDynamic
from cephalopod.strategies.heuristic import HeuristicStrategy
from cephalopod.alphazero.mcts import MCTS, MCTSNode


class FirstEmpty:
    def choose_move(self, board, color):
        empty = board.get_empty_cells()
        return (*empty[0], 1, []) if empty else None


class StopEarly:
    def choose_move(self, board, color):
        return None


class AlwaysOccupied:
    def choose_move(self, board, color):
        return (0, 0, 1, [])


class DummyNet:
    def predict(self, tensor, legal_moves):
        probs = np.zeros(25, dtype=float)
        for r, c in legal_moves:
            probs[5 * r + c] = 1 / len(legal_moves)
        return probs, 0.0


class DummyAgent:
    def __init__(self):
        self.model = DummyNet()

    def encode_board(self, board, player):
        return None


class CoreRegressionTests(unittest.TestCase):
    def test_clone_keeps_nondefault_size_and_deep_copies(self):
        b = Board(3)
        b.place_die(0, 1, Die("B", 4))
        clone = b.clone()
        self.assertEqual(clone.size, 3)
        self.assertEqual(len(clone.grid), 3)
        self.assertEqual(clone.grid[0][1].top_face, 4)
        clone.grid[0][1].top_face = 2
        self.assertEqual(b.grid[0][1].top_face, 4)

    def test_capture_subset_does_not_mutate_options(self):
        options = [([(0, 0), (1, 1)], 3), ([(0, 0), (1, 1), (2, 2)], 5)]
        snapshot = [(list(cells), total) for cells, total in options]
        selected, total = choose_capturing_subset(options)
        self.assertEqual((selected, total), ([(0, 0), (1, 1), (2, 2)], 5))
        self.assertEqual(options, snapshot)

    def test_dynamic_finished_game_has_winner_metadata(self):
        game = CephalopodGameDynamic(FirstEmpty(), FirstEmpty())
        log = game.simulate_game()
        self.assertTrue(game.board.is_full())
        self.assertEqual(len([x for x in log if x["player"] in ("B", "W")]), 25)
        self.assertEqual(log[-1]["player"], "WINNER")
        self.assertEqual(log[-1]["winner"], "B")
        self.assertEqual(log[-1]["captured"], "B")

    def test_early_stalemate_is_not_a_fabricated_win(self):
        game = CephalopodGameDynamic(StopEarly(), StopEarly())
        game.board.place_die(0, 0, Die("B", 1))
        game.board.place_die(1, 1, Die("W", 1))
        self.assertEqual(game.simulate_game()[-1]["winner"], "DRAW")
        again = CephalopodGameDynamic(StopEarly(), StopEarly())
        again.board.place_die(0, 0, Die("B", 1))
        again.board.place_die(1, 1, Die("W", 1))
        self.assertEqual(again.simulate_game2(), "DRAW")

    def test_move_onto_occupied_cell_fails(self):
        game = CephalopodGameDynamic(AlwaysOccupied(), AlwaysOccupied())
        self.assertTrue(game.simulate_move())
        with self.assertRaisesRegex(ValueError, "Illegal move"):
            game.simulate_move()

    def test_heuristic_import_and_move(self):
        b = Board()
        move = HeuristicStrategy().choose_move(b, "B")
        self.assertIn(move[:2], b.get_empty_cells())

    def test_mcts_capture_face_uses_sum(self):
        b = Board()
        b.place_die(0, 1, Die("B", 2))
        b.place_die(1, 0, Die("W", 3))
        node = MCTSNode(b, "B")
        node.expand(DummyAgent(), "B")
        self.assertEqual(node.children[0].game.grid[0][0].top_face, 5)

    def test_mcts_only_selects_legal_cells(self):
        b = Board()
        b.place_die(0, 0, Die("B", 1))
        policy = MCTS(DummyAgent(), num_simulations=5).run(b, "W")
        self.assertEqual(len(policy), 25)
        self.assertAlmostEqual(float(policy.sum()), 1)
        self.assertEqual(float(policy[0]), 0)

    def test_mcts_terminal_returns_empty_distribution(self):
        b = Board()
        for r in range(5):
            for c in range(5):
                b.place_die(r, c, Die("B", 1))
        probs = MCTS(DummyAgent(), num_simulations=2).run(b, "W")
        self.assertTrue(np.all(probs == 0))

    def test_mcts_rejects_zero_simulations(self):
        with self.assertRaises(ValueError):
            MCTS(DummyAgent(), num_simulations=0).run(Board(), "B")


if __name__ == "__main__":
    unittest.main()
