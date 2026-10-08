"""CPU-only neural sanity checks for the AlphaZero-style prototype."""
import unittest
import numpy as np

try:
    import torch
except ImportError:  # lightweight core smoke tests don't require PyTorch
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch not installed")
class NeuralSmokeTests(unittest.TestCase):
    def test_policy_value_shapes_and_legal_mask(self):
        from cephalopod.alphazero.neural_network import NeuralNetwork
        torch.manual_seed(7)
        net = NeuralNetwork()
        state = torch.zeros((3, 5, 5))
        logits, value = net(state.unsqueeze(0))
        self.assertEqual(tuple(logits.shape), (1, 25))
        self.assertEqual(tuple(value.shape), (1, 1))
        probs, predicted_value = net.predict(state, [(0, 1), (4, 4)])
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertEqual(int(np.count_nonzero(probs)), 2)
        self.assertEqual(float(probs[0]), 0.0)
        self.assertTrue(-1 <= predicted_value <= 1)
        empty, _ = net.predict(state, [])
        self.assertTrue(np.all(empty == 0))

    def test_train_accepts_soft_mcts_targets(self):
        from cephalopod.alphazero.neural_network import NeuralNetwork
        from cephalopod.alphazero.train_cephalopod_zero import train
        torch.manual_seed(11)
        net = NeuralNetwork()
        policy = np.zeros(25, dtype=np.float32)
        policy[0], policy[1] = 0.25, 0.75
        examples = [
            (np.zeros((3, 5, 5), dtype=np.float32), policy, 1.0),
            (np.ones((3, 5, 5), dtype=np.float32), policy, -1.0),
        ]
        before = net.fc_policy.weight.detach().clone()
        train(net, examples, epochs=1)
        self.assertFalse(torch.equal(before, net.fc_policy.weight))
        with self.assertRaises(ValueError):
            train(net, [], epochs=1)

    def test_winner_has_draw_state(self):
        from cephalopod.alphazero.cephalopod_zero_dynamic import CephalopodZero
        from cephalopod.alphazero.neural_network import NeuralNetwork
        from cephalopod.core.board import Board, Die
        agent = CephalopodZero(NeuralNetwork(), mcts_simulations=1)
        board = Board()
        board.place_die(0, 0, Die("B", 1))
        board.place_die(0, 1, Die("W", 1))
        self.assertEqual(agent.evaluate_winner(board), 0)


if __name__ == "__main__":
    unittest.main()
