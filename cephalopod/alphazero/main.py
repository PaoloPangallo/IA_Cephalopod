# 📁 File: cephalopod/alphazero/main.py
import os

import torch
from cephalopod.alphazero.neural_network import NeuralNetwork
from cephalopod.alphazero.cephalopod_zero_dynamic import CephalopodZero
from cephalopod.alphazero.ui import CephalopodUI


def main():
    print("📦 Carico il modello CephalopodZero...")
    model = NeuralNetwork()
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "cephalopod_zero.pth"), map_location="cpu")
)
    model.eval()

    print("🧠 Inizializzo agente con MCTS...")
    agent = CephalopodZero(model=model, mcts_simulations=25)

    print("🖥️ Avvio UI auto-play...")
    CephalopodUI(agent)


if __name__ == "__main__":
    main()