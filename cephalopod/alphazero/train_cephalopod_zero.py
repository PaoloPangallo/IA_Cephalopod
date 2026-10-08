import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from cephalopod.alphazero.neural_network import NeuralNetwork
from cephalopod.alphazero.cephalopod_zero_dynamic import CephalopodZero

# Parametri
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_GAMES = 30
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "cephalopod_zero.pth")


def train(model, data, epochs=EPOCHS):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn_value = nn.MSELoss()

    for epoch in range(epochs):
        np.random.shuffle(data)
        total_loss = 0

        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            states, target_policies, target_values = zip(*batch)

            x = torch.tensor(np.array(states), dtype=torch.float32)  # (B, 3, 5, 5)
            if x.ndim != 4 or x.shape[1:] != (3, 5, 5):
                raise ValueError(f"Unexpected board tensor shape: {tuple(x.shape)}")

            y_policy = torch.tensor(np.array(target_policies), dtype=torch.float32)
            y_value = torch.tensor(np.array(target_values), dtype=torch.float32).unsqueeze(1)

            policy_logits, value = model(x)
            policy_loss = -(y_policy * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = loss_fn_value(value, y_value)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(data) // BATCH_SIZE)
        print(f"📈 Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    print("🧠 Inizializzo il modello neurale...")
    model = NeuralNetwork()

    print("🎮 Creo agente CephalopodZero con MCTS...")
    agent = CephalopodZero(model=model, mcts_simulations=25)

    print(f"♻️ Avvio self-play con {NUM_GAMES} partite...")
    training_data = []
    for i in range(NUM_GAMES):
        print(f"  ▶️ Partita {i + 1}/{NUM_GAMES} in corso...")
        training_data.extend(agent.play_game())
    print(f"✅ Dati raccolti: {len(training_data)} esempi")

    print("🏋️ Inizio training della rete neurale...")
    train(model, training_data, epochs=EPOCHS)

    print(f"💾 Salvo modello in: {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("🎉 Training completato con successo!")