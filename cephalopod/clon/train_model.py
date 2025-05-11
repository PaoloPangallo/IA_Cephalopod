import torch
from torch.utils.data import DataLoader
from clon.behavior import BehaviorCloningModel
from expert_dataset import ExpertDataset
import random
from cephalopod.core.board import Die, Board
from cephalopod.core.mechanics import find_capturing_subsets, choose_capturing_subset
import pickle
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt


def is_move_legal(board, move, color):
    r, c, top_face, captured = move
    if not board.in_bounds(r, c):
        return False
    if board.grid[r][c] is not None:
        return False
    if captured:
        sum_pips = sum(board.grid[rr][cc].top_face for (rr, cc) in captured)
        if sum_pips != top_face:
            return False
    # ⚠️ Se la posizione ha subset catturabili con somma == top_face, la cattura è obbligatoria
    capturing_options = find_capturing_subsets(board, r, c)
    for subset, total in capturing_options:
        if total == top_face:
            # Se il giocatore poteva catturare e non ha catturato => illegale
            if set(captured) != set(subset):
                return False
    return True


def generate_expert_dataset(num_games=1000, save_path="expert_dataset.pkl"):
    from cephalopod.strategies.smart_lookahead5 import SmartLookaheadStrategy5
    expert = SmartLookaheadStrategy5()
    data = []

    for i in range(num_games):
        board = Board()
        color = "B"

        for _ in range(4):
            empty = board.get_empty_cells()
            if not empty:
                break
            r, c = random.choice(empty)
            state = str([[str(cell) if cell else "" for cell in row] for row in board.grid])
            data.append((state, (r, c, 1, [])))
            board.place_die(r, c, Die(color, 1))
            color = "W" if color == "B" else "B"

        while not board.is_full():
            move = expert.choose_move(board, color)
            if not is_move_legal(board, move, color):
                continue
            state = str([[str(cell) if cell else "" for cell in row] for row in board.grid])
            data.append((state, move))
            r, c, top_face, captured = move
            for (rr, cc) in captured:
                board.grid[rr][cc] = None
            board.place_die(r, c, Die(color, top_face))
            color = "W" if color == "B" else "B"

        print(f"[INFO] Gioco {i + 1}/{num_games} completato.")

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"[✓] Dataset salvato in '{save_path}'")


def train_behavior_model(epochs=10, model_save_path="policy_bc.pt", optimizer_path="optimizer_state.pt",
                         log_path="training_log.csv", patience=5, verbose=True, resume=True):
    print("[INFO] Caricamento del dataset...")
    dataset = ExpertDataset("expert_dataset.pkl")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    total_batches = len(dataloader)
    print(f"[INFO] Dataset caricato con {len(dataset)} esempi.")
    print(f"[INFO] Ogni epoca avrà circa {total_batches} batch.")

    print("[INFO] Inizializzazione del modello...")
    model = BehaviorCloningModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_epoch = 0
    best_loss = float('inf')
    epochs_without_improvement = 0

    if resume and os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location="cpu"))
        print(f"[✓] Modello pre-addestrato caricato da '{model_save_path}'")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
            print(f"[✓] Stato dell'optimizer caricato da '{optimizer_path}'")

    criterion = torch.nn.MSELoss()

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "batch", "batch_loss", "total_loss", "timestamp"])

    print("[INFO] Inizio training...")
    all_losses = []
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        for batch_idx, (states, actions) in enumerate(dataloader):
            try:
                preds = model(states)

                actions = actions.clone()
                preds = preds.clamp(min=0, max=5)
                loss = criterion(preds.float(), actions.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss

                if verbose and batch_idx % 10 == 0:
                    print(f"[Ep {epoch + 1} | Batch {batch_idx}/{total_batches}] Loss: {batch_loss:.4f}")

                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, batch_idx, batch_loss, total_loss, datetime.now().isoformat()])

            except Exception as e:
                print(f"[ERRORE] Ep {epoch + 1} | Batch {batch_idx} — {e}")

        all_losses.append(total_loss)
        print(f"[✓] Epoca {epoch + 1} completata. Loss totale: {total_loss:.2f}")

        if total_loss < best_loss:
            best_loss = total_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            print(f"[✓] Nuovo best model salvato in '{model_save_path}'")
        else:
            epochs_without_improvement += 1
            print(f"[INFO] Nessun miglioramento. ({epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                print("[STOP] Early stopping attivato.")
                break

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(all_losses) + 1), all_losses, marker='o')
    plt.title("Loss per Epoca")
    plt.xlabel("Epoca")
    plt.ylabel("Loss Totale")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_plot.png")
    print("[✓] Grafico salvato in 'training_loss_plot.png'")


if __name__ == "__main__":
    train_behavior_model()
