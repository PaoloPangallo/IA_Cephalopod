# utils/cephalop_file_manager.py
import os
import pandas as pd

from core.config import DATA_FOLDER

DEFAULT_CSV_FILE = "data/default_cephalopod.csv"  # puoi importarlo da config se vuoi


class CephalopFileManager:
    """
    Gestisce il caricamento e il salvataggio di file CSV.
    """

    def __init__(self, default_file=DEFAULT_CSV_FILE):
        self.current_file = default_file
        self.df = None

    def load_csv(self, file_path):
        if not os.path.exists(file_path):
            return None
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Errore caricamento CSV {file_path}: {e}")
            return None

    def set_file(self, file_path):
        self.current_file = file_path
        self.df = self.load_csv(file_path)

    def append_or_create_csv(self, df_to_add, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
            old_df = self.load_csv(file_path)
            if old_df is not None:
                combined_df = pd.concat([old_df, df_to_add], ignore_index=True)
                combined_df.to_csv(file_path, index=False)
                return combined_df
        df_to_add.to_csv(file_path, index=False)
        return df_to_add

    def get_next_game_id(self):
        if self.df is not None and not self.df.empty:
            return self.df["game_id"].max() + 1
        return 1

    def list_csv_files(self, folder="data"):
        if os.path.isdir(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
            print(f"[DEBUG] CSV trovati nella cartella '{folder}':", files)
            return files
        print(f"[DEBUG] Cartella '{folder}' non trovata.")
        return []


def refresh_csv_list(self):
    csv_files = self.file_manager.list_csv_files(DATA_FOLDER)
    self.csv_file_combo['values'] = csv_files
    if csv_files:
        self.csv_file_var.set(csv_files[0])
    else:
        self.csv_file_var.set("")


# Alla fine di create_widgets():
