import optuna
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

def carica_dati(study_name="cephalopod_tuning_v3", db_path="sqlite:///cephalopod_v3.db"):
    print("📥 Caricamento dati tuning da Optuna...")
    study = optuna.load_study(study_name=study_name, storage=db_path)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    data = []
    for t in completed_trials:
        row = t.params.copy()
        row['composite_score'] = t.value
        row['trial_number'] = t.number
        row['win_rate_minimax'] = t.user_attrs['win_rates'][0]
        row['win_rate_smart5'] = t.user_attrs['win_rates'][1]
        row['win_rate_self'] = t.user_attrs['win_rates'][2]
        row['avg_margin'] = sum(t.user_attrs['margins']) / len(t.user_attrs['margins'])
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv("risultati_cephalopod.csv", index=False)
    print("✅ Dati caricati e salvati in risultati_cephalopod.csv")
    return df

def mostra_statistiche(df):
    print("\n📊 Statistiche principali:")
    print(df.describe())

def plot_score_vs(df, column, filename=None):
    if column not in df.columns:
        print(f"❌ Colonna '{column}' non trovata.")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(df[column], df['composite_score'], alpha=0.7)
    plt.title(f"Composite Score vs {column}")
    plt.xlabel(column)
    plt.ylabel("composite_score")
    plt.grid(True)
    plt.tight_layout()
    if not filename:
        filename = f"score_vs_{column}.png"
    plt.savefig(filename)
    print(f"📈 Salvato: {filename}")

    try:
        os.startfile(filename)
    except Exception as e:
        print(f"⚠️ Impossibile aprire automaticamente il grafico: {e}")

def salva_top_trials(df, top_n=5, output_path="top_trials.json"):
    df_sorted = df.sort_values(by='composite_score', ascending=False).head(top_n)
    top_trials = df_sorted.to_dict(orient="records")

    with open(output_path, "w") as f:
        json.dump(top_trials, f, indent=4)

    print(f"\n🏆 Salvati i Top {top_n} trial in {output_path}")

if __name__ == "__main__":
    df = carica_dati()
    mostra_statistiche(df)
    plot_score_vs(df, 'bonus_six_weight')
    plot_score_vs(df, 'risk_factor')
    plot_score_vs(df, 'safe_placement_weight')
    plot_score_vs(df, 'dominance_weight')
    salva_top_trials(df)