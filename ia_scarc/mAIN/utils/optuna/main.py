from analisi import (
    carica_dati,
    mostra_statistiche,
    plot_score_vs,
    salva_top_trials
)

def main():
    df = carica_dati()
    mostra_statistiche(df)

    # Plotta i parametri più significativi
    plot_score_vs(df, 'bonus_six_weight')
    plot_score_vs(df, 'risk_factor')
    plot_score_vs(df, 'safe_placement_weight')
    plot_score_vs(df, 'dominance_weight')

    # Salva i migliori trial
    salva_top_trials(df, top_n=5)

if __name__ == "__main__":
    main()
