from graphviz import Digraph

class Bracket8:
    """
    Disegna un bracket a 8 partecipanti (2^3).
    L'idea è di posizionare i match in 3 round:
      - Round 1 (4 match): M1, M2, M3, M4
      - Round 2 (2 match): M5, M6
      - Round 3 (1 match): M7 (finale)

    Per ogni match, creiamo un nodo 'Mx', e poi un nodo per il vincitore
    con eventuale punteggio. Così otteniamo un tabellone simmetrico.
    """

    def __init__(self, participants, results):
        """
        participants: lista di 8 nomi in ordine di seeding
          es: ["SmartLookahead", "Minimax", "Naive", "Heuristic",
               "Expectimax", "SmartPositional", "SmartBA", "SmartMini"]

        results: lista di 7 tuple (p1, p2, winner, score)
          - Round 1: M1, M2, M3, M4
          - Round 2: M5, M6
          - Round 3: M7 (finale)
          Esempio:
            [
              ("SmartLookahead","Minimax","SmartLookahead","B:14,W:11"),
              ("Naive","Heuristic","Heuristic","B:9,W:14"),
              ("Expectimax","SmartPositional","Expectimax","B:18,W:3"),
              ("SmartBA","SmartMini","SmartMini","B:10,W:12"),
              ("SmartLookahead","Heuristic","SmartLookahead","B:16,W:9"),
              ("Expectimax","SmartMini","SmartMini","B:14,W:16"),
              ("SmartLookahead","SmartMini","SmartLookahead","B:15,W:14")
            ]
        """
        if len(participants) != 8:
            raise ValueError("Occorrono esattamente 8 partecipanti!")
        if len(results) != 7:
            raise ValueError("Occorrono esattamente 7 match (4 quarti, 2 semifinali, 1 finale).")

        self.participants = participants
        self.results = results

    def render(self, output_file="bracket_8"):
        dot = Digraph("Bracket8", format="png")
        # Tipico bracket sportivo top-down
        dot.attr(rankdir="TB", splines="polyline")

        # 1) Creiamo i nodi per Round 1 (M1..M4), Round 2 (M5..M6), Round 3 (M7)
        #    e i partecipanti (P1..P8).
        #    Usiamo subgraph con rank="same" per disporre i match in colonne distinte.

        # Round 1 (colonna 1): 4 match
        with dot.subgraph(name="cluster_round1") as c:
            c.attr(rank="same")
            c.node("M1", label="Match 1", shape="circle")
            c.node("M2", label="Match 2", shape="circle")
            c.node("M3", label="Match 3", shape="circle")
            c.node("M4", label="Match 4", shape="circle")

        # Round 2 (colonna 2): 2 match
        with dot.subgraph(name="cluster_round2") as c:
            c.attr(rank="same")
            c.node("M5", label="Semifinale 1", shape="circle")
            c.node("M6", label="Semifinale 2", shape="circle")

        # Round 3 (colonna 3): 1 match (finale)
        with dot.subgraph(name="cluster_round3") as c:
            c.attr(rank="same")
            c.node("M7", label="Finale", shape="circle")

        # 2) Creiamo i nodi partecipanti (colonna 0)
        #    P1 vs P2 -> M1
        #    P3 vs P4 -> M2
        #    P5 vs P6 -> M3
        #    P7 vs P8 -> M4
        #    in un subgraph con rank="same"
        with dot.subgraph(name="cluster_participants") as c:
            c.attr(rank="same")
            for p in self.participants:
                c.node(p, p, shape="box")

        # 3) Colleghiamo partecipanti ai match Round 1
        #    (il seeding tipico è: P1 vs P8, P4 vs P5, ecc. - ma qui assumiamo P1 vs P2, P3 vs P4, ecc.)
        #    Cambia l'ordine se vuoi un bracket differente
        dot.edge(self.participants[0], "M1")
        dot.edge(self.participants[1], "M1")
        dot.edge(self.participants[2], "M2")
        dot.edge(self.participants[3], "M2")
        dot.edge(self.participants[4], "M3")
        dot.edge(self.participants[5], "M3")
        dot.edge(self.participants[6], "M4")
        dot.edge(self.participants[7], "M4")

        # 4) Colleghiamo Round 1 -> Round 2
        #    M5 = winner(M1) vs winner(M2)
        #    M6 = winner(M3) vs winner(M4)
        dot.edge("M1", "M5")
        dot.edge("M2", "M5")
        dot.edge("M3", "M6")
        dot.edge("M4", "M6")

        # 5) Colleghiamo Round 2 -> Round 3
        dot.edge("M5", "M7")
        dot.edge("M6", "M7")

        # 6) Ora disegniamo i “riquadri” dei vincitori con punteggio per ciascun match
        #    results = 7 tuple (p1, p2, winner, score)
        #    Ordine: M1, M2, M3, M4, M5, M6, M7
        for i, (p1, p2, w, score) in enumerate(self.results, start=1):
            match_id = f"M{i}"
            # Creiamo un nodo in cui scriviamo "Winner + punteggio"
            winner_label = f"{w}\n({score})"
            winner_node = f"Winner_{i}"
            dot.node(winner_node, winner_label, shape="box", style="filled", fillcolor="lightgrey")
            # Colleghiamo Mx -> winner_node
            dot.edge(match_id, winner_node)

        # 7) Render su file
        dot.render(output_file, view=True)
        print(f"Bracket salvato in {output_file}.png")
