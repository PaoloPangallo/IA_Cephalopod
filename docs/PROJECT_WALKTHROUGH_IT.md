# IA_Cephalopod — Guida per raccontare il progetto

> Motivazioni, architettura e preparazione al colloquio. La descrizione è ricavata dal codice presente nel repository e distingue gli esperimenti dalle funzionalità verificate.
>
> [System design con diagrammi](SYSTEM_DESIGN.md) · [README](../README.md)

## 1. Partiamo dal problema, non dalle librerie

In un gioco strategico a turni, ogni mossa modifica un ambiente condiviso. Per scegliere bene non basta trovare una mossa immediatamente conveniente: bisogna considerare cosa potrebbe fare l'avversario e valutare le conseguenze delle decisioni.

Questo rende il gioco un buon contesto per una domanda più ampia:

**Come affrontano lo stesso problema decisionale tecniche di intelligenza artificiale basate su regole, ricerca e apprendimento?**

IA_Cephalopod esplora questa domanda utilizzando il gioco Cephalopod come ambiente di sperimentazione. Il valore del lavoro non dipende dall'aver realizzato il giocatore «più forte» in assoluto, ma dall'aver implementato più modalità di rappresentare e selezionare le azioni.

## 2. Spiegazione del gioco in un minuto

Il sistema principale usa una scacchiera 5×5 in cui due giocatori, identificati come B e W, posizionano a turno dadi nelle celle libere.

Quando un dado viene posizionato accanto ad altri dadi, determinate combinazioni di **almeno due vicini ortogonali** possono essere catturate, purché la somma dei valori sia al massimo 6. I dadi catturati vengono rimossi e il nuovo dado assume il valore della somma. In assenza di cattura, il dado ha valore 1.

Nel motore di simulazione principale la partita termina quando la scacchiera è piena oppure un giocatore non produce più mosse. Vince chi possiede il maggior numero di dadi presenti; un pareggio viene rappresentato come `DRAW`.

Le strategie condividono principalmente una convenzione: ricevono lo stato della scacchiera e il colore del giocatore, poi restituiscono la mossa scelta.

```text
choose_move(board, color)
    -> (riga, colonna, valore_dado, dadi_catturati)
```

Questa interfaccia è semplice ed è il punto di partenza per confrontare comportamenti differenti. Tuttavia gli esperimenti neurali e la seconda implementazione `ia_scarc/` non sono tutti collegati allo stesso motore attraverso un adapter rigorosamente uniforme.

## 3. L'idea architetturale

Il progetto separa, almeno nella sua struttura principale, il **motore del gioco** dalle **strategie**.

Il motore rappresenta la scacchiera, applica le mosse, gestisce turni e catture e registra l'esito delle partite. La strategia decide *quale azione proporre* a partire dallo stato corrente. Gli script di simulazione e torneo permettono di far giocare gli agenti tra loro e produrre risultati da analizzare.

Questa separazione ha una motivazione precisa: **mantenere le regole relativamente stabili mentre cambiano gli algoritmi decisionali**.

L'organizzazione del repository contiene più percorsi sperimentali, non un'unica piattaforma perfettamente unificata. Le simulazioni del blocco `cephalopod/` e quelle di `ia_scarc/` vanno documentate come implementazioni correlate ma distinte.

## 4. Le famiglie di algoritmi e perché studiarle

### A. Baseline casuali ed euristiche

La strategia più semplice seleziona una cella disponibile; una strategia euristica può preferire mosse che consentono una buona cattura immediata.

**Perché inserirle?** Danno un riferimento interpretabile e poco costoso rispetto al quale studiare scelte più sofisticate.

**Limite:** non modellano necessariamente la risposta futura dell'avversario.

### B. Minimax e potatura Alpha-Beta

Minimax simula mosse proprie e contromosse avversarie, valutando gli stati con una funzione euristica quando raggiunge la profondità massima. Le varianti con potatura Alpha-Beta evitano l'esplorazione di rami che non possono migliorare la decisione.

**Perché inserirli?** Permettono di mostrare il passaggio da una scelta locale a un ragionamento avversario esplicito.

**Trade-off:** aumentare la profondità migliora potenzialmente l'orizzonte decisionale ma può far crescere rapidamente il costo computazionale. Conta anche la qualità della funzione di valutazione.

Nel repository sono presenti varianti con pesi euristici regolabili e script di tuning tramite Optuna.

### C. Reinforcement Learning a valori tabulari

L'agente RL prova azioni, conserva valori associati agli **stati risultanti** e aggiorna le stime usando ricompense finali e reward shaping. Un meccanismo di esplorazione ε-greedy alterna mosse esplorative e scelte guidate dalle stime.

**Perché inserirlo?** Qui il comportamento non dipende soltanto da pesi progettati manualmente: una parte delle preferenze emerge dalle partite precedenti.

**Precisione tecnica:** l'implementazione non è una DQN; usa una tabella di valori per stati successivi e una procedura di aggiornamento personalizzata. Presentarla genericamente come «Q-learning deep» sarebbe scorretto.

### D. Behavior Cloning

Questa linea genera esempi stato–azione utilizzando una strategia esperta e addestra una rete PyTorch in modo supervisionato.

**Perché inserirlo?** Per studiare l'apprendimento per imitazione, in cui il segnale di addestramento arriva dalle decisioni di un esperto invece che dall'esplorazione autonoma.

Nel codice attuale il modello predice **coordinate e valore del dado come variabili continue**, con arrotondamenti e fallback in inferenza. Non è un classificatore mascherato di tutte le mosse legali. Questo aspetto va raccontato come scelta/prototipo sperimentale, non come garanzia di legalità.

### E. AlphaZero-style: rete neurale e MCTS

La linea più avanzata unisce una piccola rete convoluzionale, che produce probabilità sulle posizioni e una stima del valore dello stato, a un Monte Carlo Tree Search guidato da queste stime.

La procedura sperimentale di self-play produce esempi per addestrare la rete: la distribuzione di visite MCTS costituisce il target della policy, mentre l'esito della partita fornisce il target del valore.

**Perché inserirlo?** Combina due idee: imparare dai dati e continuare a ragionare sulle alternative attraverso la ricerca.

**Limite importante:** questa è una realizzazione *ispirata ad AlphaZero*, non una riproduzione completa o una dimostrazione che superi Minimax. I checkpoint precedenti alle correzioni recenti non sono stati riaddestrati con il nuovo codice.

## 5. Una struttura da ricordare

```text
              STATO DEL GIOCO
                     |
           scelta dell'approccio
                     |
        +------------+------------+
        |            |            |
     EURISTICHE    RICERCA    APPRENDIMENTO
        |            |            |
     greedy       Minimax      RL tabulare
                  Alpha-Beta   Behavior Cloning
                               Neural MCTS
                     |
                MOSSA / AZIONE
                     |
               MOTORE DI GIOCO
                     |
             ESITO DELLA PARTITA
```

I rami rappresentano **approcci alternativi**. Non indicano che tutti gli agenti girino in sequenza durante la stessa decisione.

## 6. Come raccontarlo in 30 secondi

> Ho utilizzato il gioco strategico Cephalopod come ambiente per sperimentare diversi paradigmi di AI. Ho implementato strategie euristiche, ricerca avversaria con Minimax e Alpha-Beta e percorsi di apprendimento con reinforcement learning, imitation learning e un prototipo AlphaZero-style che combina rete neurale e MCTS. La motivazione era studiare come cambia la selezione delle azioni passando da regole esplicite a ricerca e apprendimento. Il progetto include anche simulazioni e strumenti di tuning per esplorare le differenze tra gli approcci.

## 7. Come raccontarlo in circa 90 secondi

> IA_Cephalopod nasce dall'interesse per un problema tipico dell'intelligenza artificiale: come prendere decisioni in un ambiente competitivo in cui ogni scelta influenza le mosse successive dell'avversario.
>
> Ho usato un gioco a turni con regole deterministiche come ambiente sperimentale. Il motore mantiene lo stato della scacchiera e applica le regole, mentre diverse strategie propongono le mosse da eseguire.
>
> La prima famiglia comprende strategie casuali ed euristiche, utili come baseline. Ho poi esplorato Minimax e potatura Alpha-Beta, dove l'agente analizza possibili contromosse e valuta gli stati futuri. Alcune varianti includono pesi configurabili e tuning.
>
> Sul lato apprendimento ho sviluppato un percorso di reinforcement learning basato su valori tabulari degli stati e reward shaping, e un percorso di Behavior Cloning che tenta di imitare le decisioni di una strategia esperta.
>
> Infine ho esplorato un agente AlphaZero-style: una rete convoluzionale policy/value guida un MCTS, e il self-play produce esempi usati per addestrare la rete. È un prototipo di ricerca e non lo presento come una replica completa di AlphaZero.
>
> L'elemento più interessante è il confronto concettuale tra fonti diverse della decisione: euristiche scritte a mano, ricerca dell'albero, esperienza accumulata, imitazione dell'esperto e stime neurali combinate con la ricerca. Il passo successivo sarebbe consolidare un benchmark con regole, seed, budget e metriche uniformi per confrontare quantitativamente questi metodi.

## 8. Domande che potrebbero farti

| Domanda | Risposta difendibile |
| --- | --- |
| **Perché usare un gioco invece di un dataset statico?** | Perché permette di osservare decisioni sequenziali, interazione con l'avversario ed effetti delle azioni nel tempo. |
| **Qual è la differenza tra greedy e Minimax?** | La prima ottimizza prevalentemente un vantaggio locale; Minimax considera la risposta avversaria simulando stati futuri. |
| **Cosa cambia con Alpha-Beta?** | La soluzione minimax rimane equivalente se l'algoritmo è corretto, ma alcuni rami possono essere scartati evitando lavoro non necessario. |
| **Perché serve una funzione euristica?** | La ricerca fino agli stati terminali può essere costosa: ai cutoff serve una stima dell'utilità della posizione. |
| **Il tuo RL è Q-learning o DQN?** | È più preciso descriverlo come value-based RL tabulare su stati successivi, con ε-greedy e reward shaping, non come DQN. |
| **Behavior Cloning e RL sono la stessa cosa?** | No. Il BC imita esempi di decisioni esperte; il RL aggiorna le proprie preferenze sulla base delle ricompense ottenute giocando. |
| **Che cosa predice la rete AlphaZero-style?** | Una distribuzione sulle 25 posizioni della scacchiera e un valore scalare della posizione dal punto di vista del giocatore. |
| **A cosa serve MCTS se hai già una rete?** | La rete fornisce una stima iniziale, mentre la ricerca esplora in modo selettivo possibili sviluppi del gioco. |
| **Hai dimostrato che AlphaZero è più forte?** | No: ci sono prototipi, esperimenti e checkpoint, ma non un benchmark recente, controllato e comparabile tra tutte le famiglie. |
| **Qual è il miglioramento architetturale prioritario?** | Definire un contratto unico e validato per stato e mossa, seguito da un harness di valutazione comune agli agenti. |

## 9. Risultati, limiti e prossimo esperimento

Nel repository sono presenti partite simulate, CSV storici, configurazioni ottimizzate e modelli salvati. Non basta però leggere questi artefatti per attribuire una vittoria scientificamente significativa a un determinato algoritmo.

Per una comparazione robusta servirebbe fissare le stesse regole, alternare i colori, controllare seed e posizioni iniziali, misurare vittorie/pareggi/sconfitte e tempi decisionali, oltre a ripetere abbastanza partite per poter riportare l'incertezza delle stime.

In particolare, alcuni vecchi script gestiscono il pareggio in modo non uniforme e la linea AlphaZero-style è cambiata dopo la creazione dei checkpoint già presenti. Non dichiarerei quindi miglioramenti numerici non rivalidati.

I test automatici esistenti coprono casi del motore, MCTS e del training neurale su CPU, **non** certificano tutte le strategie o gli esperimenti storici.

## 10. La frase che sintetizza il progetto

> **Ho usato un ambiente decisionale condiviso per esplorare come un agente possa scegliere una mossa attraverso euristiche, ricerca dell'avversario oppure apprendimento dai dati e dall'esperienza.**
