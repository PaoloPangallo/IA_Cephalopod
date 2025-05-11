# playerMinimax.py
from mAIN.strategies.fixed import AlphaBetaMinimaxStrategyPaoluz
from mAIN.strategies.minimax_zobreist import RobustDynamicMinimaxStrategy


# Importa solo la strategia che ti interessa

def playerStrategy(game, state):
    player = state.to_move
    # Istanzia la classe con il nome corretto
    strategy = AlphaBetaMinimaxStrategyPaoluz(time_limit=3.0)
    move = strategy.choose_move(game, state, player)
    return move
