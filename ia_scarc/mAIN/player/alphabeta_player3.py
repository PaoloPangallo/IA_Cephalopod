from mAIN.strategies.alphabeta2 import AlphaBetaMinimaxStrategyPaolo
from mAIN.strategies.alphabeta3 import AlphaBetaMinimaxStrategyPaolino
from mAIN.strategies.alphabeta_minimax import AlphaBetaMinimaxStrategy


def playerStrategy(game, state):
    strategy = AlphaBetaMinimaxStrategyPaolino(time_limit=3.0)
    return strategy.choose_move(game, state, state.to_move)
