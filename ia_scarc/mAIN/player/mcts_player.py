from mAIN.strategies.adv_minimax import AlphaBetaMinimaxStrategy2
from mAIN.strategies.alphabeta_minimax import AlphaBetaMinimaxStrategy


def playerStrategy(game, state):
    strategy = AlphaBetaMinimaxStrategy(time_limit=3.0)
    return strategy.choose_move(game, state, state.to_move)
