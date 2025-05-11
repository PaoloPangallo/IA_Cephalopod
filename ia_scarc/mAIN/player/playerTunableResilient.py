from mAIN.strategies.resilent_minimax import ResilientMinimaxStrategy
from mAIN.utils.optuna.tunable2 import TunableResilient2MinimaxStrategy
from mAIN.utils.optuna.tunable_resilient_minimax import TunableResilientMinimaxStrategy


def playerStrategy(game, state):
    player = state.to_move
    strategy = TunableResilient2MinimaxStrategy(depth=4)
    move = strategy.choose_move(game, state, player)
    return move
