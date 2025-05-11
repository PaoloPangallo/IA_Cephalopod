# playerMinimax.py
from mAIN.strategies.minimax_strategies import DynamicMinimaxStrategy


def playerStrategy(game, state):
    player = state.to_move
    strategy = DynamicMinimaxStrategy(time_limit=3.0)
    move = strategy.choose_move(game, state, player)
    return move


