# playerMinimax.py
from mAIN.strategies.alphabeta_minimax import AlphaBetaMinimaxStrategy


def playerStrategy(game, state):
    player = state.to_move
    strategy = AlphaBetaMinimaxStrategy(time_limit=3.0)
    move = strategy.choose_move(game, state, player)
    return move


