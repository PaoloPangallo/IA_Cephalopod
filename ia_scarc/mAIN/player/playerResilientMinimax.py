from mAIN.strategies.resilent_minimax import ResilientMinimaxStrategy


def playerStrategy(game, state):
    player = state.to_move
    strategy = ResilientMinimaxStrategy(depth=3)
    move = strategy.choose_move(game, state, player)
    return move
