from mAIN.strategies.smart_lookahead5 import SmartLookaheadStrategy5

def playerStrategy(game, state):
    player = state.to_move
    strategy = SmartLookaheadStrategy5()
    move = strategy.choose_move(game, state, player)
    return move

