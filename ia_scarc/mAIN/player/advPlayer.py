



def playerStrategy(game, state):
    strategy = AdvancedMinimaxStrategy(time_limit=3.0)
    return strategy.choose_move(game, state, state.to_move)
