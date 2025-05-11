from mAIN.utils.optuna.trial44_strategy import Trial44BestStrategyTimed



def playerStrategy(game, state):
    strategy = Trial44BestStrategyTimed()
    return strategy.choose_move(game, state, state.to_move)

