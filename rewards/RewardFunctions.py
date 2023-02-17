from features.Features import State

class PnL:
    def calculate(self, current_state: State, next_state: State) -> float:
        current_value = current_state.portfolio.cash + current_state.portfolio.inventory * current_state.market.close
        next_value = next_state.portfolio.cash + next_state.portfolio.inventory * next_state.market.close
        return next_value - current_value

    def reset(self):
        pass