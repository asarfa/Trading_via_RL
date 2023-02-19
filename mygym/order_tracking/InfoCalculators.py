import numpy as np
import pandas as pd
from features.Features import State


class InfoCalculator:
    def __init__(
            self,
            verbose: bool = True
    ):
        self.verbose = verbose

    def reset_episode(self):
        self.dates = []
        self.pnls, self.pnl = [], 0
        self.market_pnls, self.market_pnl = [], 0
        self.positions = []
        self.aums, self.market_aums= [], []
        self.n_trades = 0
        self.sharpe = 0

    def calculate(self, internal_state: State, reward: float, reward_mkt: float, init_cash: float) -> pd.DataFrame:
        self.dates.append(internal_state.now_is)
        self.n_trades = internal_state.portfolio.n_trades
        self.pnl += reward
        self.pnls.append(self.pnl)
        self.market_pnl += reward_mkt
        self.market_pnls.append(self.market_pnl)
        self.positions.append(internal_state.portfolio.position)
        self.aums.append(self.calculate_aum(internal_state))
        self.market_aums.append(init_cash+self.market_pnl)
        self.sharpe = self.calculate_sharpe(np.array(self.aums))
        return self._compute_info(internal_state)

    def _compute_info(self, internal_state: State) -> pd.DataFrame:
        pnl_mp_pe = (self.pnls[-1] - self.pnls[-2]) if len(self.pnls) > 1 else self.pnls[0]
        if internal_state.portfolio.position == 'neutral' and pnl_mp_pe==-2:
            print('neutral + double fees weird')
        info_dict = dict(
            posititon=self.positions[-1],
            pnl_per_episode=pnl_mp_pe,
            aum_agent=self.aums[-1],
            aum_market=self.market_aums[-1],
            sharp=self.sharpe,
            n_trades=self.n_trades
        )
        info = pd.DataFrame([info_dict], index=[self.dates[-1]])
        return info

    def calculate_aum(self, internal_state: State) -> float:
        return internal_state.portfolio.cash + internal_state.market.close * internal_state.portfolio.inventory

    @staticmethod
    def calculate_sharpe(aum_array):
        if aum_array[-1]<0: aum_array=aum_array[:-1]
        if len(aum_array) > 100 and np.min(aum_array) > 0:
            log_returns = np.diff(np.log(aum_array[~np.isnan(aum_array)]))
            simple_returns = np.exp(log_returns) - 1
            # ddof = 1 to get divisor n-1 in std
            # add 1e-10 to avoid e.g. 0/0 = inf
            sharpe = np.mean(simple_returns) / (np.std(simple_returns, ddof=1) + 1e-10)
            return sharpe*(252**0.5)
        else:
            return np.nan