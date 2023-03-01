from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta, date

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import abc

from typing import Optional, Union


@dataclass
class Portfolio:
    inventory: float
    cash: float
    position: str
    n_trades: int
    entry_price: float


@dataclass
class Market:
    high: float
    low: float
    close: float
    volume: float

@dataclass
class State:
    market: Market
    portfolio: Portfolio
    now_is: datetime


class Feature(metaclass=abc.ABCMeta):
    def __init__(
        self,
        name: str,
        update_frequency: timedelta,
        lookback_periods: int,
        normalisation_on: bool,
        max_norm_len: int = 10000,
    ):
        self.name = name
        self.update_frequency = update_frequency
        self.lookback_periods = lookback_periods
        self.normalisation_on = normalisation_on
        self.max_norm_len = max_norm_len
        self.current_value = 0.0
        self.first_usage_time = datetime.min
        self.scalar = MinMaxScaler([-1, 1])
        if self.normalisation_on:
            self.history: deque = deque(maxlen=max_norm_len)

    @property
    def window_size(self) -> timedelta:
        return self.lookback_periods * self.update_frequency

    def normalise(self, value: float) -> float:
        if value is not np.nan:
            self.history.append(value)
        else:
            return np.nan
        return self.scalar.fit_transform(np.array(self.history).reshape(-1, 1))[-1]

    @abc.abstractmethod
    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        pass

    def update(self, state: State) -> None:
        if state.now_is >= self.first_usage_time:
            self._update(state)
            if self.normalisation_on: self.current_value = self.normalise(self.current_value)

    @abc.abstractmethod
    def _update(self, state: State) -> None:
        pass

    def _reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.first_usage_time = first_usage_time or datetime.min
        if self.normalisation_on:
            self.history.clear()
        self._update(state)


########################################################################################################################
#                                                  Price features                                                      #
########################################################################################################################
n_hours_trade_per_day = 7

class Returns(Feature):
    """
    The log returns of the price is calculated as the sign of the difference between the price at t - price at t-1
    """

    def __init__(
        self,
        name: str = "Returns",
        update_frequency: timedelta = timedelta(hours=1),
        lookback_periods: int = n_hours_trade_per_day+1,
        normalisation_on: bool = True,
    ):
        super().__init__(name, update_frequency, lookback_periods, normalisation_on)
        self.closes: deque = deque(maxlen=self.lookback_periods)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.closes = deque(maxlen=self.lookback_periods)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.closes.appendleft(state.market.close)
        if len(self.closes) < self.lookback_periods:
            self.current_value = np.nan
        elif len(self.closes) >= self.lookback_periods:
            self.current_value = np.log(self.closes[0]/self.closes[-1])



class Direction(Feature):
    """
    The direction is calculated as the sign of the difference between the price at t and the price at t-8
    """

    def __init__(
        self,
        name: str = "Direction",
        update_frequency: timedelta = timedelta(hours=1),
        lookback_periods: int = n_hours_trade_per_day+1,
        normalisation_on: bool = True,
    ):
        super().__init__(name, update_frequency, lookback_periods, normalisation_on)
        self.closes: deque = deque(maxlen=self.lookback_periods)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.closes = deque(maxlen=self.lookback_periods)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.closes.appendleft(state.market.close)
        if len(self.closes) < self.lookback_periods:
            self.current_value = np.nan
        elif len(self.closes) >= self.lookback_periods:
            self.current_value = np.sign(self.closes[0] - self.closes[-1])


class Volatility(Feature):
    """
    The volatility of the price series over a trailing window.
    The lookback period = 14*6, as we want to compute the indicator over the last 14 days (6hours session per day)
    """

    def __init__(
        self,
        name: str = "Volatility",
        update_frequency: timedelta = timedelta(hours=1),
        lookback_periods: int = 14*n_hours_trade_per_day,
        normalisation_on: bool = True,
    ):
        name += '_' + str(lookback_periods)
        super().__init__(name, update_frequency, lookback_periods, normalisation_on)
        self.closes: deque = deque(maxlen=self.lookback_periods + 1)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.closes = deque(maxlen=self.lookback_periods + 1)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.closes.append(state.market.close)
        if len(self.closes) < self.lookback_periods:
            self.current_value = np.nan
        elif len(self.closes) >= self.lookback_periods:
            pct_returns = np.log(np.array(self.closes)[-1:] / np.array(self.closes)[1:])
            self.current_value = np.std(pct_returns)


class RSI(Feature):
    """
    The relative strength index (RSI) is a momentum indicator used in technical analysis, measures the speed and
    magnitude of a security's recent price changes to evaluate overvalued or undervalued conditions
    in the price of that security
    """

    def __init__(
        self,
        name: str = "RSI",
        update_frequency: timedelta = timedelta(hours=1),
        lookback_periods: int = 14 * n_hours_trade_per_day,
        normalisation_on: bool = True,
    ):
        super().__init__(name, update_frequency, lookback_periods, normalisation_on)
        self.closes: deque = deque(maxlen=self.lookback_periods + 1)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.closes = deque(maxlen=self.lookback_periods + 1)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.closes.append(state.market.close)
        if len(self.closes) < self.lookback_periods:
            self.current_value = np.nan
        elif len(self.closes) >= self.lookback_periods:
            pct_returns = np.log(np.array(self.closes)[-1:] / np.array(self.closes)[1:])
            if np.any(np.where(pct_returns > 0)[0]) and np.any(np.where(pct_returns < 0)[0]):
                avg_up = np.mean(pct_returns[np.where(pct_returns > 0)])
                avg_down = np.abs(np.mean(pct_returns[np.where(pct_returns < 0)]))
                self.current_value = 100 * avg_up/(avg_up+avg_down)
            else:
                self.current_value = 0


class EWMA(Feature):
    """
    The exponential moving average (EMA) is a trend-following momentum indicator that measures the two exponential moving averages (EMAs) of a security’s price
    """

    def __init__(
        self,
        name: str = "EWMA",
        update_frequency: timedelta = timedelta(hours=1),
        lookback_periods: int = 7 * n_hours_trade_per_day,
        normalisation_on: bool = True,
    ):
        super().__init__(name, update_frequency, lookback_periods, normalisation_on)
        self.closes: deque = deque(maxlen=self.lookback_periods)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.closes = deque(maxlen=self.lookback_periods)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.closes.append(state.market.close)
        if len(self.closes) < self.lookback_periods:
            self.current_value = np.nan
        elif len(self.closes) >= self.lookback_periods:
            self.current_value = pd.Series(self.closes).ewm(span=self.lookback_periods, adjust=False, min_periods=self.lookback_periods).mean().values[-1]


class MACD(Feature):
    """
    The moving average convergence/divergence (MACD) is a trend-following momentum indicator that shows the relationship between two exponential moving averages (EMAs) of a security’s price
    """

    def __init__(
        self,
        name: str = "MACD",
        update_frequency: timedelta = timedelta(hours=1),
        lookback_periods: int = 26 * n_hours_trade_per_day,
        normalisation_on: bool = True,
    ):
        super().__init__(name, update_frequency, lookback_periods, normalisation_on)
        self.closes: deque = deque(maxlen=self.lookback_periods)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.closes = deque(maxlen=self.lookback_periods)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.closes.append(state.market.close)
        if len(self.closes) < self.lookback_periods:
            self.current_value = np.nan
        elif len(self.closes) >= self.lookback_periods:
            k = pd.Series(self.closes).ewm(span=12*n_hours_trade_per_day, adjust=False, min_periods=12*n_hours_trade_per_day).mean().values[-1]
            d = pd.Series(self.closes).ewm(span=26*n_hours_trade_per_day, adjust=False, min_periods=26*n_hours_trade_per_day).mean().values[-1]
            self.current_value = k - d


class ATR(Feature):
    """
    The average true range (ATR) measures market volatility by decomposing the entire range of an asset price for that period
    """

    def __init__(
        self,
        name: str = "ATR",
        update_frequency: timedelta = timedelta(hours=1),
        lookback_periods: int = 14 * n_hours_trade_per_day,
        normalisation_on: bool = True,
    ):
        super().__init__(name, update_frequency, lookback_periods, normalisation_on)
        self.closes: deque = deque(maxlen=self.lookback_periods)
        self.highs: deque = deque(maxlen=self.lookback_periods)
        self.lows: deque = deque(maxlen=self.lookback_periods)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.closes: deque = deque(maxlen=self.lookback_periods)
        self.highs: deque = deque(maxlen=self.lookback_periods)
        self.lows: deque = deque(maxlen=self.lookback_periods)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.closes.append(state.market.close)
        self.highs.append(state.market.high)
        self.lows.append(state.market.low)
        if len(self.closes) < self.lookback_periods:
            self.current_value = np.nan
        elif len(self.closes) >= self.lookback_periods:
            mami = np.array(self.highs) - np.array(self.lows)
            mac = np.abs(np.array(self.highs) - np.concatenate((np.full(1, np.nan), np.array(self.closes)[:-1])))
            mic = np.abs(np.array(self.lows) - np.concatenate((np.full(1, np.nan), np.array(self.closes)[:-1])))
            atr = np.maximum(mami, mac)[1:]
            atr = np.maximum(atr, mic[1:])
            self.current_value = np.mean(atr)


class STOCH(Feature):
    """
    The stochastic oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time.
    """

    def __init__(
        self,
        name: str = "STOCH",
        update_frequency: timedelta = timedelta(hours=1),
        lookback_periods: int = 14 * n_hours_trade_per_day,
        normalisation_on: bool = True,
    ):
        super().__init__(name, update_frequency, lookback_periods, normalisation_on)
        self.closes: deque = deque(maxlen=self.lookback_periods)
        self.highs: deque = deque(maxlen=self.lookback_periods)
        self.lows: deque = deque(maxlen=self.lookback_periods)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.closes: deque = deque(maxlen=self.lookback_periods)
        self.highs: deque = deque(maxlen=self.lookback_periods)
        self.lows: deque = deque(maxlen=self.lookback_periods)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.closes.append(state.market.close)
        self.highs.append(state.market.high)
        self.lows.append(state.market.low)
        if len(self.closes) < self.lookback_periods:
            self.current_value = np.nan
        elif len(self.closes) >= self.lookback_periods:
            maxi = np.max(np.array(self.highs))
            mini = np.min(np.array(self.lows))
            slowk = (np.array(self.closes) - mini)*100/(maxi-mini)
            slowd = np.mean(slowk[-3*n_hours_trade_per_day:])
            self.current_value = slowd - slowk[-1]


class WilliamsR(Feature):
    """
    The Williams Percent Range is a momentum indicator measuring overbought and oversold levels.
    """

    def __init__(
        self,
        name: str = "Williams %R",
        update_frequency: timedelta = timedelta(hours=1),
        lookback_periods: int = 14 * n_hours_trade_per_day,
        normalisation_on: bool = True,
    ):
        super().__init__(name, update_frequency, lookback_periods, normalisation_on)
        self.closes: deque = deque(maxlen=self.lookback_periods)
        self.highs: deque = deque(maxlen=self.lookback_periods)
        self.lows: deque = deque(maxlen=self.lookback_periods)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.closes: deque = deque(maxlen=self.lookback_periods)
        self.highs: deque = deque(maxlen=self.lookback_periods)
        self.lows: deque = deque(maxlen=self.lookback_periods)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.closes.append(state.market.close)
        self.highs.append(state.market.high)
        self.lows.append(state.market.low)
        if len(self.closes) < self.lookback_periods:
            self.current_value = np.nan
        elif len(self.closes) >= self.lookback_periods:
            maxi = np.max(np.array(self.highs))
            mini = np.min(np.array(self.lows))
            self.current_value = (maxi-self.closes[-1])/(maxi-mini)*(-100)


class OBV(Feature):
    """
    The On-balance volume (OBV) is a technical trading momentum indicator that uses volume flow to predict changes in stock price
    """

    def __init__(
        self,
        name: str = "OBV",
        update_frequency: timedelta = timedelta(hours=1),
        lookback_periods: int = 8,
        normalisation_on: bool = True,
    ):
        super().__init__(name, update_frequency, lookback_periods, normalisation_on)
        self.closes: deque = deque(maxlen=self.lookback_periods)
        self.volumes: deque = deque(maxlen=self.lookback_periods)
        self.obvs: deque = deque(maxlen=1)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.closes: deque = deque(maxlen=self.lookback_periods)
        self.volumes: deque = deque(maxlen=self.lookback_periods)
        self.obvs: deque = deque(maxlen=1)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.closes.appendleft(state.market.close)
        self.volumes.appendleft(state.market.volume)
        if len(self.obvs) == 0 and len(self.closes) < self.lookback_periods:
            self.current_value = np.nan
        elif len(self.obvs) == 0 and len(self.closes) == self.lookback_periods:
            obvs = 0 + np.nan_to_num(np.sign(self.closes[0]-self.closes[-1]))*self.volumes[0]
            self.obvs.append(obvs)
            self.current_value = np.nan
        elif len(self.obvs) == 1 and len(self.closes) == self.lookback_periods:
            self.obvs[0] += np.nan_to_num(np.sign(self.closes[0]-self.closes[-1]))*self.volumes[0]
            self.current_value = self.obvs[0]


class ChaikinFlow(Feature):
    """
    The Chaikin Money Flow measures the amount of Money Flow Volume over a specific period.
    """

    def __init__(
        self,
        name: str = "ChaikinMoneyFlow",
        update_frequency: timedelta = timedelta(hours=1),
        lookback_periods: int = 20 * n_hours_trade_per_day,
        normalisation_on: bool = True,
    ):
        super().__init__(name, update_frequency, lookback_periods, normalisation_on)
        self.highs: deque = deque(maxlen=self.lookback_periods)
        self.lows: deque = deque(maxlen=self.lookback_periods)
        self.closes: deque = deque(maxlen=self.lookback_periods)
        self.volumes: deque = deque(maxlen=self.lookback_periods)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.highs: deque = deque(maxlen=self.lookback_periods)
        self.lows: deque = deque(maxlen=self.lookback_periods)
        self.closes: deque = deque(maxlen=self.lookback_periods)
        self.volumes: deque = deque(maxlen=self.lookback_periods)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.highs.append(state.market.high)
        self.lows.append(state.market.low)
        self.closes.append(state.market.close)
        self.volumes.append(state.market.volume)
        if len(self.closes) < self.lookback_periods:
            self.current_value = np.nan
        elif len(self.closes) >= self.lookback_periods:
            mfv = np.sum(np.array(self.volumes) * (2*np.array(self.closes)-np.array(self.highs)-np.array(self.lows))/(np.array(self.highs)-np.array(self.lows)))
            self.current_value = mfv/np.sum(np.array(self.volumes))

########################################################################################################################
#                                                  Agent features                                                      #
########################################################################################################################


class Inventory(Feature):
    def __init__(
        self,
        name: str = "Inventory",
        update_frequency: timedelta = timedelta(hours=1),
        normalisation_on: bool = True,
    ):
        super().__init__(name, update_frequency, 0, normalisation_on)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.current_value = state.portfolio.inventory

