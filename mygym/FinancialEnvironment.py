from mygym.order_tracking.InfoCalculators import InfoCalculator
from features.Features import *
from orders.models import MarketOrder, OrderDict
from database import HistoricalDatabase
from rewards.RewardFunctions import PnL
from copy import deepcopy


class FinancialEnvironment:
    def __init__(
            self,
            database: HistoricalDatabase = None,
            features: list = None,
            ticker: str = None,
            step_size: timedelta = None,
            initial_portfolio: Portfolio = None,
            start_of_trading: datetime = None,
            end_of_trading: datetime = None,
            per_step_reward_function: PnL = None,
            normalisation_on: bool = None,
            n_lags_feature: int = None,
            manage_risk: bool = None,
            verbose: bool = False
    ):

        self.database = database
        self.ticker = ticker
        self.step_size = step_size
        self.start_of_trading = start_of_trading
        self.end_of_trading = end_of_trading
        self.per_step_reward_function = per_step_reward_function
        self.n_lags_feature = n_lags_feature if n_lags_feature == 0 else n_lags_feature - 1
        self.manage_risk = manage_risk
        self.info_calculator = InfoCalculator(verbose=verbose)
        self._check_params()
        self.verbose = verbose
        self.features = features or self.get_default_features(step_size, normalisation_on)
        self.max_feature_window_size = max([feature.window_size for feature in self.features])
        self.state: State = None
        self.aum_threshold: int = 0
        self.volume_per_trade: int = 100
        self.init_cash: float = 1000.
        self.initial_portfolio = initial_portfolio or self._get_init_ptf()
        self.trade_cost: float = 1. #1$ fee per trade
        if self.manage_risk: self.sl = 0.05
        if self.manage_risk: self.tp = 0.10

    def reset(self) -> np.ndarray:
        now_is = self.start_of_trading  # (self.max_feature_window_size + self.step_size * self.n_lags_feature)
        self.state = State(market=self._get_market_data(now_is), portfolio=self._get_init_ptf(), now_is=now_is)
        self.info_calculator.reset_episode()
        self._reset_features(now_is)
        if self.n_lags_feature > 0: self.lags_feature = np.zeros((self.n_lags_feature + 1, len(self.features)))
        for step in range(int(self.max_feature_window_size / self.step_size) + self.n_lags_feature - 1):
            self._forward(None)
            if self.n_lags_feature > 0:
                self._set_lags_features(step)
        return self.get_features() if self.n_lags_feature == 0 else self.lags_feature

    def step(self, action: int):
        if self.manage_risk: self._manage_risks()
        action -= 1 # if action = 0 then action = -1 (short), if action = 1 then action = 0 (neutral) else long
        done = False
        current_state = deepcopy(self.state)
        self._forward(action)
        next_state = self.state
        reward = self.per_step_reward_function.calculate(current_state, next_state)
        features = self.get_features()
        if self.end_of_trading <= next_state.now_is or (self.aum_value <= self.aum_threshold):
            done = True
        reward_market = (next_state.market.close - current_state.market.close) * self.volume_per_trade
        info = self.info_calculator.calculate(self.state, reward, reward_market, self.init_cash)
        return features, reward, done, info

    def _forward(self, action: int):
        order = self._convert_action_to_order(action) if action is not None else None
        self.update_internal_state(order)
        self._update_features()

    def _get_features(self) -> np.ndarray:
        return np.array([feature.current_value for feature in self.features]).squeeze()

    def get_features(self) -> np.ndarray:
        if self.n_lags_feature == 0:
            return self._get_features()
        else:
            self.lags_feature[:-1] = self.lags_feature[1:]
            self.lags_feature[-1] = self._get_features()
            return self.lags_feature

    def update_internal_state(self, order):
        self._update_portfolio(order)
        self.state.now_is += self.step_size
        if self.state.now_is not in self.database.calendar:
            self.state.now_is = self.database.get_next_timestep(self.state.now_is, self.ticker)
        self.state.market = self._get_market_data(self.state.now_is)

    def _convert_inventory_to_position(self):
        if self.state.portfolio.inventory == 0:
            return "neutral"
        elif self.state.portfolio.inventory < 0:
            return "short"
        elif self.state.portfolio.inventory > 0:
            return "long"

    def _convert_action_to_order(self, action: int) -> MarketOrder:
        previous_position = self._convert_inventory_to_position()
        if previous_position == "neutral":
            if action == -1:
                # going from neutral to short, hence selling for one factor of volume
                order_dict = self._get_default_order_dict(direction="sell", factor_volume=1)
            elif action == 0:
                # going from neutral to neutral, hence no order
                return None
            else:
                # going from neutral to long, hence buying for one factor of volume
                order_dict = self._get_default_order_dict(direction="buy", factor_volume=1)
        elif previous_position == "long":
            if action == -1:
                # going from long to short, hence selling for two factor of volume
                order_dict = self._get_default_order_dict(direction="sell", factor_volume=2)
            elif action == 0:
                # going from long to neutral, hence selling for one factor of volume
                order_dict = self._get_default_order_dict(direction="sell", factor_volume=1)
            else:
                # going from long to long, hence no order
                return None
        elif previous_position == "short":
            if action == -1:
                # going from short to short, hence no order
                return None
            elif action == 0:
                # going from short to neutral, hence buying for one factor of volume
                order_dict = self._get_default_order_dict(direction="buy", factor_volume=1)
            else:
                # going from short to long, hence buying for two factor of volume
                order_dict = self._get_default_order_dict(direction="buy", factor_volume=2)
        order = MarketOrder(**order_dict.__dict__)
        return order

    def _reset_features(self, episode_start: datetime):
        for feature in self.features:
            first_usage_time = episode_start - feature.window_size
            feature.reset(self.state, first_usage_time)

    def _update_features(self):
        for feature in self.features:
            feature.update(self.state)

    def _update_portfolio(self, order):
        if order is None:
            self.state.portfolio.n_trades += 0
        else:
            self.state.portfolio.entry_price = order.price
            last_trades = 2 if order.volume > self.volume_per_trade else 1
            fees = last_trades*self.trade_cost
            if order.direction == "sell":
                self.state.portfolio.inventory -= order.volume
                self.state.portfolio.cash += order.volume * self.state.portfolio.entry_price - fees
            elif order.direction == "buy":
                self.state.portfolio.inventory += order.volume
                self.state.portfolio.cash -= order.volume * self.state.portfolio.entry_price + fees
            self.state.portfolio.n_trades += last_trades
            self.state.portfolio.position = self._convert_inventory_to_position()

    def _manage_risks(self):
        # stop loss order
        """
        The SL order should protect, in general, from unfavorable market movements that are larger than typical
        price movements (noise).
        """
        if self.sl is not None and self.state.portfolio.inventory != 0:  # Checks whether a SL is defined and whether the position is not neutral
            self.__compute_stoploss(self.state.market.close)
        # take profit order
        """
        The TP order should protect from decent profits that might not be secured and positions might remain open for 
        too long until they give up previous profits.
        """
        if self.tp is not None and self.state.portfolio.inventory != 0:  # Checks whether a TP is defined and whether the position is not neutral
            self.__compute_takeprofit(self.state.market.close)

    def __compute_stoploss(self, price: float):
        """
        A SL order closes out a position that has reached a certain loss level.
        If the entry price for an unleveraged position is 100 and the SL level is set to 5%, then a long posi‐
        tion is closed out at 95 while a short position is closed out at 105
        """
        rc = (price - self.state.portfolio.entry_price) / self.state.portfolio.entry_price #Calculates the performance based on the entry price for the last trade
        if self.state.portfolio.position == "long" and rc < -self.sl: #Checks whether an SL event is given for a long position
            #print(f'*** STOP LOSS (LONG  | {rc:.4f}) ***')
            order = self._convert_action_to_order(0) #Closes the long position, at the current price
            self._update_portfolio(order)  # Sets the position to neutral
        elif self.state.portfolio.position == "short" and rc > self.sl: #Checks whether an SL event is given for a short position
            #print(f'*** STOP LOSS (SHORT | -{rc:.4f}) ***')
            order = self._convert_action_to_order(0) #Closes the long position, at the current price
            self._update_portfolio(order)  # Sets the position to neutral

    def __compute_takeprofit(self, price: float):
        """
        A TP order closes out a position that has reached a certain profit level.
        If an unleveraged long position is opened at a price of 100 and the
        TP order is set to a level of 5%. If the price reaches 105, the position is closed
        """
        rc = (price - self.state.portfolio.entry_price) / self.state.portfolio.entry_price
        if self.state.portfolio.position == "long" and rc > self.tp:
            #print(f'*** TAKE PROFIT (LONG  | {rc:.4f}) ***')
            order = self._convert_action_to_order(0)
            self._update_portfolio(order)
        elif self.state.portfolio.position == "short" and rc < -self.tp:
            #print(f'*** TAKE PROFIT (SHORT | {-rc:.4f}) ***')
            order = self._convert_action_to_order(0)
            self._update_portfolio(order)

    def _check_params(self):
        assert self.start_of_trading <= self.end_of_trading, "Start of trading Nonsense"

    def _set_lags_features(self, step):
        thrs = 2
        if step == int(self.max_feature_window_size / self.step_size) - thrs:
            self.lags_feature[0] = self._get_features()
        elif step > int(self.max_feature_window_size / self.step_size) - thrs:
            self.lags_feature[step - int(self.max_feature_window_size / self.step_size) + thrs] = self._get_features()

    def _get_market_data(self, datepoint: timedelta):
        data = self.database.get_last_snapshot(datepoint, self.ticker)
        return Market(**{k.lower(): v for k, v in data.to_dict().items()})

    @property
    def aum_value(self):
        return self.state.portfolio.cash + self.state.portfolio.inventory * self.state.market.close

    def _get_init_ptf(self):
        return Portfolio(inventory=0, cash=self.init_cash, position="neutral", n_trades=0, entry_price=0)

    def _get_default_order_dict(self, direction: str, factor_volume: int) -> OrderDict:
        return OrderDict(
            timestamp=self.state.now_is,
            price=self.state.market.close,
            volume=factor_volume * self.volume_per_trade,
            direction=direction,
            ticker=self.ticker,
        )

    @staticmethod
    def get_default_features(step_size: timedelta, normalisation_on: bool):
        return [Returns(update_frequency=step_size,
                        normalisation_on=normalisation_on),
                Direction(update_frequency=step_size,
                          normalisation_on=normalisation_on),
                Volatility(update_frequency=step_size,
                           normalisation_on=normalisation_on),
                RSI(update_frequency=step_size,
                    normalisation_on=normalisation_on),
                EWMA(lookback_periods=7 * 7,
                     update_frequency=step_size,
                     normalisation_on=normalisation_on),
                EWMA(lookback_periods=14 * 7,
                     update_frequency=step_size,
                     normalisation_on=normalisation_on),
                EWMA(lookback_periods=21 * 7,
                     update_frequency=step_size,
                     normalisation_on=normalisation_on),
                MACD(update_frequency=step_size,
                     normalisation_on=normalisation_on),
                ATR(update_frequency=step_size,
                    normalisation_on=normalisation_on),
                STOCH(update_frequency=step_size,
                      normalisation_on=normalisation_on),
                WilliamsR(update_frequency=step_size,
                          normalisation_on=normalisation_on),
                OBV(update_frequency=step_size,
                    normalisation_on=normalisation_on),
                ChaikinFlow(update_frequency=step_size,
                            normalisation_on=normalisation_on),
                Inventory(update_frequency=step_size,
                          normalisation_on=normalisation_on)]
