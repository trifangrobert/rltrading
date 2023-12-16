import pandas as pd
import numpy as np
from collections import deque
import yfinance as yf
import random

from rl.utils import TradingGraph

class TradingEnv:
    def __init__(self, stock_data: pd.DataFrame, period: str = "1y", initial_balance: int = 1000, lookback_window_size: int = 50, render_range: int = 100) -> None:
        print("Stock initialized.")
        self.stock_data = stock_data
        self.stock_history = self.stock_data.history(period=period)
        self.stock_history = self.stock_history.dropna().reset_index()

        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size

        self.total_steps = len(self.stock_history) - 1
        self.action_space = np.array([0, 1, 2])
        self.orders_history = deque(maxlen=self.lookback_window_size) 
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.state_size = (self.lookback_window_size, 10) # market_history and order_history length

        self.render_range = render_range

    def _get_curr_order(self) -> np.array:
        return np.array([self.balance, self.net_worth, self.bought_shares, self.sold_shares, self.held_shares])

    def _get_curr_market(self, current_step) -> np.array:
        return self.stock_history.loc[current_step, ["Open", "High", "Low", "Close", "Volume"]].to_numpy()

    def reset(self, env_step_size: int = 0) -> np.array:
        self.visualization = TradingGraph(render_range=self.render_range)
        self.trades = deque(maxlen=self.render_range) # this list will be used for arrows (buy or sell) on the graph

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance

        self.held_shares = 0
        self.sold_shares = 0
        self.bought_shares = 0

        if env_step_size > 0: # this is for training
            self.start_step = random.randint(self.lookback_window_size, self.total_steps - env_step_size)
            self.end_step = self.start_step + env_step_size
        else:
            self.start_step = self.lookback_window_size
            self.end_step = self.total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            order_now = self._get_curr_order()
            market_now = self._get_curr_market(current_step)
            self.orders_history.append(order_now)
            self.market_history.append(market_now)

        state = np.concatenate((self.orders_history, self.market_history), axis=1)
        return state

    def step(self, action: int) -> tuple:
        self.bought_shares = 0
        self.sold_shares = 0
        self.current_step += 1

        current_price = random.uniform(
            self.stock_history.loc[self.current_step, "Open"],
            self.stock_history.loc[self.current_step, "Close"]
        )

        date = self.stock_history.loc[self.current_step, "Date"]
        high = self.stock_history.loc[self.current_step, "High"]
        low = self.stock_history.loc[self.current_step, "Low"]

        if action == 0: # hold
            pass

        elif action == 1 and self.balance > 0:
            # buy as many shares as possible with current balance
            self.bought_shares = self.balance / current_price
            self.held_shares += self.bought_shares
            self.balance -= self.bought_shares * current_price
            self.trades.append({"date": date, "high": high, "low": low, "type": "buy"})

        elif action == 2 and self.held_shares > 0:
            # sell all shares
            self.sold_shares = self.held_shares
            self.balance += self.sold_shares * current_price
            self.held_shares -= self.sold_shares
            self.trades.append({"date": date, "high": high, "low": low, "type": "sell"})

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.held_shares * current_price

        self.orders_history.append(self._get_curr_order())

        reward = self.net_worth - self.prev_net_worth
        
        if self.net_worth <= self.initial_balance / 2: # esti praf, du-te acasa
            done = True
        else:
            done = False

        obs = self._next_observation()

        return obs, reward, done

    def _next_observation(self) -> np.array:
        self.market_history.append(self._get_curr_market(self.current_step))
        obs = np.concatenate((self.orders_history, self.market_history), axis=1)
        return obs

    def render(self, visualize: bool = False) -> None:
        if visualize:
            self.visualization.render(
                date = self.stock_history.loc[self.current_step, "Date"],
                open = self.stock_history.loc[self.current_step, "Open"],
                high = self.stock_history.loc[self.current_step, "High"],
                low = self.stock_history.loc[self.current_step, "Low"],
                close = self.stock_history.loc[self.current_step, "Close"],
                volume = self.stock_history.loc[self.current_step, "Volume"],
                net_worth = self.net_worth,
                trades = self.trades
            )
        # print(f"Step: {self.current_step - self.start_step + 1}, Net Worth: {self.net_worth}")


if __name__ == "__main__":
    env = TradingEnv(stock_data=yf.Ticker("AAPL"), period="1y", initial_balance=1000, lookback_window_size=50)
    print(env.stock_history.head())
