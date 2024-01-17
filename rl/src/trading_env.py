import pandas as pd
import numpy as np
from collections import deque
import yfinance as yf
import random

from rl.utils import TradingGraph

class TradingEnv:
    def __init__(self, stock_data: pd.DataFrame, period: str = "1y", initial_balance: int = 1000, lookback_window_size: int = 50, render_range: int = 100, punish_coef: float = 0.1) -> None:
        print("Stock initialized.")
        self.stock_data = stock_data
        self.stock_history = self.stock_data.history(period=period)
        self.stock_history = self.stock_history.dropna().reset_index()

        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size

        self.total_steps = len(self.stock_history) - 1
        self.action_space = np.array([0, 1, 2]) # hold buy sell
        self.orders_history = deque(maxlen=self.lookback_window_size) 
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.episode_orders = 0

        self.render_range = render_range

        self.punish_coef = punish_coef
        self.normalize_value = 10000


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

        self.episode_orders = 0
        self.prev_episode_orders = 0

        self.rewards = deque(maxlen=self.render_range)
        self.punish_value = 0
        self.env_step_size = env_step_size

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

        # current_price = (self.stock_history.loc[self.current_step, "Open"] + self.stock_history.loc[self.current_step, "Close"]) / 2

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
            self.trades.append({"date": date, "high": high, "low": low, "type": "buy", 'total': self.bought_shares, 'current_price': current_price})
            self.episode_orders += 1

        elif action == 2 and self.held_shares > 0:
            # sell all shares
            self.sold_shares = self.held_shares
            self.balance += self.sold_shares * current_price
            self.held_shares -= self.sold_shares
            self.trades.append({"date": date, "high": high, "low": low, "type": "sell", 'total': self.sold_shares, 'current_price': current_price})
            self.episode_orders += 1

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.held_shares * current_price

        # self.orders_history.append(self._get_curr_order())

        # reward = self.net_worth - self.prev_net_worth
        reward = self.get_reward() / self.normalize_value
        
        if self.net_worth <= self.initial_balance / 2: # esti praf, du-te acasa
            done = True
        else:
            done = False

        obs = self._next_observation() 

        return obs, reward, done
    
    def get_reward(self):
        self.punish_value += self.net_worth * self.punish_coef
        if len(self.trades) >= 2 and self.episode_orders > 1 and self.episode_orders > self.prev_episode_orders:
            self.prev_episode_orders = self.episode_orders
            if self.trades[-1]['type'] == "buy" and self.trades[-2]['type'] == "sell":
                reward = self.trades[-2]['total']*self.trades[-2]['current_price'] - self.trades[-1]['total']*self.trades[-1]['current_price']
                reward -= self.punish_value
                self.punish_value = 0
                self.trades[-1]["Reward"] = reward
                return reward
            elif self.trades[-1]['type'] == "sell" and self.trades[-2]['type'] == "buy":
                reward = self.trades[-1]['total']*self.trades[-1]['current_price'] - self.trades[-2]['total']*self.trades[-2]['current_price']
                reward -= self.punish_value
                self.punish_value = 0
                self.trades[-1]["Reward"] = reward
                return reward
            else:
                return 0 - self.punish_value
        else:
            return 0 - self.punish_value

    def _next_observation(self) -> np.array:
        self.orders_history.append(self._get_curr_order())
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
