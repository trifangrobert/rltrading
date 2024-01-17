from src.actor_critic_model import ActorCriticAgent
from src.trading_env import TradingEnv
import yfinance as yf
import random

if __name__ == "__main__":
    lookback_window_size = 50

    agent = ActorCriticAgent(lookback_window_size=lookback_window_size)
    print(agent.shared_layers)
    print(agent.actor)
    print(agent.critic) 

    stock_tickers = ["MSFT", "AAPL", "AMZN", "GOOG", "META", "TSLA", "NVDA", "BB", "AMD", "INTC", "PLTR", "ABNB", "UBER", "DASH", "ZM"]
    # shuffle the tickers
    random.shuffle(stock_tickers)
    train_tickers = stock_tickers[:10]

    print(f"Training on: {train_tickers}")

    for ticker in train_tickers:
        print(f"Training on {ticker}")
        env = TradingEnv(stock_data=yf.Ticker(ticker), period="5y", initial_balance=1000, lookback_window_size=lookback_window_size, punish_coef=0.01)
        agent.train(env, train_episodes=50, save_path="./experiments/")

