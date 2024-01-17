from src.actor_critic_model import ActorCriticAgent
from src.trading_env import TradingEnv
import yfinance as yf

if __name__ == "__main__":
    lookback_window_size = 50

    agent = ActorCriticAgent(lookback_window_size=lookback_window_size)
    print(agent.shared_layers)
    print(agent.actor)
    print(agent.critic) 

    env = TradingEnv(stock_data=yf.Ticker("MSFT"), period="3y", initial_balance=1000, lookback_window_size=lookback_window_size, punish_coef=0.01)
    # agent.train(env, train_episodes=50, save_path="./experiments/")
    # agent.load("./experiments/2023-12-22-20-50-10_model")
    agent.load("./experiments/2024-01-17-22-32-59_model")
    agent.test(env, test_episodes=1)