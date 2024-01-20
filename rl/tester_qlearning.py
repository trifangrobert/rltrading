import cv2 as cv
import os
import yfinance as yf


from rl.src.qlearning_agent import QLearningAgent, QLearningParams, test_q_learning_agent, q_table_factory
from rl.src.trading_env import TradingEnv


env = TradingEnv(stock_data=yf.Ticker("MSFT"), period="6mo", initial_balance=1000, lookback_window_size=50)
params = QLearningParams()
agent = QLearningAgent(env, params)
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "qlearning")
agent.load(base_dir, "model_lbk_5_1000")
try:
    test_q_learning_agent(env, agent, num_episodes=100, visualize=False)
except KeyboardInterrupt:
    print("Training stopped manually.")
finally:
    print("Press any key to exit.")
    cv.waitKey(0)
    cv.destroyAllWindows()
