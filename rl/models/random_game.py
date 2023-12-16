from rl.trading_env import TradingEnv
import numpy as np
from tqdm import tqdm
import yfinance as yf

def random_trading(env, train_episodes: int = 1, training_batch_size: int = 500): # training_batch_size is actually env_step_size in the TradingEnv
    print("Random trading")
    average_net_worth = 0
    for episode in tqdm(range(train_episodes), desc="Random trading"):
        state = env.reset(training_batch_size)
        while True:
            env.render(visualize=True)
            action = np.random.randint(3, size=1)[0]
            state, reward, done = env.step(action)
            if done:
                average_net_worth += env.net_worth
                print("The model lost all its money in episode {}.".format(episode))
                break
                
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("The model made it to the end of episode {} with {}.".format(episode, env.net_worth))
                break
    print("Average net worth: {}".format(average_net_worth / train_episodes))

        
if __name__ == "__main__":
    env = TradingEnv(stock_data=yf.Ticker("MSFT"), period="5y", initial_balance=1000, lookback_window_size=50)
    random_trading(env)
    