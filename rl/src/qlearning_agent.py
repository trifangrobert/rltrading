from collections import defaultdict
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time
from tqdm import tqdm
from typing import Literal
import yfinance as yf

from rl.src.trading_env import TradingEnv


class QLearningParams:
    def __init__(self) -> None:
        self.start_epsilon: float = 1
        self.end_epsilon: float = 0.01
        self.epsilon_decay: float = 0.995
        self.use_epsilon_decay: bool = True

        self.gamma: float = 0.5

        self.start_alpha = 0.3
        self.end_alpha = 0.01
        self.alpha_decay = 0.995

        self.lookback_window_size: int = 6
        self.stock_norm_type: Literal["log"] = "log"

    def __str__(self) -> str:
        return "\n".join([f"{k}: {v}" for k, v in self.__dict__.items()])


def q_table_factory() -> np.ndarray:
    return np.zeros(3)


class QLearningAgent:
    def __init__(self, env: TradingEnv, params: QLearningParams) -> None:
        self.env = env
        self.params = params
        self.n_actions = len(self.env.action_space)

        self.q_table: dict[np.ndarray, np.ndarray] = defaultdict(q_table_factory)  # action value

    # get best action for given q state; tie breaker is random
    def __get_best_action(self, q_state: tuple) -> int:
        max_action_value = np.max(self.q_table[q_state])

        return np.random.choice(np.flatnonzero(self.q_table[q_state] == max_action_value))

    def __get_model_name(self, n_train_episodes: int) -> str:
        return "model_lbk_" + str(self.params.lookback_window_size) + "_" + str(n_train_episodes)

    # translates env state of shape (lookback_window_size, 10)
    # to q state of shape (params.lookback_window_size)
    # q states take the following form:
    #  - first element is the stock price, normalized
    #  - the rest are the differences between stock prices in consecutive days, normalized to +-1,
    #    with the first element being the most recent
    def env_state_to_q_state(self, env_state: np.ndarray) -> tuple:
        q_state = list(np.zeros(self.params.lookback_window_size))  # use list so we can have both int and float

        if self.params.stock_norm_type == "log":
            # stock price normed = floor(log2(stock price))
            open_price = env_state[-1][5]
            close_price = env_state[-1][8]
            stock_price = (open_price + close_price) / 2
            stock_price_normed = int(np.floor(np.log2(stock_price)))
        else:
            raise ValueError(f"Unknown stock norm type: {self.params.stock_norm_type}")
        q_state[0] = stock_price_normed

        for i in range(1, self.params.lookback_window_size):
            # stock price (i - 1) days ago - stock price (i) days ago
            stock_price_today = (env_state[-i][5] + env_state[-i][8]) / 2
            stock_price_yesterday = (env_state[-i - 1][5] + env_state[-i - 1][8]) / 2
            q_state[i] = (
                0
                if stock_price_today == stock_price_yesterday
                else int(
                    np.sign(stock_price_today - stock_price_yesterday)
                    * np.log2(np.abs(stock_price_today - stock_price_yesterday))
                )
            )

        return tuple(q_state)

    # get epsilon based on params
    def get_epsilon(self, n_episode: int) -> float:
        if self.params.use_epsilon_decay:
            epsilon = max(self.params.start_epsilon * (self.params.epsilon_decay**n_episode), self.params.end_epsilon)
        else:
            epsilon = self.params.start_epsilon
        return epsilon

    # get alpha based on params
    def get_alpha(self, n_episode: int) -> float:
        if self.params.use_epsilon_decay:
            alpha = max(self.params.start_alpha * (self.params.alpha_decay**n_episode), self.params.end_alpha)
        else:
            alpha = self.params.start_alpha
        return alpha

    # select action based on epsilon greedy
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        q_state = self.env_state_to_q_state(state)
        random_action = np.random.choice(self.env.action_space)
        best_action = self.__get_best_action(q_state)
        if np.random.rand() > epsilon:
            action = best_action
        else:
            action = random_action

        return action

    def update_experience(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, n_episode: int
    ) -> None:
        q_state = self.env_state_to_q_state(state)
        q_next_state = self.env_state_to_q_state(next_state)
        best_actionIndex_fromNextState = self.__get_best_action(q_next_state)
        best_actionValue_fromNextState = self.q_table[q_next_state][best_actionIndex_fromNextState]

        alpha = self.get_alpha(n_episode)
        target = reward + self.params.gamma * best_actionValue_fromNextState
        self.q_table[q_state][action] = self.q_table[q_state][action] + (
            alpha * (target - self.q_table[q_state][action])
        )

    # save to experiments folder
    def save(self, n_train_episodes: int) -> None:
        model_name = self.__get_model_name(n_train_episodes)
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "qlearning")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        q_table_save_path = os.path.join(base_dir, model_name + "_q_table")
        pickle.dump(self.q_table, open(q_table_save_path, "wb"))
        params_save_path = os.path.join(base_dir, model_name + "_params")
        pickle.dump(self.params, open(params_save_path, "wb"))
        print(f"Saved model {model_name} to {base_dir}.")

    # load from experiments folder
    def load(self, base_dir: str, model_name: str) -> None:
        q_table_save_path = os.path.join(base_dir, model_name + "_q_table")
        self.q_table = pickle.load(open(q_table_save_path, "rb"))
        params_save_path = os.path.join(base_dir, model_name + "_params")
        self.params = pickle.load(open(params_save_path, "rb"))
        print(f"Loaded model {model_name} from {base_dir}.")


def train_qlearning_agent(
    env: TradingEnv, agent: QLearningAgent, num_episodes: int = 200, training_batch_size: int = 500, plot: bool = True
) -> None:
    min_rewards = []
    max_rewards = []
    avg_rewards = []
    min_net_worths = []
    max_net_worths = []
    avg_net_worths = []
    final_net_worths = []

    t_start = time.time()
    for episode in tqdm(range(num_episodes), desc="Training Q-learning agent"):
        state = env.reset(env_step_size=training_batch_size)
        epsilon = agent.get_epsilon(episode)

        rewards = []
        net_worths = []

        for _ in range(training_batch_size):
            # Choose next action based on epsilon greedy
            action = agent.select_action(state, epsilon)

            # Take action, observe reward and next state
            next_state, reward, done = env.step(action)

            # Update agent's knowledge
            agent.update_experience(state, action, reward, next_state, episode)
            state = next_state

            rewards.append(reward)
            net_worths.append(env.net_worth)

            if done:
                break

        min_rewards.append(np.min(rewards))
        max_rewards.append(np.max(rewards))
        avg_rewards.append(np.mean(rewards))
        min_net_worths.append(np.min(net_worths))
        max_net_worths.append(np.max(net_worths))
        avg_net_worths.append(np.mean(net_worths))
        final_net_worths.append(net_worths[-1])
    t_end = time.time()

    print(f"Finished training")
    q_table_values = np.array(list(agent.q_table.values()))
    zero = np.all(q_table_values == np.zeros(3), axis=1)
    print(f"Number of non-zero q-values: {np.sum(~zero)}")
    if plot:
        fig, ax = plt.subplots(nrows=2, ncols=1)
        idx_rewards = 0
        idx_net_worths = 1
        ax[idx_rewards].plot(min_rewards, c="r", label="min rewards")
        ax[idx_rewards].plot(max_rewards, c="limegreen", label="max rewards")
        ax[idx_rewards].plot(avg_rewards, c="k", label="avg rewards")
        ax[idx_rewards].legend()
        ax[idx_net_worths].plot(min_net_worths, c="r", label="min net worths")
        ax[idx_net_worths].plot(max_net_worths, c="limegreen", label="max net worths")
        ax[idx_net_worths].plot(avg_net_worths, c="k", label="avg net worths")
        # ax[idx_net_worths].plot(final_net_worths, c="b", label="final net worths")
        ax[idx_net_worths].legend()
        plt.tight_layout()
        plt.show()

        final_profits = np.array(final_net_worths) - env.initial_balance
        print(f"Finished training for {num_episodes} episodes")
        print(f"Params:")
        print(str(agent.params))
        print()

        print(f"Total training time: {t_end - t_start:.3f} seconds")
        print(f"Average training time per episode: {(t_end - t_start) / num_episodes:.3f} seconds")
        print(f"Initial balance: {env.initial_balance}")
        print(f"Mean final net worth: {np.mean(final_net_worths):.3f}")
        print(f"Standard deviation of final net worth: {np.std(final_net_worths):.3f}")
        print(f"Best final net worth: {np.max(final_net_worths):.3f}")
        print(f"Mean profit: {np.mean(final_profits):.3f}")
        print(f"Standard deviation of profit: {np.std(final_profits):.3f}")
        print(f"Best profit: {np.max(final_profits):.3f}")


def test_q_learning_agent(
    env: TradingEnv, agent: QLearningAgent, num_episodes: int = 1, visualize: bool = True
) -> None:
    average_net_worth = 0
    average_orders = 0
    average_profit = 0
    no_profit_episodes = 0
    profit = 0

    for episode in tqdm(range(num_episodes), desc="Testing Q-learning agent"):
        state = env.reset()
        epsilon = agent.get_epsilon(1e6)

        while True:
            if visualize and episode == num_episodes - 1:
                env.render(visualize=True)

            # Choose next action based on epsilon greedy
            action = agent.select_action(state, epsilon)

            # Take action, observe reward and next state
            next_state, reward, done = env.step(action)

            state = next_state

            if env.current_step == env.end_step:
                # print("The model made it to the end of episode {} with {}.".format(episode, env.net_worth))
                break
            if done:
                # print("Episode {} finished with {}.".format(episode, env.net_worth))
                break

        profit = env.net_worth - env.initial_balance
        average_profit += profit
        average_net_worth += env.net_worth
        average_orders += env.episode_orders
        if env.net_worth < env.initial_balance:
            no_profit_episodes += 1

    print(f"Finished testing")
    print("Average net worth: {}".format(average_net_worth / num_episodes))
    print("Average profit: {}".format(average_profit / num_episodes))
    print("Average orders: {}".format(average_orders / num_episodes))
    print("No profit episodes: {}".format(no_profit_episodes))
    print("Profit: {}".format(profit))


if __name__ == "__main__":
    params = QLearningParams()
    env = TradingEnv(
        stock_data=yf.Ticker("MSFT"),
        period="5y",
        initial_balance=1000,
        lookback_window_size=params.lookback_window_size,
    )
    agent = QLearningAgent(env, params)
    try:
        n_train_episodes = 3000
        train_qlearning_agent(env, agent, num_episodes=n_train_episodes, plot=True)
        agent.save(n_train_episodes)
    except KeyboardInterrupt:
        print("Training stopped manually.")
    finally:
        print("Press any key to exit.")
        cv.waitKey(0)
        cv.destroyAllWindows()
