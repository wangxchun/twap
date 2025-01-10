import numpy as np
from custom_market import MarketEnvironment
import wandb
from tqdm import tqdm
import argparse

class UniformSellingStrategy:
    def __init__(self, total_shares, time_horizon):
        self.total_shares = total_shares
        self.time_horizon = time_horizon
        self.shares_per_step = total_shares / time_horizon

    def run(self, args):
        env = MarketEnvironment(time_horizon=self.time_horizon, total_shares=self.total_shares)
        state = env.reset(i_episode=0)

        total_capture_list = []
        shares_remaining_list = []
        return_list = []
        performance_list = []

        with tqdm(total=num_episodes, desc='Training') as pbar:
            for i_episode in range(num_episodes):
                state = env.reset(i_episode)
                done = False
                episode_transitions = []
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, info = env.step(action)
                    episode_transitions.append((state, action, reward, next_state, done))
                    state = next_state

                    # Collect performance data
                    if done:
                        total_capture_list.append(info["Total Capture"])
                        shares_remaining_list.append(info["Shares Remaining"])
                        return_list.append(info["Reward"])
                        if not np.isinf(info["Performance"]):
                            performance_list.append(info["Performance"])     

        env.stop_transactions()

if __name__ == "__main__":
    total_shares = 1000
    time_horizon = 100

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default=f'test', help="Wandb Run Name")
    args = parser.parse_args()

    wandb.init(project='Naive Twap', name="test")
    strategy = UniformSellingStrategy(total_shares=total_shares, time_horizon=time_horizon)
    strategy.run(args)


