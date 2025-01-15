import numpy as np
from custom_market import MarketEnvironment
import wandb
import argparse

class UniformSellingStrategy:
    def __init__(self, total_shares, time_horizon, nums_day, data_path, market_average_price_file):
        self.total_shares = total_shares
        self.time_horizon = time_horizon
        self.shares_per_step = total_shares / time_horizon
        self.nums_day = nums_day
        self.data_path = data_path
        self.market_average_price_file = market_average_price_file

    def run(self):
        env = MarketEnvironment(nums_day=self.nums_day, data_path=self.data_path, market_average_price_file=self.market_average_price_file, time_horizon=self.time_horizon, total_shares=self.total_shares)
        state = env.reset(i_episode=0)

        for t in range(self.time_horizon):
            action = np.array([self.shares_per_step / self.total_shares])
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                print("Episode finished")
                break

        env.stop_transactions()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uniform Selling Strategy")
    parser.add_argument('--nums_day', type=int, default=100, help='Number of days for training data')
    # parser.add_argument('--data_path', type=str, default='./test_data/taida_processed_25_days_data.csv', help='Path to training data CSV')
    # parser.add_argument('--market_average_price_file', type=str, default='./test_data/weighted_avg_price_25_days.csv', help='Path to market average price CSV')
    parser.add_argument('--data_path', type=str, default='./train_data_minutes/taida_processed_100_days_data.csv', help='Path to training data CSV')
    parser.add_argument('--market_average_price_file', type=str, default='./train_data_minutes/weighted_avg_price_100_days.csv', help='Path to market average price CSV')
    args = parser.parse_args()

    total_shares = 1000
    time_horizon = 100

    wandb.init(project='Naive Twap', name="test")
    strategy = UniformSellingStrategy(total_shares=total_shares, time_horizon=time_horizon, nums_day=args.nums_day, data_path=args.data_path, market_average_price_file=args.market_average_price_file)
    strategy.run()
