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
        env = MarketEnvironment(nums_day = args.nums_day, data_path = args.data_path, market_average_price_file = args.market_average_price_file, time_horizon=self.time_horizon, total_shares=self.total_shares)
        state = env.reset(i_episode=0)

        performance_list = []

        for i_episode in range(args.nums_day):
            state = env.reset(i_episode)
            done = False
            while not done:
                action = np.array([self.shares_per_step / self.total_shares])  # 每個時間步驟均分的股票數
                next_state, reward, done, info = env.step(action)

                # Collect performance data
                if done and not np.isinf(info["Performance"]):
                    performance_list.append(info["Performance"])
    
             # Log performance every nums_day episodes
            if (i_episode + 1) % args.nums_day == 0:
                    avg_performance = sum(performance_list) / len(performance_list) * 1e5
                    print("avg_performance: ", avg_performance)
                    performance_list = []
                        

        env.stop_transactions()

if __name__ == "__main__":
    total_shares = 1000
    time_horizon = 100

    parser = argparse.ArgumentParser()
    parser.add_argument("--nums-day", type=int, default=25, help="Number of episodes for testing")
    parser.add_argument('--data_path', type=str, default='./test_data/taida_processed_25_days_data.csv', help='Path to training data CSV')
    parser.add_argument('--market_average_price_file', type=str, default='./test_data/weighted_avg_price_25_days.csv', help='Path to market average price CSV')
    args = parser.parse_args()

    wandb.init(project='Naive Twap', name="test")
    strategy = UniformSellingStrategy(total_shares=total_shares, time_horizon=time_horizon)
    strategy.run(args)


