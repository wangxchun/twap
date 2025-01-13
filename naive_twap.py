import numpy as np
from custom_market import MarketEnvironment
import wandb

class UniformSellingStrategy:
    def __init__(self, total_shares, time_horizon):
        self.total_shares = total_shares
        self.time_horizon = time_horizon
        self.shares_per_step = total_shares / time_horizon

    def run(self):
        env = MarketEnvironment(nums_day = 2, data_path = f'./train_data/taida_processed_{2}_days_data.csv', market_average_price_file = f'./train_data/weighted_avg_price_{2}_days.csv', time_horizon=self.time_horizon, total_shares=self.total_shares)
        state = env.reset(i_episode=0)

        # while not done:
        for t in range(self.time_horizon):
            action = np.array([self.shares_per_step / self.total_shares])  # 每個時間步驟均分的股票數
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                print("Episode finished")
                break            

        env.stop_transactions()

if __name__ == "__main__":
    total_shares = 1000
    time_horizon = 100

    wandb.init(project='Naive Twap', name="test")
    strategy = UniformSellingStrategy(total_shares=total_shares, time_horizon=time_horizon)
    strategy.run()
