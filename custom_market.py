import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
import wandb

class MarketEnvironment:
    def __init__(self, nums_day = 2, data_path = f'./train_data/taida_processed_{2}_days_data.csv', market_average_price_file = f'./train_data/weighted_avg_price_{2}_days.csv', time_horizon=100, total_shares=1000):
        self.data = self.load_csv(data_path)
        self.market_ave_data = self.load_csv(market_average_price_file)
        self.nums_day = nums_day
        print("CSV Column Headers:", self.data.columns.tolist())  # 打印欄位標題

        # Set the variables for the initial state
        self.totalShares = total_shares
        self.timeHorizon = time_horizon

        self.reset(i_episode=0)
        
    def load_csv(self, csv_file):
        return pd.read_csv(csv_file)

    def calculate_log_returns(self):
        prices = self.data['d']  # 成交價
        log_returns = np.log(prices / prices.shift(1)).fillna(0).to_numpy()
        return log_returns

    def reset(self, i_episode):
        self.shares_remaining = 1
        self.totalCapture = 0
        self.logReturns = collections.deque(np.zeros(6))
        self.market_prices = []

        self.current_time = i_episode * self.timeHorizon
        self.market_average_price = self.market_ave_data['weighted_avg_price'][i_episode % self.nums_day].item()
        self.state = np.array(list(self.logReturns) + [self.timeHorizon / self.timeHorizon, self.shares_remaining])
        self.trade_list = []
        self.done = False
        self.prevPrice = self.data['market_price'].iloc[self.current_time % (self.nums_day * self.timeHorizon)].item() # TODO
        self.currentPrice = self.data['market_price'].iloc[self.current_time % (self.nums_day * self.timeHorizon)].item() # TODO
        return self.state

    def start_transactions(self):
        print("Starting transactions...")

    def step(self, action):
        info = {}

        if self.done:
            raise ValueError("Environment has finished. Please reset.")

        shares_to_sell = min(action.item(), self.shares_remaining)

        # 更新市場價格
        # self.currentPrice = self.data['market_price'].iloc[self.current_time % (self.nums_day * self.timeHorizon)].item()
        self.currentPrice = self.data['market_price'].iloc[self.current_time // self.timeHorizon]

        self.shares_remaining -= shares_to_sell
        self.totalCapture += shares_to_sell * self.totalShares * self.currentPrice
        self.trade_list.append((self.current_time, shares_to_sell))

        # Update log returns but do it only if not done yet
        self.logReturns.append(np.log(self.currentPrice/self.prevPrice))
        self.logReturns.popleft()

        # Calculate the weighted average price and market average price only after checking if done
        weighted_average_price = 0
        if len(self.trade_list) > 0:
            total_value_sold = sum([trade[1] * self.totalShares * self.data['market_price'].iloc[trade[0] // self.timeHorizon] for trade in self.trade_list])
            total_shares_sold = sum([trade[1] * self.totalShares for trade in self.trade_list])
            weighted_average_price = total_value_sold / total_shares_sold if total_shares_sold > 0 else 0

        reward =  weighted_average_price
        print("self.shares_remaining:", self.shares_remaining)

        # 更新狀態，將最近6個價格納入
        logReturns_list = [float(x) for x in self.logReturns]
        self.state = np.array(logReturns_list + [(self.timeHorizon - self.current_time) / self.timeHorizon, self.shares_remaining])
        
        # Check for termination condition
        if (self.current_time + 1) % self.timeHorizon == 0 or self.shares_remaining <= 0:
            if self.shares_remaining > 0.5:
                reward = -1000
            performance = (weighted_average_price - self.market_average_price) / self.market_average_price
            print("weighted_average_price:", weighted_average_price)
            print("market_average_price:", self.market_average_price)
            print("performance:", performance * 1e5)
            self.done = True

            # wandb.log({
            #     "Total Capture": self.totalCapture,
            #     "Shares Remaining": self.shares_remaining,
            #     "Reward": reward,
            #     # "Performance (bp)": performance * 1e5,
            # })

            info = {"Performance": performance}

        # Update previous price and increment time after checking done condition
        self.prevPrice = self.currentPrice
        self.current_time += 1

        return self.state, reward, self.done, info


    def get_trade_list(self):
        return self.trade_list

    def observation_space_dimension(self):
        return len(self.state)

    def action_space_dimension(self):
        return 1

    def stop_transactions(self):
        print("Stopping transactions...")

    def plot_and_save_curves(self, filename='transaction_curves.png'):
        times, trades = zip(*self.trade_list)
        plt.plot(times, trades, marker='o')
        plt.xlabel('Time Step')
        plt.ylabel('Shares Sold')
        plt.title('Transaction Curves')
        plt.savefig(filename)
        plt.show()

        with open('./trade_list.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            # If the file is empty, write the header
            if file.tell() == 0:
                writer.writerow(['Time Step', 'Shares Sold'])
            for time, trade in self.trade_list:
                writer.writerow([time, trade])