import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
import wandb

class MarketEnvironment:
    def __init__(self, time_horizon=100, total_shares=1000):
        csv_file = './taida_processed_2_days_data.csv'
        market_average_price_file = 'weighted_avg_price_2_days.csv'
        self.data = self.load_csv(csv_file)
        self.market_ave_data = self.load_csv(market_average_price_file)
        # import ipdb; ipdb.set_trace()
        print("CSV Column Headers:", self.data.columns.tolist())  # 打印欄位標題

        # Set the variables for the initial state
        self.totalShares = total_shares
        self.shares_remaining = 1
        self.totalCapture = 0
        self.timeHorizon = time_horizon
        self.logReturns = collections.deque(np.zeros(6))
        self.market_prices = []

        self.reset(i_episode=0)
        
    def load_csv(self, csv_file):
        return pd.read_csv(csv_file)

    def calculate_log_returns(self):
        prices = self.data['d']  # 成交價
        log_returns = np.log(prices / prices.shift(1)).fillna(0).to_numpy()
        return log_returns

    def reset(self, i_episode):
        self.current_time = i_episode * self.timeHorizon + 1
        self.market_average_price = self.market_ave_data['weighted_avg_price'][i_episode].item()
        self.state = np.array(list(self.logReturns) + [self.timeHorizon / self.timeHorizon, self.shares_remaining])
        self.trade_list = []
        self.done = False
        self.prevPrice = self.data['market_price'].iloc[0].item() # TODO
        self.currentPrice = self.data['market_price'].iloc[0].item() # TODO
        return self.state

    def start_transactions(self):
        print("Starting transactions...")

    def step(self, action):
        if self.done:
            raise ValueError("Environment has finished. Please reset.")

        shares_to_sell = min(action.item(), self.shares_remaining)

        # 更新市場價格
        self.currentPrice = self.data['market_price'].iloc[self.current_time].item()

        self.shares_remaining -= shares_to_sell
        self.totalCapture += shares_to_sell * self.totalShares * self.currentPrice
        self.trade_list.append((self.current_time, shares_to_sell))

        # Calculate the log return for the current step and save it in the logReturn deque
        self.logReturns.append(np.log(self.currentPrice/self.prevPrice))
        self.logReturns.popleft()

        # 更新 state
        self.market_prices.append(self.currentPrice)

        # 確保市場價格數量不超過6個，若不足則補0
        last_6_prices = self.market_prices[-6:]
        if len(last_6_prices) < 6:
            last_6_prices = [0] * (6 - len(last_6_prices)) + last_6_prices

        # Weighted average price calculation (if shares were sold)
        weighted_average_price = 0
        if len(self.trade_list) > 0:
            total_value_sold = sum([trade[1] * self.totalShares * self.data['market_price'].iloc[trade[0]] for trade in self.trade_list])
            total_shares_sold = sum([trade[1] * self.totalShares for trade in self.trade_list])
            weighted_average_price = total_value_sold / total_shares_sold if total_shares_sold > 0 else 0

        # Market average price (current market price)
        market_average_price = self.currentPrice

        reward = weighted_average_price - self.market_average_price
        print("weighted_average_price:", weighted_average_price)
        print("self.market_average_price:", self.market_average_price)
        

        # 更新狀態，將最近6個價格納入
        self.state = np.array(last_6_prices + [(self.timeHorizon - self.current_time) / self.timeHorizon, self.shares_remaining])
        
        print("self.current_time:", self.current_time)
        print("self.timeHorizon:", self.timeHorizon)
        print("self.shares_remaining:", self.shares_remaining)
        if self.current_time >= self.timeHorizon or self.shares_remaining <= 0:
            self.done = True

        wandb.log({
            "Total Capture (in millions)": self.totalCapture / 10**6,
            "Shares Remaining (in millions)": self.shares_remaining * self.totalShares / 10**6,
        })

        # print("Total Capture (in millions):", self.totalCapture / 10**6)
        # print("Shares Remaining (in millions):", self.shares_remaining / 10**6)

        self.prevPrice = self.currentPrice
        self.current_time += 1

        return self.state, reward, self.done, {}

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