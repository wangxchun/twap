import pandas as pd
import argparse

# 設定命令列參數
parser = argparse.ArgumentParser(description="Calculate weighted average price from CSV data.")
parser.add_argument('--num-days', type=int, default=100, help="Number of days (N) for the training data file")
parser.add_argument('--test-days', type=int, default=25, help="Number of days (M) for the test data file")
args = parser.parse_args()

# 讀取 CSV 檔案
csv_file = './train_data/t06_2308_delta_2023_2024.csv'
data = pd.read_csv(csv_file)
num_rows = data.shape[0]
print(f"Number of rows in original data: {num_rows}")

# 確保日期是 datetime 格式
data['dd'] = pd.to_datetime(data['dd'])

# 隨機選擇 N 天作為訓練資料
N = args.num_days
random_state = 42  # 固定隨機數種子，保證可重現
random_train_days = data['dd'].drop_duplicates().sample(n=N, random_state=random_state)

# 排除選出的 N 天，從剩下的日期中隨機選擇 M 天作為測試資料
remaining_days = pd.Index(data['dd'].drop_duplicates()).difference(random_train_days)
M = args.test_days
random_test_days = remaining_days.to_series().sample(n=M, random_state=random_state + 1)

# 根據選定的日期篩選資料
train_data = data[data['dd'].isin(random_train_days)]
test_data = data[data['dd'].isin(random_test_days)]

# 輸出檔案
test_output_file = f'./test_data/taida_random_{M}_days_test_data.csv'
test_data.to_csv(test_output_file, index=False)

# 確認輸出訊息
print(f"Test data saved to {test_output_file}, number of rows: {test_data.shape[0]}")
