import pandas as pd
import argparse

# 設定命令列參數
parser = argparse.ArgumentParser(description="Calculate weighted average price from CSV data.")
parser.add_argument('--num-days', type=int, default=100, help="Number of days (N) for the data file")
args = parser.parse_args()

# 讀取CSV檔案
csv_file = './train_data/t06_2308_delta_2023_2024.csv'
data = pd.read_csv(csv_file)
num_rows = data.shape[0]
print(f"Number of rows: {num_rows}")

# 確保日期是 datetime 格式
data['dd'] = pd.to_datetime(data['dd'])

# 隨機選擇 N 天 
N = args.num_days
random_days = data['dd'].drop_duplicates().sample(n=N, random_state=42)  # 這裡 random_state 用於設置隨機數種子，保證可重現

# 根據選定的日期篩選資料
random_data = data[data['dd'].isin(random_days)]
num_rows = random_data.shape[0]
print(f"Number of rows: {num_rows}")

# 查看篩選後的資料
random_data.to_csv(f'./train_data/taida_random_{N}_days_data_check.csv', index=False)