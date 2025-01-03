import pandas as pd

# 讀取CSV檔案
csv_file = './t06_2308_delta_2023_2024.csv'
data = pd.read_csv(csv_file)
num_rows = data.shape[0]
print(f"Number of rows: {num_rows}")

# 確保日期是 datetime 格式
data['dd'] = pd.to_datetime(data['dd'])

# 隨機選擇 N 天 
N = 2
random_days = data['dd'].drop_duplicates().sample(n=N, random_state=42)  # 這裡 random_state 用於設置隨機數種子，保證可重現

# 根據選定的日期篩選資料
random_data = data[data['dd'].isin(random_days)]
num_rows = random_data.shape[0]
print(f"Number of rows: {num_rows}")

# 查看篩選後的資料
random_data.to_csv(f'taida_random_{N}_days_data.csv', index=False)