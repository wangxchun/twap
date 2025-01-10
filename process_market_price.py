import pandas as pd
import numpy as np
import argparse

# 設定命令列參數
parser = argparse.ArgumentParser(description="Calculate weighted average price from CSV data.")
parser.add_argument('--num-days', type=int, help="Number of days (N) for the data file")
args = parser.parse_args()

trading_type = 'sell'
num_step = 100

# 載入CSV檔案
N = args.num_days
csv_file = f'./train_data/taida_random_{N}_days_data.csv'
data = pd.read_csv(csv_file)

# 轉換時間為datetime格式
data['tt'] = pd.to_datetime(data['tt'], format='%H:%M:%S.%f')

# 篩選
# 设置参考时间为 09:00:00，转换为 datetime 格式
reference_time = pd.to_datetime('09:00:00', format='%H:%M:%S')
# 筛选出 09:00 之后的行
data = data[data['tt'] > reference_time]

# 1. 按日期分組
data['dd'] = pd.to_datetime(data['dd']).dt.date  # 提取日期部分
grouped = data.groupby('dd')

'''
grouped.size()
dd
2023-01-03    11639
2023-01-04    11180
2023-01-05    12043
2023-01-06    15889
2023-01-09    14118
              ...
2024-12-11    25609
2024-12-12    23829
2024-12-13    29102
2024-12-16    19429
2024-12-17    18107
Length: 471, dtype: int64
'''

# 2. 每天切割成100步
processed_data = []

for date, group in grouped:
    # 確保資料按時間排序
    group = group.sort_values('tt')

    # 計算每步的時間間隔（4.5小時/100步 = 160多秒鐘）
    start_time = group['tt'].iloc[0]
    end_time = group['tt'].iloc[-1]
    total_time = end_time - start_time
    time_interval_seconds = total_time.total_seconds()
    time_interval = time_interval_seconds / num_step

    # 3. 切割資料
    for i in range(num_step):
        step_start_time = start_time + pd.Timedelta(seconds=i * time_interval)
        step_end_time = step_start_time + pd.Timedelta(seconds=time_interval)
        
        # 篩選出在這個時間範圍內的資料
        step_data = group[(group['tt'] >= step_start_time) & (group['tt'] < step_end_time)]

        if not step_data.empty:
            # 4. 計算成交價
            if trading_type == 'sell':
                market_price = step_data['bp1']
            elif trading_type == 'buy':
                market_price = step_data['sp1']
            
            # 儲存每步的 market price
            processed_data.append({
                'date': date,
                'step': i,
                'start_time': step_start_time,
                'end_time': step_end_time,
                'market_price': market_price.mean()  # 儲存平均價格，根據需求調整
            })

# 轉換為 DataFrame
processed_df = pd.DataFrame(processed_data)
processed_df.to_csv(f'./train_data/taida_processed_{N}_days_data.csv', index=False)

# 1. 公司大概多少比例的交易會用到 WAP (預期回答，例如:一天交易額50億，大概10億用到WAP)
# 2. 除了黑K策略之外，還有哪些策略會用到 WAP
# 3. 使用的方式是交易員設定在 N 分鐘內賣出 M 張股票嗎
# 4. 如果是的話，N 是會變動的嗎，變動的範圍是多少，例如最短10分鐘 最長 4 小時