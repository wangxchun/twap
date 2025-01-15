import pandas as pd
import argparse

# 設定命令列參數
parser = argparse.ArgumentParser(description="Calculate weighted average price from CSV data.")
parser.add_argument('--num-days', type=int, default=100, help="Number of days (N) for the data file")
args = parser.parse_args()

# 載入CSV檔案
N = args.num_days
csv_file = f'../test_data/taida_random_{N}_days_test_data.csv'
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

# Define function to calculate weighted average price for each group
def calculate_weighted_avg(group):
    weighted_price = 0
    total_volume_change = 0
    previous_v = group['v'].iloc[0]  # Start with the first row's volume
    
    for _, row in group.iterrows():
        volume_change = abs(row['v']) - abs(previous_v)  # Calculate volume change (absolute difference)
        weighted_price += volume_change * row['d']  # Weighted price
        total_volume_change += volume_change  # Total volume change
        previous_v = row['v']  # Update previous volume
    
    # Calculate the weighted average price
    weighted_avg_price = weighted_price / total_volume_change if total_volume_change != 0 else 0
    return weighted_avg_price

# Group by date ('dd') and apply the weighted average calculation
result = grouped.apply(calculate_weighted_avg).reset_index(name='weighted_avg_price')

# Output the result
print(result)

# Save the result to a CSV file
# output_file = f'../train_data/weighted_avg_price_{N}_days.csv'
output_file = f'../test_data/weighted_avg_price_{N}_days.csv'
result.to_csv(output_file, index=False)

print(f"Result saved to {output_file}")