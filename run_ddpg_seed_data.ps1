# 定義不同的 --hidden-layers 配置
$hidden_layers_list = @("16")

# 固定參數設置
$agent_type = "ddpg"
$actor_lr = 1e-3
$critic_lr = 1e-2
$n_episodes = 10000
$hidden_layers = 128
$gamma = 0.98
$tau = 0.005
$buffer_size = 10000
$minimal_size = 1000
$batch_size = 64
$project_name = '01_13_custom_ddpg_10000ep_data_ave_savemodel'

# 定義 nums_day 的範圍
$nums_day_list = @(2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
# $nums_day_list = @(2)

# 循環執行
foreach ($nums_day in $nums_day_list) {
    # 動態生成對應的路徑
    $data_path = "./train_data/taida_processed_${nums_day}_days_data.csv"
    $market_average_price_file_path = "./train_data/weighted_avg_price_${nums_day}_days.csv"

    foreach ($hidden_layers in $hidden_layers_list) {
        # 將字串分割為陣列
        $hidden_layers_array = $hidden_layers.Split(" ")

        foreach ($seed in 1..5) {
            $wandb_run_name = "nums_day_${nums_day}"

            Write-Host "執行訓練: --hidden-layers $hidden_layers_array --seed $seed --nums-day $nums_day"

            # 傳遞參數，使用反引號進行多行命令
            python ./train8_rl_utils.py `
                --agent-type $agent_type `
                --actor-lr $actor_lr `
                --critic-lr $critic_lr `
                --n-episodes $n_episodes `
                --hidden-layers @($hidden_layers_array) `
                --gamma $gamma `
                --tau $tau `
                --buffer-size $buffer_size `
                --minimal-size $minimal_size `
                --batch-size $batch_size `
                --seed $seed `
                --project-name $project_name `
                --nums-day $nums_day `
                --data-path $data_path `
                --market-average-price-file-path $market_average_price_file_path `
                --wandb-run-name $wandb_run_name
        }
    }
}
