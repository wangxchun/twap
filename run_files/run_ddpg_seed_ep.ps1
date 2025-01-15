# 固定參數設置
$agent_type = "ddpg"
$actor_lr = 1e-3
$critic_lr = 1e-2
$hidden_layers = 128
$gamma = 0.98
$tau = 0.005
$buffer_size = 10000
$minimal_size = 1000
$batch_size = 64
$project_name = '01_13_custom_ddpg_ep_ave_savemodel'
$nums_day = 100
$data_path = "./train_data/taida_processed_${nums_day}_days_data.csv"
$market_average_price_file_path = "./train_data/weighted_avg_price_${nums_day}_days.csv"

# 定義 n_episodes 數量
$n_episodes_list = @(200, 500, 1000, 5000, 10000)
$hidden_layers_list = @("16")

# 循環執行
foreach ($hidden_layers in $hidden_layers_list) {
    # 將字串分割為陣列
    $hidden_layers_array = $hidden_layers.Split(" ")

    foreach ($n_episodes in $n_episodes_list) {
        foreach ($seed in 1..5) {
            $wandb_run_name = "episodes_${n_episodes}"
            
            Write-Host "執行訓練: --hidden-layers $hidden_layers_array --n-episodes $n_episodes --seed $seed"

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
