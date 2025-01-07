# 定義不同的 --hidden-layers 配置
$hidden_layers_list = @(
    "16",
    "16 16",
    "16 16 16",
    "32",
    "32 32",
    "32 32 32",
    "64",
    "64 64",
    "64 64 64",
    "128",
    "128 128",
    "128 128 128",
    "256",
    "256 256",
    "256 256 256",
    "512",
    "512 512",
    "512 512 512",
    "1024",
    "1024 1024",
    "1024 1024 1024"
)

# 固定參數設置
$agent_type = "DDPG"
$actor_lr = 1e-3
$critic_lr = 1e-2
$n_episodes = 200
$hidden_layers = 128
$gamma = 0.98
$tau = 0.005
$buffer_size = 10000
$minimal_size = 1000
$batch_size = 64
$project_name = 'custom_ddpg_arch'
$nums_day = 100
$data_path = $data_path = "./train_data/taida_processed_${nums_day}_days_data.csv"
$market_average_price_file_path = "./train_data/weighted_avg_price_${nums_day}_days.csv"

# 循環執行
foreach ($hidden_layers in $hidden_layers_list) {
    # 將字串分割為陣列
    $hidden_layers_array = $hidden_layers.Split(" ")

    foreach ($seed in 1..5) {
        Write-Host "執行訓練: --hidden-layers $hidden_layers_array --seed $seed"

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
            --market-average-price-file-path $market_average_price_file_path
    }
}
