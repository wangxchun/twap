# 固定參數設置
$agent_type = "ddpg"
$hidden_layers = "16"
$test_nums_day = 25
$train_nums_day = 100
$data_path = "./test_data/taida_processed_${test_nums_day}_days_data.csv"
$market_average_price_file_path = "./test_data/weighted_avg_price_${test_nums_day}_days.csv"
$n_test_episodes = 100
$n_episodes_list = @(200, 500, 1000, 5000, 10000)
$project_name = 'Test_DDPG_ep_new_'

# 循環執行
foreach ($n_episodes in $n_episodes_list) {
    foreach ($seed in 1..5) {
        $wandb_run_name = "ep_${n_episodes}"

        # 設置模型載入路徑
        $load_path_actor = "./model/ddpg_actor_ep_${n_episodes}_day_${train_nums_day}_seed_${seed}.pth"
        $load_path_critic = "./model/ddpg_critic_ep_${n_episodes}_day_${train_nums_day}_seed_${seed}.pth"


        Write-Host "執行測試: --hidden-layers $hidden_layers --n-episodes $n_episodes --seed $seed"

        # 傳遞參數，使用反引號進行多行命令
        python ./train8_rl_utils.py `
            --agent-type $agent_type `
            --train-or-test "test" `
            --n-episodes $n_episodes `
            --n-test-episodes $n_test_episodes `
            --hidden-layers $hidden_layers `
            --seed $seed `
            --project-name $project_name `
            --data-path $data_path `
            --market-average-price-file-path $market_average_price_file_path `
            --nums-day $test_nums_day `
            --load-path-actor $load_path_actor `
            --load-path-critic $load_path_critic `
            --wandb-run-name $wandb_run_name `       
    }
}
