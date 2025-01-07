# 定義不同的 --hidden-layers 配置
$hidden_layers_list = @(
    "16"
    "16 16",
    "16 16 16",
    "32",
    "32 32",
    "32 32 32",
    "64",
    "64 64"
    "64 64 64",
    "128",
    "128 128",
    "128 128 128",
    "256",
    "256 256",
    "256 256 256"
    "512",
    "512 512",
    "512 512 512",
    "1024",
    "1024 1024",
    "1024 1024 1024"
)

# $hidden_layers_list = @(

#     # 小 -> 中 -> 大
#     "16 64 256",
#     "32 128 512",
#     "64 256 512",

#     # 大 -> 中 -> 小
#     "512 256 64",
#     "512 128 32",
#     "256 64 16",

#     # 中 -> 小 -> 大
#     "128 64 256",
#     "256 128 512",
#     "64 32 128",

#     # 大 -> 小 -> 中
#     "512 64 256",
#     "512 32 128",
#     "256 16 64",

#     # 隨機變化 (非線性增減)
#     "32 128 32",
#     "256 64 128",
#     "64 256 512"
# )


# 其他固定參數
$agent_type = "PPO"
$actor_lr = 1e-5
$critic_lr = 5e-5
$n_episodes = 
$hidden_layers = 128
$gamma = 0.98
$lmbda = 0.95
$ppo_epoch = 10
$epsilon = 0.2
$sigma = 0.01
$project_name = 'custom_ppo_arch'

# 循環執行
foreach ($hidden_layers in $hidden_layers_list) {
    # 將字串分割為陣列
    $hidden_layers_array = $hidden_layers.Split(" ")

    foreach ($seed in 1..5) {
        Write-Host "執行訓練: --hidden-layers $hidden_layers_array --seed $seed"

        # 傳遞參數，展開陣列中的值
        python ./train8_rl_utils.py `
           --agent-type $agent_type `
            --actor-lr $actor_lr `
            --critic-lr $critic_lr `
            --n-episodes $n_episodes `
            --hidden-layers @($hidden_layers_array) `
            --gamma $gamma `
            --lmbda $lmbda `
            --ppo-epoch $ppo_epoch `
            --epsilon $epsilon `
            --sigma $sigma `
            --seed $seed `
            --project-name $project_name
    }
}
