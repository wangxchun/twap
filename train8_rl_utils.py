import argparse
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from ddpg_agent import DDPGAgent
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent
from model import Actor, Critic
import utils
import rl_utils
from rl_utils import str2bool
import random 

def train_agent(env, args, device):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    replay_buffer = rl_utils.ReplayBuffer(args.buffer_size)
    state_dim = env.observation_space_dimension()
    action_dim = env.action_space_dimension()
    action_bound = 0.1
    if args.agent_type == 'DDPG':
        agent = DDPGAgent(state_dim, args.hidden_layers, action_dim, action_bound, args.sigma, args.actor_lr, args.critic_lr, args.tau, args.gamma, device)
    if args.agent_type == 'DQN':
        agent = DQNAgent(state_dim, args.hidden_layers, action_dim,  action_bound, args.lr, args.gamma, args.epsilon, args.target_update, device)
    if args.agent_type == 'PPO':
        agent = PPOAgent(state_dim, args.hidden_layers, action_dim, args.actor_lr, args.critic_lr, args.lmbda, args.ppo_epoch, args.epsilon, args.gamma, device)

    return_list = rl_utils.train_off_policy_agent(env, agent, args.n_episodes, replay_buffer, args.minimal_size, args.batch_size, save_interval=args.save_interval)
    
    # env.plot_and_save_curves()

    return return_list


def test_ddpg(env, args, device):
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space_dimension()
    action_dim = env.action_space_dimension()
    action_bound = 0.1
    if args.agent_type == 'DDPG':
        agent = DDPGAgent(state_dim, args.hidden_layers, action_dim, action_bound)
    if args.agent_type == 'DQN':
        agent = DQNAgent(state_dim, args.hidden_layers, action_dim,  action_bound)
    if args.agent_type == 'PPO':
        agent = DQNAgent(state_dim, args.hidden_layers, action_dim)
    agent.load_model(actor_path=args.load_path_actor, critic_path=args.load_path_critic)

    total_rewards = []

    for episode in range(args.n_test_episodes):
        state = env.reset()  # Reset the environment for each episode
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                action = agent.take_action(state)  # Select action without noise
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        print(f'Episode {episode + 1}: Total Reward = {episode_reward:.2f}')

    avg_reward = sum(total_rewards) / args.n_test_episodes
    print(f'Average Reward over {args.n_test_episodes} episodes: {avg_reward:.2f}')
    return total_rewards, avg_reward

# Define a function to parse hidden_layers
def parse_hidden_layers(s):
    """Parse a string of integers separated by commas into a list of integers."""
    return list(map(int, s.split(",")))

def main(args):
    
    if args.train_or_test == 'train':
        # Initialize Weights & Biases
        # name=",".join(map(str, args.hidden_layers)
        wandb.init(project=args.project_name, config=args, name=args.wandb_run_name)
        
        # Load environment parameters
        env = utils.get_env_param(args.nums_day, args.data_path, args.market_average_price_file_path)
        env_name = 'Stocks Trading'

        # Train the agent
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return_list = train_agent(env, args, device=device)

        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DDPG on {}'.format(env_name))
        plt.savefig('Returns.png')

        # Finish the wandb session
        wandb.finish()

    elif args.train_or_test == 'test':
        # Initialize Weights & Biases
        wandb.init(project="Test_DDPG_Trading_Agent", config=args)

        # Load environment parameters
        env, fp_table, acp_table = utils.get_env_param()
        env_name = 'Stocks Trading'

        # Test the agent
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        total_rewards, avg_reward = test_ddpg(env, args, device=device)

        # Finish the wandb session
        wandb.finish()

    else:
        print("Error")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="TradingEnv", help="Environment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--buffer-size", type=int, default=int(1e5), help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor")
    parser.add_argument("--n-episodes", type=int, default=100, help="Number of episodes for training")

    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[128], help="Hidden Layers")
    parser.add_argument("--minimal-size", type=int, default=100, help="Minimal Size")
    parser.add_argument("--sigma", type=float, default=0.01, help="sigma") # 高斯噪声标准差
    parser.add_argument("--save-interval", type=int, default=100, help="Save Interval")
    parser.add_argument("--project-name", type=str, default="DDPG_Seed", help="Project Name")

    # TEST
    parser.add_argument("--train-or-test", type=str, default='train', help="Train or Test")
    parser.add_argument("--n-test-episodes", type=int, default=10, help="Number of episodes for testing")

    # Agent Type
    parser.add_argument("--agent-type", type=str, default='DDPG', help="Agnet Type")

    # DQN & PPO
    parser.add_argument("--epsilon", type=float, default=0.2, help="epsilon")

    # DQN
    parser.add_argument("--lr", type=float, default=2e-3, help="lr")
    parser.add_argument("--target_update", type=int, default=10, help="Target Update")

    # PPO
    parser.add_argument("--lmbda", type=float, default=0.9, help="lmbda")
    parser.add_argument("--ppo-epoch", type=int, default=10, help="PPO Epoch")

    # DDPG
    parser.add_argument("--tau", type=float, default=1e-3, help="Soft update parameter")
    
    # DDPG & PPO
    parser.add_argument("--actor-lr", type=float, default=1e-4, help="Learning rate for actor")
    parser.add_argument("--critic-lr", type=float, default=5e-3, help="Learning rate for critic")

    # custom market
    parser.add_argument("--nums-day", type=int, default=2, help="How many days of data is used to train?")
    parser.add_argument("--data-path", type=str, default=f'./train_data/taida_processed_{2}_days_data.csv', help="Data Path")
    parser.add_argument("--market-average-price-file-path", type=str, default=f'./train_data/weighted_avg_price_{2}_days.csv', help="Market Average Price File Path")
    parser.add_argument("--wandb-run-name", type=str, default=f'test', help="Wandb Run Name")

    args = parser.parse_args()
    main(args)

