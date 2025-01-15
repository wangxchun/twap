from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import wandb
import os

def str2bool(v):
    return v.lower() == "true"

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_off_policy_agent(env, agent, nums_day, num_episodes, replay_buffer, minimal_size, batch_size, seed, save_path_actor=None, save_path_critic=None, save_interval=100, load_actor_path=None, load_critic_path=None):

    # 確保 model 資料夾存在
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Get Agent Type
    agent_str = str(agent)
    start = agent_str.find('<') + 1
    end = agent_str.find('.')
    if start != -1 and end != -1:
        module_name = agent_str[start:end]
        agent_type = module_name.split('_')[0]

    # Load model if paths are provided
    if load_actor_path and load_critic_path:
        agent.load_model(load_actor_path, load_critic_path)

    total_capture_list = []
    shares_remaining_list = []
    return_list = []
    performance_list = []

    with tqdm(total=num_episodes, desc='Training') as pbar:
        for i_episode in range(num_episodes):
            # episode_return = 0
            state = env.reset(i_episode%2)
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, info = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                # episode_return += reward
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    agent.update(transition_dict)

                # Collect performance data
                if done:
                    total_capture_list.append(info["Total Capture"])
                    shares_remaining_list.append(info["Shares Remaining"])
                    return_list.append(info["Reward"])
                    if not np.isinf(info["Performance"]):
                        performance_list.append(info["Performance"])

             # Log performance every nums_day episodes
            if (i_episode + 1) % nums_day == 0:
                    avg_total_capture = sum(total_capture_list) / len(total_capture_list)
                    avg_shares_remaining = sum(shares_remaining_list) / len(shares_remaining_list)
                    avg_return = sum(return_list) / len(return_list)
                    avg_performance = sum(performance_list) / len(performance_list) * 1e5
                    print("avg_performance: ", avg_performance)
                    wandb.log({
                        "Total Capture": avg_total_capture,
                        "Shares Remaining": avg_shares_remaining,
                        "Return": avg_return,
                        "Average Performance": avg_performance,
                    })
                    total_capture_list = []
                    shares_remaining_list = []
                    return_list = []
                    performance_list = []

            pbar.update(1)

            # Save model at specified intervals
            if (i_episode + 1) % save_interval == 0:
                if agent_type == 'ddpg':
                    save_path_actor = os.path.join(model_dir, f"ddpg_actor_ep_{num_episodes}_day_{nums_day}_seed_{seed}.pth")
                    save_path_critic = os.path.join(model_dir, f"ddpg_critic_ep_{num_episodes}_day_{nums_day}_seed_{seed}.pth")
                    agent.save_model(save_path_actor, save_path_critic)
                    print(f"Model saved to: {save_path_actor} and {save_path_critic}")
                if agent_type == 'dqn' or agent_type == 'ppo':
                    save_path = os.path.join(model_dir, f"{agent_type}_ep_{num_episodes}_day_{nums_day}") 
                    agent.save_model(save_path)
                    print(f"Model saved to: {save_path}")

    return return_list

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)
                