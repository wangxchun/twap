import random
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_layers, action_dim, action_bound):
        print("hidden_layers:", hidden_layers)
        super(Qnet, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.hidden_layers = torch.nn.ModuleList(layers)
        self.output_layer = torch.nn.Linear(input_dim, action_dim)

        self.action_bound = action_bound

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        # return torch.sigmoid(self.output_layer(x))
        return torch.tanh(self.output_layer(x)) * self.action_bound

class DQNAgent:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_layers, action_dim, action_bound, learning_rate, gamma, epsilon, target_update, device):
        
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_layers, self.action_dim, action_bound).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_layers, self.action_dim, action_bound).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.random()
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).item()  # 直接输出 [0, 1] 的值
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 計算當前狀態的 Q 值
        q_values = self.q_net(states)

        # 計算下一狀態的最大 Q 值
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states)

        # TD 誤差目標
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        # 計算損失（MSE 損失）
        dqn_loss = torch.mean((q_values - q_targets) ** 2)

        # 梯度更新
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # 每隔一定步數更新目標網路
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1

    def save_model(self, filepath):
        '''保存模型到指定路径'''
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'count': self.count
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        '''从指定路径加载模型'''
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.count = checkpoint.get('count', 0)  # 默认为0以防文件中没有保存计数器
        print(f"Model loaded from {filepath}")
