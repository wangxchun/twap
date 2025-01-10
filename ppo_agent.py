import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super(PolicyNetContinuous, self).__init__()

        # Shared hidden layers
        fc_layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            fc_layers.append(torch.nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.hidden_layers = torch.nn.ModuleList(fc_layers)

        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.sigmoid(layer(x))  # Apply ReLU activation for each hidden layer
        mu = torch.sigmoid(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_layers):
        super(ValueNet, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.hidden_layers = torch.nn.ModuleList(layers)
        self.output_layer = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


class PPOAgent:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        # import ipdb; ipdb.set_trace()
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        # Revised
        action = torch.clamp(action, 0.0, 0.1)
        # return action.item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        rewards = (rewards + 8.0) / 8.0  # 对奖励进行归一化处理
        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - self.critic(states)
            advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            states_eps = 1e-8
            states = states + states_eps
            mu, std = self.actor(states)
            try:
                action_dists = torch.distributions.Normal(mu, std)
            except Exception as e:
                print("Error occurred while creating Normal distribution")
                print("states:", states)
                print("mu:", mu)
                print("std:", std)
                print("Exception:", e)
            log_probs = action_dists.log_prob(actions)
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
