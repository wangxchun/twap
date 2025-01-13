import random
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

########## Basic Network ########## 

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_layers, action_dim, action_bound):
        """
        PolicyNet allows custom number and size of hidden layers.
        :param state_dim: Dimension of input state.
        :param hidden_layers: List containing the number of neurons in each hidden layer.
        :param action_dim: Dimension of action output.
        :param action_bound: Maximum value of actions allowed by the environment.
        """
        super(PolicyNet, self).__init__()
        layers = []
        input_dim = state_dim

        # Add hidden layers dynamically
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.hidden_layers = torch.nn.ModuleList(layers)
        self.output_layer = torch.nn.Linear(input_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x): # x.size() = 8879996 ??
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return torch.tanh(self.output_layer(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_layers, action_dim):
        """
        QValueNet allows custom number and size of hidden layers.
        :param state_dim: Dimension of input state.
        :param action_dim: Dimension of input action.
        :param hidden_layers: List containing the number of neurons in each hidden layer.
        """
        super(QValueNet, self).__init__()
        layers = []
        input_dim = state_dim + action_dim

        # Add hidden layers dynamically
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.hidden_layers = torch.nn.ModuleList(layers)
        self.output_layer = torch.nn.Linear(input_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # Concatenate state and action
        for layer in self.hidden_layers:
            cat = F.relu(layer(cat))
        return self.output_layer(cat)
    
########## DDPG ##########

class DDPGAgent:
    ''' DDPG '''
    def __init__(self, state_dim, hidden_layers, action_dim, action_bound, test=False, load_path_actor=None, load_path_critic=None, sigma=None, actor_lr=None, critic_lr=None, tau=None, gamma=None, device='cuda'):
        if test == False:
            self.actor = PolicyNet(state_dim, hidden_layers, action_dim, action_bound).to(device)
            self.critic = QValueNet(state_dim, hidden_layers, action_dim).to(device)
            self.target_actor = PolicyNet(state_dim, hidden_layers, action_dim, action_bound).to(device)
            self.target_critic = QValueNet(state_dim, hidden_layers, action_dim).to(device)
            # Initialize the target value network and set it to the same parameters as the value network
            self.target_critic.load_state_dict(self.critic.state_dict())
            # Initialize the target policy network and set it to the same parameters as the policy network
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
            self.gamma = gamma
            self.sigma = sigma  # Standard deviation for Gaussian noise, with the mean set to 0
            self.tau = tau  # Soft update parameter for the target network
            self.action_dim = action_dim
            self.device = device
            self.action_bound = action_bound

        else:
            self.actor = PolicyNet(state_dim, hidden_layers, action_dim, action_bound).to(device)
            self.critic = QValueNet(state_dim, hidden_layers, action_dim).to(device)
            self.target_actor = PolicyNet(state_dim, hidden_layers, action_dim, action_bound).to(device)
            self.target_critic = QValueNet(state_dim, hidden_layers, action_dim).to(device)
            self.device = device
            self.action_bound = action_bound

            # Load pre-trained models using load_model method
            self.load_model(load_path_actor, load_path_critic)

            # Set the networks to evaluation mode
            self.actor.eval()
            self.critic.eval()

    # def take_action(self, state):
    #     state = np.array(state)
    #     state = torch.tensor(state, dtype=torch.float).to(self.device)
    #     # import ipdb; ipdb.set_trace()
    #     action = self.actor(state).item()
    #     # Add noise to actions to encourage exploration
    #     action = np.clip(action + self.sigma * np.random.randn(self.action_dim), 0, self.action_bound)
    #     return action

    def take_action(self, state, test=False):
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        if not test:
            # Add noise to actions to encourage exploration during training
            action += self.sigma * np.random.randn(self.action_dim)
        action = np.clip(action, 0, self.action_bound)  # Ensure action is within bounds
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # soft update policy network
        self.soft_update(self.critic, self.target_critic)  # soft update value network

    def save_model(self, actor_path, critic_path):
        """Saves the actor and critic network weights to the specified paths."""
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Actor model saved to {actor_path}")
        print(f"Critic model saved to {critic_path}")

    def load_model(self, actor_path, critic_path):
        """Loads the actor and critic network weights from the specified paths."""
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        print(f"Actor model loaded from {actor_path}")
        print(f"Critic model loaded from {critic_path}")


########## DDPG ########## 

"""
Price Matrix:
 [[100.5 101.  101.5 102.  102.5]
  [103.5 104.  104.5 105.  105.5]]

Order Volume Matrix:
 [[50 45 40 35 30]
  [25 20 15 10  5]]
  
"""