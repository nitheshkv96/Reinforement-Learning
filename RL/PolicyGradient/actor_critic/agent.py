import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_h = 256, fc2_h = 256):
        super().__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_h)
        self.fc2 = nn.Linear(fc1_h, fc2_h)
        self.pi = nn.Linear(fc2_h, n_actions)
        self.v = nn.Linear(fc2_h, 1)


        self.optim = optim.Adam(self.parameters(), lr = lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)

        return (pi, v)
    

class TDAgent:
    def __init__(self, input_dims, lr, n_actions, fc1_h, fc2_h,  gamma = 0.99):
        self.gamma = gamma
        self.lr = lr
        self.log_prob = None
        self.fc1_h = fc1_h
        self.fc2_h = fc2_h
        self.actor_critic = ActorCriticNetwork(self.lr, input_dims, n_actions)

    def choose_action(self, obs):
        state = T.tensor([obs]).to(self.actor_critic.device)
        probs, _ = self.actor_critic(state)
        probs = F.softmax(probs, dim = 1)
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        self.log_prob = action_probs.log_prob(action)

        return action.item()
    

    def learn(self, state, reward, state_, done):
        self.actor_critic.optim.zero_grad()
        
        state = T.tensor([state], dtype = T.float).to(self.actor_critic.device)
        state_ = T.tensor([state_], dtype = T.float).to(self.actor_critic.device)
        reward = T.tensor(reward, dtype = T.float).to(self.actor_critic.device)

        _, critic_v = self.actor_critic(state) 
        _, critic_v_ = self.actor_critic(state_)


        delta = reward + self.gamma*critic_v_*(1-int(done)) - critic_v

        actor_loss = -self.log_prob*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor_critic.optim.step()

































