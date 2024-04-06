import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class CriticNetwork(nn.Module):
    def __init__(self, in_dims, n_actions, lr, name, h_dims = [400,300], chkpt_dir = 'tmp/ddpg'):
        super().__init__()
        self.in_dims = in_dims
        self.n_actions = n_actions
        self.lr =lr
        self.h_dims = h_dims
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name + '_ddpg')

        self.fc1 = nn.Linear(*in_dims, h_dims[0])
        self.fc2 = nn.Linear(h_dims[0], h_dims[1])

        self.relu = nn.ReLU()
        self.bn1 = nn.LayerNorm(h_dims[0])
        self.bn2 = nn.LayerNorm(h_dims[1])

        self.action_value = nn.Linear(self.n_actions, h_dims[1])
        self.q = nn.Linear(h_dims[1], 1)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1,f1)


        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2,f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.weight.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4,f4)
        self.action_value.bias.data.uniform_(-f4, f4)


        self.optim= optim.Adam(self.parameters(), lr=lr, weight_decay = 0.1) 
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_val = self.fc1(state)
        state_val = self.bn1(state_val)
        state_val = self.relu(state_val)
        state_val = self.fc2(state_val)
        state_val = self.bn2(state_val)

        action_val = self.action_value(action)
        state_action_value = self.relu(T.add(state_val, action_val))

        q = self.q(state_action_value)
        return q
    
    def save_chkpt(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_chkpt(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, in_dims, n_actions, h_dims, name, chkpt_dir = 'tmp/ddpg'):
        super().__init__()
        self.in_dims = in_dims
        self.n_actions = n_actions
        self.h_dims = h_dims
        self.alpha = alpha
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name + '_ddpg')

        self.fc1 = nn.Linear(*in_dims, h_dims[0])
        self.fc2 = nn.Linear(h_dims[0], h_dims[1])

        self.bn1 = nn.LayerNorm(h_dims[0])
        self.bn2 = nn.LayerNorm(h_dims[1])
        self.relu = nn.ReLU()

        self.mu = nn.Linear(h_dims[1],n_actions)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1,f1)
        self.fc1.bias.data.uniform_(-f1,f1)

        
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2,f2)
        self.fc2.bias.data.uniform_(-f2,f2)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optim = optim.Adam(self.parameters(), lr = alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def new_method(self, n_actions):
        return n_actions

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = T.tanh(self.mu(x))
        return x

    def save_chkpt(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_chkpt(self):
        self.load_state_dict(T.load(self.chkpt_file))