import torch as T
import torch.nn.functional as F
import numpy as np
from ounoise import OUActionNoise
from network import ActorNetwork, CriticNetwork
from replaybuffer import ReplayBuffer


class Agent():
    def __init__(self, alpha, beta, in_dims, tau, n_actions, gamma = 0.99,
                 max_size = 1000000, h_dims = [400,300], batch_size = 64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.memory = ReplayBuffer(max_size, in_dims, n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, in_dims, n_actions, h_dims, name='actor')
        self.critic = CriticNetwork(in_dims, n_actions, lr = beta, name='critic',h_dims = [400,300])
        self.target_actor = ActorNetwork(alpha, in_dims, n_actions, h_dims,  name='target_actor')
        self.target_critic = CriticNetwork(in_dims, n_actions,lr = beta, name='target_critic',h_dims = [400,300])

        self.update_network_params(tau=1)

    def choose_actions(self, obs):
        self.actor.eval()
        state = T.tensor([obs], dtype = T.float).to(self.actor.device)
        mu = self.actor(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype= T.float).to(self.actor.device)
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]
    

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transitions(state, action, reward, state_, done)


    def save_models(self):
        self.actor.save_chkpt()
        self.critic.save_chkpt()
        self.target_actor.save_chkpt()
        self.target_critic.save_chkpt()


    def load_models(self):
        self.actor.load_chkpt()
        self.critic.load_chkpt()
        self.target_actor.load_chkpt()
        self.target_critic.load_chkpt()

    def learn(self):
        if self.memory.cntr < self.batch_size:
            return
        
        states, actions, rewards,states_, dones = \
                    self.memory.sample_buffer(self.batch_size)
        
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)

        target_actions = self.target_actor(states_)
        critic_value_ = self.target_crtitic(states_, target_actions)
        critic_value = self.critic(states, actions)

        critic_value_[dones] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size,1)


        self.critic.optim.zero_grad()
        critic_loss = F.mse_loss(target, critic_value) 
        critic_loss.backward()
        self.critic.optim.step()

        self.actor.optim.zero_grad()
        actor_loss = -self.critic(states, self.actor(states))
        actor_loss = T.mean(actor_loss)                  
        actor_loss.backward()
        self.actor.optim.step()

        self.update_network_params()

    def update_network_params(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                    (1-tau)*target_critic_state_dict[name].clone()
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

        # With Batch Normalization
        # self.target_critic.load_state_dict(critic_state_dict, strict = False)
        # self.target_actor.load_state_dict(actor_state_dict, strict = False)











        