import numpy as np


class ReplayBuffer():
    def __init__(self, mem_size, input_shape, n_actions):
        self.mem_size = mem_size
        self.cntr = 0

        self.states = np.zeros((mem_size,*input_shape), dtype=np.float)
        self.actions = np.zeros((mem_size, n_actions), dtype=np.float)
        self.states_ = np.zeros((mem_size, *input_shape), dtype=np.float)
        self.rewards = np.zeros(mem_size, dtype=np.float)
        self.dones = np.zeros(mem_size, dtype=np.bool)


    def store_transitions(self, state, action, reward, state_, done,):
        idx = self.cntr % self.mem_size
        self.states[idx] = state
        self.rewards[idx] = reward
        self.states_[idx] = state_
        self.actions[idx] = action
        self.dones[idx] = done

        self.cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        states_ = self.states_[batch]
        dones = self.dones[batch]


        return (states, actions, rewards, states_, dones)