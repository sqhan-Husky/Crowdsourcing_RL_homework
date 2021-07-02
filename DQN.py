import torch
import torch.nn as nn
import numpy as np
import collections
from Embedding import Net

class DQN():
    def __init__(self, seq_len, n_actions, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
        self.seq_len = seq_len    # max_padding_len before input
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity

        self.eval_net = Net()
        self.target_net = Net()

        self.learn_step_counter = 0    # for target updating
        self.memory_counter = 0        # for counting records
        self.memory = np.zeros((memory_capacity, seq_len*2 + 2))   # store records
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()

    def choose_action(self, state, exist_pjs):
        # for one timestep per user
        state = torch.unsqueeze(torch.LongTensor(state), 0)    # [len(actions)] -> [1, len(actions)]
        # exist_pjs needs binary list
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net.forward(state)
            actions_value = actions_value.mul(torch.LongTensor(exist_pjs))
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.ranint(0, self.n_actions)
            while exist_pjs[action] == 0:
                action = np.random.ranint(0, self.n_actions)

        state_next = self.update_state(state, action)
        return action, state_next

    def update_state(self, state, action):
        # seq_update
        current_seq = state[0:self.seq_len]
        if current_seq[self.seq_len-1] != 0:
            seq_next = np.append(state, action)[1:]
        else:
            timestep = np.argwhere(current_seq == 0)[0][0]   # find first non-zero step
            current_seq[timestep] = action
            seq_next = current_seq

        # todo: histo_update
        current_histo = state[self.seq_len:]
        histo_next = current_histo

        state_next = np.hstack(seq_next, histo_next)
        return state_next

    def store_transition(self, state, action, reward, s_next):
        transition = np.hstack((state, [action, reward], s_next))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(collections.OrderedDict(self.eval_net.state_dict()))
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_capacity, self.batch_size)  # capacity >= bs
        b_memory = self.memory[sample_index]
        b_state = torch.LongTensor(b_memory[:, : self.seq_len])
        b_action = torch.LongTensor(b_memory[:, self.seq_len: self.seq_len + 1])
        b_reward = torch.FloatTensor(b_memory[:, self.seq_len + 1: self.seq_len + 2])
        b_state_next = torch.LongTensor(b_memory[:, -self.seq_len:])

        q_eval = self.eval_net(b_state).gather(dim=1, index=b_action)  # gather ?
        q_next = self.target_net(b_state_next).detach()

        q_target = b_reward + self.gamma * q_next.max(dim=1)[0].view(self.batch_size, 1)

        loss = self.loss_function(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TODO: tensorboard - loss



