import torch
import torch.nn as nn
import numpy as np
import collections
from Embedding import Net
# import random


class DQN():
    def __init__(self, seq_len, histo_len, n_actions, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
        self.seq_len = seq_len    # max_padding_len before input
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity
        self.histo_len = histo_len
        self.task_len = 83
        self.state_len = 1+self.seq_len*3+ self.task_len*3
        #net 参数
        self.max_pj = 2600
        self.embedding_dim = 30
        self.rnn_hidden_dim = 100
        self.n_layers = 5
        self.histo_len = 36
        self.eval_net = Net(self.max_pj, self.embedding_dim, self.seq_len, self.n_actions,
                            self.histo_len,self.rnn_hidden_dim,self.n_layers).cuda()
        self.target_net = Net(self.max_pj, self.embedding_dim, self.seq_len, self.n_actions,
                              self.histo_len,self.rnn_hidden_dim,self.n_layers).cuda()
        # self.eval_net = Net(self.max_pj, self.embedding_dim, self.seq_len, self.n_actions,
        #                     self.histo_len, self.rnn_hidden_dim, self.n_layers)
        # self.target_net = Net(self.max_pj, self.embedding_dim, self.seq_len, self.n_actions,
        #                       self.histo_len, self.rnn_hidden_dim, self.n_layers)
        self.learn_step_counter = 0    # for target updating
        self.memory_counter = 0        # for counting records
        #self.memory = []   # store records
        self.memory = np.zeros((memory_capacity, (1+seq_len*3+self.task_len*3) * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()

    def choose_action(self, state, exist_pjsl):
        # for one timestep per user
        #todo :cuda
        state_cuda = torch.unsqueeze(torch.LongTensor(state).cuda(), 0)# [len(actions)] -> [1, len(actions)]
        # state_cuda = torch.unsqueeze(torch.LongTensor(state), 0)
        # exist_pjs needs binary list
        np.random.seed(5)
        if np.random.uniform() < self.epsilon:

            #可能出现actions_value为负数，所以现在将对应exist_pjs 的actions_value取出来取最大的
            # actions_value = self.eval_net.forward(state_cuda).cpu()
            actions_value = self.eval_net.forward(state_cuda)
            actions_value = actions_value[0,0:len(exist_pjsl)]
            action = torch.max(actions_value, 0)[1].cpu().data.numpy()
            #action = action[0][0]

        else:
            index = exist_pjsl
            action = np.random.randint(0, len(exist_pjsl))
            # while exist_pjsb[action] == 0:
            #     action = index[np.random.randint(0, exist_pjs.count(1))]

        return np.array([state]), action

    def store_transition(self, state, action, reward, s_next):
        if s_next.shape[0] == 1:
            transition = np.hstack((state[0], [action, reward], s_next[0]))
        else:
            transition = np.hstack((state[0], [action, reward], s_next))
        #transition = np.hstack((state[0], [action, reward], s_next[0]))
        # self.memory.append(transition)
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1
        # if self.memory_counter > self.memory_capacity:
        #     self.memory.pop(0)
        #     self.memory_counter -= 1

    def learn(self):
        # slen = state.shape[1] - self.histo_len
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(collections.OrderedDict(self.eval_net.state_dict()))
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_capacity, self.batch_size)  # capacity >= bs
        b_memory = np.array(self.memory)[sample_index]
        #b_memory = self.memory
        b_state = torch.LongTensor([b[: self.state_len] for b in b_memory]).cuda()
        b_action = torch.LongTensor([b[self.state_len: self.state_len + 1] for b in b_memory]).cuda()
        b_reward = torch.FloatTensor([b[self.state_len+1: self.state_len + 2] for b in b_memory]).cuda()
        b_state_next = torch.LongTensor([b[-(self.state_len):] for b in b_memory]).cuda()
        # b_state = torch.LongTensor([b[: self.state_len] for b in b_memory])
        # b_action = torch.LongTensor([b[self.state_len: self.state_len + 1] for b in b_memory])
        # b_reward = torch.FloatTensor([b[self.state_len + 1: self.state_len + 2] for b in b_memory])
        # b_state_next = torch.LongTensor([b[-(self.state_len):] for b in b_memory])

        q_eval = self.eval_net(b_state).gather(dim=1, index=b_action)  # gather ?
        q_next = self.target_net(b_state_next).detach()

        q_target = b_reward + self.gamma * q_next[:,q_eval.argmax(dim=1)[0]]

        loss = self.loss_function(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

        # TODO: tensorboard - loss



