import torch
import torch.nn as nn
import numpy as np
import collections
from Embedding import Net


class DQN:
    def __init__(self, n_states, n_actions, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity
        # 利用Net创建两个Q网络: 评估网络和目标网络(延后更新,计算loss用)
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        self.learn_step_counter = 0                                           # for target updating
        self.memory_counter = 0                                               # for storing memory
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))           # 初始化记忆库，一行代表一个transition(状态转移过程)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)      # unsqueeze作用：在dim=0增加维数为1的维度
        if np.random.uniform() < self.epsilon:                    # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(state)          # 类型:tensor
            action = torch.max(actions_value, 1)[1].data.numpy()  # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]                                    # int
        else:
            action = np.random.randint(0, self.n_actions)         # 随机选择动作,这里action随机等于0或1 (N_ACTIONS = 2)
        return action                                             # 返回选择的动作 (0或1),int

    def store_transition(self, state, action, reward, s_next):                      # 定义记忆存储函数 (输入为一个transition)
        transition = np.hstack((state, [action, reward], s_next))                   # 在水平方向上拼接数组,ndarry
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 目标网络参数更新
        if self.learn_step_counter % self.target_replace_iter == 0:                      # 第一步触发，随后每xxx步触发(目标网络更新频率)
            self.target_net.load_state_dict(collections.OrderedDict(self.eval_net.state_dict()))   # 将评估网络的参数赋给目标网络(复制)
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据，之后用于训练
        # 在[0, memory_capacity)内随机抽取batch_size个数作为索引，可能会重复
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]                         # 将batch_size个索引对应的transition，存入b_memory
        # 根据之前的拼接顺序再把transition拆开还原为四个tensor
        b_state = torch.FloatTensor(b_memory[:, :self.n_states])        # b_s维度:batch_size*n_states
        # 转为LongTensor(64bit)是为了方便后面torch.gather的使用，b_a维度:batch_size*1
        b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2])       # b_r维度:batch_size*1
        b_state_next = torch.FloatTensor(b_memory[:, -self.n_states:])               # b_s_维度:batch_size*N_states

        # 获取b_memory中transition的评估值和目标值，并进行评估网络参数更新
        # eval_net输出action_values(二元组tensor，因为action有两个0和1)，后面gather代表对每行对应索引(b_action)的Q值提取进行聚合
        q_eval = self.eval_net(b_state).gather(dim=1, index=b_action)                           # 该是式是评估(估计)Q值
        q_next = self.target_net(b_state_next).detach()                                          # detach():不反向传播
        # 该式为目标Q值(即真实Q值，利用了Q-learning更新公式)；q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)
        q_target = b_reward + self.gamma * q_next.max(dim=1)[0].view(self.batch_size, 1)

        loss = self.loss_func(q_eval, q_target)                 # 输入评估Q值和目标Q值各batch_size个,利用两者之差计算loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()