import numpy as np
from functools import reduce
import operator

class Environment():
    def __init__(self, category, pj_entry,sub_category,):
        self.category = category  # a list of category corresponding to each project
        self.cumulated_reward = 0
        self.sub_category = sub_category
        self.pj_entry = pj_entry  # a list of total entry number per project
        self.current_pj_entry = [-1 for i in range(0, len(self.category))]  # a list of record current entry
        self.category_index = list(set(list(category)))
        self.seq_len = 20
        self.task_len = 83*3
        self.subcategory_index = list(set(list(sub_category)))#更新histo状态
        self.histo_len = len(self.category_index)+len(self.subcategory_index)

    def reset(self):
        self.current_pj_entry = [-1 for i in range(0, len(self.category))]
        self.cumulated_reward = 0

    def update(self, exist_pjs):
        for pj_id in exist_pjs:
            if self.current_pj_entry[pj_id] == -1:
                self.current_pj_entry[pj_id] = 0

    def step(self, state, actions, true_actions,exsit_pro,entry_num):
        cumulated_reward = self.cumulated_reward
        self.current_pj_entry[true_actions-1] += 1
        current_reward = 0
        #project下标从1开始
        if exsit_pro[actions] == true_actions:
            current_reward += 1
            state_next = self.update_state(state, true_actions,exsit_pro,entry_num)

        elif self.category[exsit_pro[actions]-1] == self.category[true_actions-1]:
            current_reward += 0.5
            state_next = state

        else:
            current_reward += 0
            state_next = state

        cumulated_reward += current_reward
        action = np.where(exsit_pro == true_actions)
        return cumulated_reward, state_next

    def task_list_padding(self,t_list, task_len):
        t = [0]
        if len(t_list) < task_len:
            t_list += t * (task_len - len(t_list))
        else:
            t_list = t_list[-task_len:]
        return t_list

    def update_state(self, state, action,exsit_pro,entry_num):
        # seq_update
        w_id = state[0][0:1]
        current_seq_pro = state[0][1:self.seq_len+1]
        seq_pro_next = np.append(current_seq_pro, action)[1:]
        current_seq_cat = state[0][self.seq_len + 1:self.seq_len*2 + 1]
        seq_cat_next = np.append(current_seq_cat, self.category[action-1] + 1)[1:]
        current_seq_sub = state[0][self.seq_len*2 + 1:self.seq_len * 3 + 1]
        seq_sub_next = np.append(current_seq_sub, self.sub_category[action-1] + 1)[1:]

        if entry_num == self.pj_entry[action-1]:
            t_l = []
            for j in range(len(exsit_pro)):
                if exsit_pro[j] != action:
                    t = []
                    t.append(exsit_pro[j])
                    t.append(self.category_index.index(self.category[exsit_pro[j] - 1]) + 1)
                    t.append(self.subcategory_index.index(self.sub_category[exsit_pro[j] - 1]) + 1)
                    t_l.append(t)
            t_l = reduce(operator.add, t_l)
            task_cat_next = self.task_list_padding(t_l, self.task_len)

        else:
            task_cat_next = state[0][self.seq_len * 3 + 1:]


        state_next = np.hstack((w_id, seq_pro_next,seq_cat_next,seq_sub_next,task_cat_next))
        return state_next