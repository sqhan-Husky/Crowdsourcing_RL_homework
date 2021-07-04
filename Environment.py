import numpy as np

class Environment():
    def __init__(self, category, pj_entry,sub_category):
        self.category = category  # a list of category corresponding to each project
        self.cumulated_reward = 0
        self.sub_category = sub_category
        self.pj_entry = pj_entry  # a list of total entry number per project
        self.current_pj_entry = [-1 for i in range(0, len(self.category))]  # a list of record current entry
        self.category_index = list(set(list(category)))
        self.subcategory_index = list(set(list(sub_category)))#更新histo状态
        self.histo_len = len(self.category_index)+len(self.subcategory_index)

    def reset(self):
        self.current_pj_entry = [-1 for i in range(0, len(self.category))]
        self.cumulated_reward = 0

    def update(self, exist_pjs):
        for pj_id in exist_pjs:
            if self.current_pj_entry[pj_id] == -1:
                self.current_pj_entry[pj_id] = 0

    def step(self, state, actions, true_actions):
        cumulated_reward = self.cumulated_reward
        self.current_pj_entry[true_actions] += 1
        current_reward = 0
        #project下标从1开始
        if actions+1 == true_actions:
            current_reward += 1
            state_next = self.update_state(state, actions)

        elif self.category[actions] == self.category[true_actions-1]:
            current_reward += 0.5
            state_next = state

        else:
            current_reward += 0
            state_next = state

        cumulated_reward += current_reward

        return cumulated_reward, state_next

    def update_state(self, state, action):
        # seq_update
        slen = state.shape[1] - self.histo_len
        clen = len(set(self.category))

        current_seq = state[0][0:slen]
        seq_next = np.append(current_seq, action)[1:]

        # todo: histo_update
        current_histo = state[0][slen:slen+clen]
        current_histo[self.category_index.index(self.category[action])] += 1
        histo_next = current_histo
        current_histo_sub = state[0][slen + clen:]
        current_histo_sub[self.subcategory_index.index(self.sub_category[action])] += 1
        histo_next_sub = current_histo_sub

        state_next = np.hstack((seq_next, histo_next,histo_next_sub))
        return state_next