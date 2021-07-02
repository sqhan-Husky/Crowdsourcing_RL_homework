import numpy as np

class Environment():
    def __init__(self, category, pj_entry):
        self.category = category  # a list of category corresponding to each project
        self.cumulated_reward = 0
        self.pj_entry = pj_entry  # a list of total entry number per project
        self.current_pj_entry = [-1 for i in range(0, len(self.category))]  # a list of record current entry

    def reset(self):
        self.current_pj_entry = [-1 for i in range(0, len(self.category))]
        self.cumulated_reward = 0

    def update(self, exist_pjs):
        for pj_id in exist_pjs:
            if self.current_pj_entry[pj_id] == -1:
                self.current_pj_entry[pj_id] = 0

    def step(self, actions, true_actions):
        cumulated_reward = self.cumulated_reward
        self.current_pj_entry[true_actions] += 1
        current_reward = 0
        if actions == true_actions:
            current_reward += 1
        elif self.category[actions] == self.category[true_actions]:
            current_reward += 0.5
        else:
            current_reward += 0

        cumulated_reward += current_reward

        return cumulated_reward
