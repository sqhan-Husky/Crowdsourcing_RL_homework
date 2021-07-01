import numpy as np

class Environment():
    def __init__(self, category, pj_entry):
        self.category = category  # a list of category corresponding to each project
        self.cumulated_reward = 0
        self.pj_entry = pj_entry  # a list of total entry number per project
        # question: when to update? in step?
        self.current_pj_entry = [0 for i in range(0, len(self.category))]  # a list of record current entry

    def reset(self):
        self.current_pj_entry = [0 for i in range(0, len(self.category))]
        self.cumulated_reward = 0

    def step(self, actions, true_actions):
        cumulated_reward = self.cumulated_reward
        current_reward = 0
        for k in range(len(actions)):
            if actions[k] == true_actions[k]:
                current_reward += 1
            elif self.category[actions[k]] == self.category[true_actions[k]]:
                current_reward += 0.5
            else:
                current_reward += 0

        cumulated_reward += current_reward

        return cumulated_reward
