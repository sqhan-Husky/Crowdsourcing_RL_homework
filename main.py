from DQN import DQN
from Embedding import Net
from Environment import Environment
import numpy as np
import collections
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    EPOCH = 200
    BATCH_SIZE = 32
    LR = 0.01
    EPSILON = 0.9
    GAMMA = 0.9
    TARGET_REPLACE_ITER = 100
    MEMORY_CAPACITY = 2000
    N_ACTIONS = 3000
    SEQ_LEN = 20

    # TODO: CUDA

    data = pd.read_csv('')
    category = data['category'].values.tolist()
    pj_entry = data['entry_num'].values.tolist()

    train = pd.read_csv('')

    env = Environment(category, pj_entry)
    dqn = DQN(SEQ_LEN, N_ACTIONS, BATCH_SIZE, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY)

    for i in tqdm(range(0, EPOCH)):
        env.reset()
        reward_sum = 0

        for index, row in train.iterrows():
            state = row['sequence']
            env.update(row['exist_pjs'])
            action, state_next = dqn.choose_action(state, row['exist_pjs'])
            reward = env.step(action, row['true_action'])

            dqn.store_transition(state, action, reward, state_next)
            reward_sum += reward

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            # TODO: print reward

    # TODO: Test - metric