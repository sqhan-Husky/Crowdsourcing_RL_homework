from DQN import DQN
from Embedding import Net
from Environment import Environment

import numpy as np
import collections
import pandas as pd
from tqdm import tqdm

def split_data(data,per):

    train_data = data[0:int(per*len(data))]
    test_data = data[int(per*len(data)):]

    return train_data,test_data

if __name__ == "__main__":
    EPOCH = 200
    BATCH_SIZE = 32
    LR = 0.01
    EPSILON = 0.9
    GAMMA = 0.9
    TARGET_REPLACE_ITER = 100
    MEMORY_CAPACITY = 2000
    N_ACTIONS = 2600
    SEQ_LEN = 20
    HISTO_LEN = 7
    # TODO: CUDA

    data = pd.read_pickle('data/data.pkl')
    category = data['category'].values.tolist()
    pj_entry = data['entry_num'].values.tolist()


    all_data = pd.read_pickle('data/alldata.pkl')
    train,test = split_data(all_data, 0.8)

    env = Environment(category, pj_entry)
    dqn = DQN(SEQ_LEN,HISTO_LEN,category, N_ACTIONS, BATCH_SIZE, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY)
    #dqn.cuda()

    for i in range(0, EPOCH):
        env.reset()
        reward_sum = 0

        for j in tqdm(range(0, len(train))):
        # for index, row in tqdm(train.iterrows()):
            record = train[j:j+1]
            state = record['sequence'].values[0]
            env.update(record['exist_pjs_list'].values[0]) #exist_pjs_list列表的形式
            state, action = dqn.choose_action(state, record['exist_pjs_list'].values[0])
            reward, state_next = env.step(state, action, record['true_action'].values[0])

            dqn.store_transition(state, action, reward, state_next)
            reward_sum += reward

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            # TODO: print reward
            if j % 1000 == 0:
                print('EPOCH: %d   timetamps: %d   reward:%f,'%(i, j, reward_sum/(j+1)))

    # TODO: Test - metric
    test_reward = 0
    count = 0
    for index, row in test.iterrows():
        state = row['sequence']
        env.update(row['exist_pjs_b'])
        state, action = dqn.choose_action(state, row['exist_pjs_b'], row['exist_pjs_list'])
        reward, state_next = env.step(state, action, row['true_action'])

        if action+1 == row['true_action']:
            count +=1
        test_reward += reward
    print('Test reward: %d   CR: %f '% (test_reward, count/len(test)))