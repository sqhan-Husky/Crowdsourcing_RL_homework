from DQN import DQN
from Embedding import Net
from Environment import Environment

import numpy as np
import collections
import pandas as pd
from tqdm import tqdm
from tensorboardX import SummaryWriter
writer = SummaryWriter('logs/0704_1')

def split_data(data,per):

    train_data = data[0:int(per*len(data))]
    test_data = data[int(per*len(data)):]

    return train_data,test_data

if __name__ == "__main__":
    EPOCH = 2
    BATCH_SIZE = 32
    LR = 0.01
    EPSILON = 0.9
    GAMMA = 0.9
    TARGET_REPLACE_ITER = 100
    MEMORY_CAPACITY = 100
    N_ACTIONS = 2600
    SEQ_LEN = 20
    HISTO_LEN = 36

    data = pd.read_pickle('data/data.pkl')
    category = data['category'].values.tolist()
    pj_entry = data['entry_num'].values.tolist()
    sub_category = data['sub_category'].values.tolist()

    all_data = pd.read_pickle('data/alldata_task.pkl')
    train,test = split_data(all_data, 0.8)

    env = Environment(category, pj_entry,sub_category)
    dqn = DQN(SEQ_LEN,HISTO_LEN, N_ACTIONS, BATCH_SIZE, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY)
    # dqn.cuda()
    cnt = 0
    cnt_learn = 0
    for i in range(0, EPOCH):
        env.reset()
        reward_sum = 0


        for j in tqdm(range(0, len(train))):
        # for index, row in tqdm(train.iterrows()):
            record = train[j:j+1]
            state = record['sequence'].values[0]
            if np.sum(np.array(state[0:SEQ_LEN]) > 0) <= 5:
                continue
            env.update(record['exist_pjs_list'].values[0]) #exist_pjs_list列表的形式
            state, action = dqn.choose_action(state, record['exist_pjs_list'].values[0])
            reward, state_next = env.step(state, action, record['true_action'].values[0],record['exist_pjs_list'].values[0],record['entry_num'].values[0])

            dqn.store_transition(state, action, reward, state_next)
            reward_sum += reward
            cnt += 1
            writer.add_scalar('Train/Loss', reward_sum/cnt, cnt)
            if dqn.memory_counter > MEMORY_CAPACITY:
                loss = dqn.learn()
                writer.add_scalar('Train/Loss', loss, cnt_learn)

                cnt_learn += 1

            # TODO: print reward
            if j % 1000 == 0:
               print('EPOCH: %d   timetamps: %d  reward:%f,' % (i, j,  reward_sum/cnt))
        print('EPOCH: %d   timetamps: %d  reward:%f,' % (i, j, reward_sum / cnt))


    # TODO: Test - metric
    test_reward = 0
    count = 0
    for j in tqdm(range(0, len(test))):
        record = test[j:j + 1]
        state = record['sequence'].values[0]
        env.update(record['exist_pjs_list'].values[0])  # exist_pjs_list列表的形式
        state, action = dqn.choose_action(state, record['exist_pjs_list'].values[0])

        if record['exist_pjs_list'].values[0][action] == record['true_action'].values[0]:
            count +=1
            test_reward += reward
    print('Test reward: %d   CR: %f '% (test_reward/len(test), count/len(test)))