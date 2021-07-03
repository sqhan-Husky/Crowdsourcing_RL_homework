import pandas as pd
import numpy as np
state_info = pd.read_pickle('state_info_padding.pkl')
pro_info = pd.read_csv('pro_info.csv')
exit_pro = pd.read_csv('projects_to_do.csv')


data = {'category': [], 'entry_num': []}
input_data = {'sequence': [], 'exist_pjs_list': [], 'true_action': []}

data['entry_num'] = pro_info['entry_count']
data['category'] = pro_info['category']

data = pd.DataFrame(data)
data.to_pickle('data.pkl')
print('data is ok')




def delete_null(data):
    input_data = {'sequence': [], 'exist_pjs_b': [], 'true_action': [],'exist_pjs_list':[]}
    sequence = []
    exist_pjs_b = []
    true_action = []
    exist_pjs_list = []
    # ToDo update len(data['exist_pjs'])
    l = len(data['exist_pjs'])
    for i in range(l):

        if len(data['sequence'][i]) == 7:
            continue
        else:
            sequence.append(data['sequence'][i])
            exist_pjs_b.append(data['exist_pjs_b'][i])
            true_action.append(data['true_action'][i])
            exist_pjs_list.append(data['exist_pjs_list'][i])

    input_data['sequence'] = sequence
    input_data['exist_pjs_b'] = exist_pjs_b
    input_data['true_action'] = true_action
    input_data['exist_pjs_list'] = exist_pjs_list

    return input_data


def merge_seqhistory(data):

    sequence = []
    history_taskstate = data['history_taskstate']
    history_catstate = data['history_catstate']
    # ToDo update len(data['exist_pjs'])
    for i in range(len(data['history_taskstate'])):
        if sum(history_catstate[i]) != 0:
            his_norl = np.array(history_catstate[i])
            s = history_taskstate[i] + his_norl.tolist()
        else:
            his_norl = history_catstate[i]
            s = history_taskstate[i] + his_norl
        sequence.append(s)
    return sequence

def exitpro_onehot(exit_pro,num):
    # exitpro_onehot = []
    exitpro = []
    for i in range(len(exit_pro)):
        project_to_do = exit_pro[i][1:-1].split(',')
        project_to_do = [int(i) for i in project_to_do]
        l = [0] * num

        for j in range(len(project_to_do)):
            l[project_to_do[j]] = 1
        # exitpro_onehot.append(l)
        exitpro.append(project_to_do)
    return exitpro

#
# project_to_do = [int(i) for i in project_to_do]

sequence = merge_seqhistory(state_info)

input_data['sequence'] = sequence
exitpro = exitpro_onehot(exit_pro['project_to_do'], 2600)
# input_data['exist_pjs_b'] = exitpro_onehot
input_data['exist_pjs_list'] = exitpro
input_data['true_action'] = state_info['project_id']



# input_data = pd.DataFrame(input_data)
# data = delete_null(input_data)
input_data = pd.DataFrame(input_data)
input_data.to_pickle('alldata.pkl')