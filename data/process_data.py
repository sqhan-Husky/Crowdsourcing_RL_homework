import pandas as pd
import csv
import datetime
from collections import Counter
import numpy as np

project_path = 'project_info.csv'
entry_path = 'entry_info.csv'
state_path = 'state_info.csv'

# data = pd.read_csv(state_path)
project_info = pd.read_csv(project_path)
entry_info = pd.read_csv(entry_path)
worker_list = {'worker_id':[],'worker_quality':[]}



csvfile = open("worker_quality.csv", "r")
csvreader = csv.reader(csvfile)
for line in csvreader:
    worker_list['worker_id'].append(int(line[0]))
    if float(line[1]) > 0.0:
        worker_list['worker_quality'].append(float(line[1]) / 100.0)
csvfile.close()

project_list = pd.read_csv('project_list.csv')

print("read data is ok")


def datestr(timestr):
    """
    将时间转换成时间戳
    :param timestr: y-m-d : h-m-s 格式时间
    :return: 时间戳
    """
    ind = timestr.index('+')
    date = datetime.datetime.strptime(str(timestr[:19]), '%Y-%m-%d %H:%M:%S')

    return int(date.timestamp())


#将project 按照时间排序 并重新设定id
def order_project(project_infos):
    """
    :param project_infos: 原始文件
    :return: project_info
    """
    project_id = list(project_infos)[1:]

    project_info = {'project_id': [],
                    'start_date': [],
                    'deadline': [],
                    "entry_count": [],
                    'industry': [],
                    'sub_category': [],
                    'category': []}
    project_order_info = {}
    for i in range(len(project_id)):
        project_info['project_id'].append(project_id[i])
        project_info['start_date'].append(datestr(project_infos[str(project_id[i])][3]))
        project_info['deadline'].append(datestr(project_infos[str(project_id[i])][4]))
        project_info['entry_count'].append(project_infos[str(project_id[i])][2])
        project_info['industry'].append(project_infos[str(project_id[i])][5])
        project_info['sub_category'].append(project_infos[str(project_id[i])][0])
        project_info['category'].append(project_infos[str(project_id[i])][1])

    project_info = pd.DataFrame(project_info)
    project_order_info = project_info.sort_values("start_date", inplace=False)
    project_order_info.reset_index(drop=True, inplace=True)

    print('order project is ok')

    return project_order_info




def order_entry(entry_info):
    """
    entry 按照创建时间排序
    :param entry_info: entry原始数据
    :return: 处理后的数据
    """
    entry_info = entry_info.drop(['Unnamed: 0'], axis=1)
    times = []
    for i in range(len(entry_info)):
        times.append(datestr(entry_info['entry_created_at'][i]))

    entry_info['entry_created_at'] = times
    entry_info = entry_info.sort_values("entry_created_at", inplace=False)
    entry_info.reset_index(drop=True, inplace=True)

    return entry_info


def get_prostate(worker_id, worker_prodict, project_id):
    """
    更新worker做过的project的列表
    :param worker_id:
    :param worker_prodict:
    :param project_id:
    :return:
    """
    pro_list = worker_prodict.get(worker_id, [])
    history_prostate = pro_list.copy()
    pro_list.append(project_id)
    next_prostate = pro_list.copy()
    worker_prodict[worker_id] = pro_list
    return worker_prodict,history_prostate,next_prostate

def get_category(category_num,worker_catdict, cat_index,worker_id):
    """
    更新catestate的状态
    :param category_num:  category的个数
    :param worker_catdict:  每个worker存储category的个数
    :param cat_index: category的个数序号
    :param worker_id:
    :return: state dict
    """
    zerolist = [0] * category_num
    cate_list = worker_catdict.get(worker_id, zerolist)
    history_catstate = cate_list.copy()
    cate_list[cat_index] += 1
    next_catstate = cate_list.copy()
    worker_catdict[worker_id] = cate_list
    return worker_catdict, history_catstate, next_catstate

def worker_entry(worker_list, project_list, entry_info):
    """
    按照时间戳生成state
    :param worker_list:
    :param project_list:
    :param entry_info:
    :return:
    """
    worker_prodict = {}
    worker_catdict = {}

    category_u = list(set(list(project_list['category'])))
    category_u.sort()
    entry_info = order_entry(entry_info)

    state_info = {'woker_id': [],
                  'history_taskstate': [],
                  'history_catstate': [],
                  'entry_created_at': [],
                  'project_id': [],
                  'entry_number': [],
                  'category': [],
                  # 'subcategory':[],
                  'next_taskstate': [],
                  'next_catstate': [],
                  }

    for i in range(len(entry_info)):

        worker_in = worker_list['worker_id'].index(entry_info['worker'][i])
        state_info['woker_id'].append(worker_in)
        state_info['entry_created_at'].append(entry_info['entry_created_at'][i])
        pro_in = list(project_list['project_id']).index(str(entry_info['project_id'][i]))
        state_info['project_id'].append(pro_in)

        state_info['entry_number'].append(entry_info['entry_number'][i])
        state_info['category'].append(project_list['category'][pro_in])
        worker_prodict, history_prostate, next_prostate = get_prostate(worker_in, worker_prodict, pro_in)
        state_info['history_taskstate'].append(history_prostate)
        state_info['next_taskstate'].append(next_prostate)
        worker_catdict, history_catstate, next_catstate = get_category(len(category_u), worker_catdict, category_u.index(project_list['category'][pro_in]), worker_in)
        state_info['history_catstate'].append(history_catstate)
        state_info['next_catstate'].append(next_catstate)

    return state_info

project_order_info = order_project(project_info)
state_info = worker_entry(worker_list, project_order_info, entry_info)
state_info = pd.DataFrame(state_info)
state_info.to_csv('state_info1.csv', index=False)
# state_info = pd.read_csv('state_info1.csv')
print('loading state file is ok')
# padding什么？？？从那头开始
def padding(list,seq_len):

    if len(list) < seq_len:
        # zerolist = [0] * (seq_len - len(list))

        return list
    else:
        return list[len(list)-seq_len-1:len(list)-1]

def state_padding(state_info,seq_len):
    next_taskstate = []
    history_taskstate = []
    history_catstate = []
    next_catstate = []
    for i in range(len(state_info['history_taskstate'])):

        history_taskstate.append(padding(state_info['history_taskstate'][i],seq_len))
        # history_catstate.append(padding(state_info['history_catstate'][i], seq_len))
        next_taskstate.append(padding(state_info['next_taskstate'][i], seq_len))
        # next_catstate.append(padding(state_info['next_catstate'][i], seq_len))

    state_info['next_taskstate'] = next_taskstate
    state_info['history_taskstate'] = history_taskstate


    return state_info

state_info = state_padding(state_info,20)
state_info = pd.DataFrame(state_info)
state_info.to_csv('state_info_padding.csv',index=False)

print('ok')

def pro_list(pro,index,entry_info,project_list):


    st = project_list[pro]['start_date']
    count = 0
    for j in range(index,-1,-1):
        if entry_info['entry_created_at'] > st:
            break
        else:
            if list == project_list['project_id'].index(entry_info['project_id']):
                count += 1

    if count < project_list['entry_count']:
        return True
    else:
        return False



def worker_projectnum(worker_list, entry_info):

    worker_id = list(worker_list['worker_id'])

    worker_proinfo = {'worker_id': [],'pro_count': []}
    worker = list(entry_info['worker'])
    project = list(entry_info['project_id'])
    for i in range(len(worker_id)):
        index = []
        index.append([j for j, x in enumerate(worker) if x == worker_id[i]])
        w_pro = []
        for k in range(len(index[0])):
            w_pro.append(project[index[0][k]])

        w_pro_count = Counter(w_pro)
        worker_proinfo['worker_id'].append(worker_id[i])
        worker_proinfo['pro_count'].append(w_pro_count)

    print("ok")
    return worker_proinfo



# worker_proinfo = worker_projectnum(worker_list, entry_info, project_list)
print('ok')