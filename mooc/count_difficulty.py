# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:47:29 2020

@author: shshen

统计题目难度值（题目平均得分）， 平均分越高，难度越低

"""
from collections import Counter
import numpy as np
import json
import random


train_item = []
with open('Task2_data_0804/problem_act_train.json', 'r', encoding='utf8') as fi1:
    for line in fi1:
        item = json.loads(line)
        train_item.append(item)

with open('Task2_data_0804/problem_act_train_2.json', 'r', encoding='utf8') as fi2:
    for line in fi2:
        item = json.loads(line)
        train_item.append(item)
  
test_item = {}
with open('Task2_data_0804/problem_act_test_new.json', 'r', encoding='utf8') as fi2:
    for line in fi2:
        item = json.loads(line)
        test_item[item['item_id']] = item['problem_id']
    

problem = set([x['problem_id'] for x in train_item])  #获取训练集里题目id

p_avg = {}  # 平均得分作为难度值
all_ss = []
for pp in problem:
    s = [x['label'] for x in train_item if x['problem_id'] == pp]
    all_ss.extend(s)
    p_avg[pp] = np.mean(s)

p_0 = []  #筛选难度值低于0,75的题目，即难度较大的题， 训练时去掉这些题目
for key in p_avg.keys():
    if p_avg[key] < 0.75:
        p_0.append(key)

np.save('data/p_0.npy', p_0)