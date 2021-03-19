# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:47:29 2020

@author: shshen

数据预处理

"""
from collections import Counter
import numpy as np
import json

train_item = []

#读入训练集
with open('Task2_data_0804/problem_act_train.json', 'r', encoding='utf8') as fi1:
    for line in fi1:
        item = json.loads(line)
        train_item.append(item)

with open('Task2_data_0804/problem_act_train_2.json', 'r', encoding='utf8') as fi2:
    for line in fi2:
        item = json.loads(line)
        train_item.append(item)

#读入测试集
test_item = []
with open('Task2_data_0804/problem_act_test_new.json', 'r', encoding='utf8') as fi2:
    for line in fi2:
        item = json.loads(line)
        test_item.append(item)
    

all_data =  train_item + test_item

all_data = sorted(all_data, key=lambda x:int(x['student_id'][2:]))   #按student_id排列数据


#保存所有答题数据
with open('data/all_data.json', 'w', encoding='utf8') as fo:
    json.dump(all_data, fo, indent=2, ensure_ascii=False)


student2id = {}  #存储学生id
idx = 0
temp = []
train_data = []  #训练数据
test_data = []   #线上测试数据
is_test = False  #标记值，是否含有线上测试记录
for i in range(len(all_data) - 1):
    temp.append(all_data[i])
    if 'item_id' in all_data[i].keys():
        is_test = True
    if all_data[i+1]['student_id'] != all_data[i]['student_id']:   #判定单个学生答题序列是否终止
        if len(temp) >= 5 or is_test:                             #答题序列长度低于5 的数据过滤了
            student2id[all_data[i]['student_id']] = idx
            idx += 1
            if is_test:
                test_data.append(temp)
            else:
                train_data.append(temp)
        temp = []
        is_test = False
temp.append(all_data[-1])
if 'item_id' in all_data[-1].keys():
    is_test = True
if len(temp) >= 5 or is_test:
    student2id[all_data[i]['student_id']] = idx
    idx += 1
    if is_test:
        test_data.append(temp)
    else:
        train_data.append(temp)


np.save("data/test_data.npy",np.array(test_data))
np.save("data/train_data.npy",np.array(train_data))

with open('data/student2id', 'w', encoding='utf8') as fo:   #存储学生id 字典
    fo.write(str(student2id))




#获取题目对应知识点
problem_info = []
problem2id = {}          #存储id 字典
idx = 0
with open('Task2_data_0804/problem_info.json', 'r', encoding='utf8')as fi:
    for line in fi:
        item = json.loads(line)
        problem2id[item['problem_id']] = idx
        idx += 1
        problem_info.append(item)
with open('data/problem2id', 'w', encoding='utf8') as fo:    #存储题目id 字典
    fo.write(str(problem2id))
    
knowponints = []
for x in problem_info:
    knowponints.extend(x['concept'])

knowponints = set(knowponints)  #获得所有知识点
idx = 0
k2id = {}
for i in knowponints:
    k2id[i] = idx
    idx+=1
     
problem2kpoint = {}   # 题目转化知识点  字典
idx = 0
with open('Task2_data_0804/problem_info.json', 'r', encoding='utf8')as fi:
    for line in fi:
        item = json.loads(line)
        sss = []
        for ii in item['concept']:
            sss.append(k2id[ii])
        problem2kpoint[problem2id[item['problem_id']]] = sss

with open('data/problem2kpoint', 'w', encoding='utf8') as fo:   #题目转化知识点  字典 写入文件
    fo.write(str(problem2kpoint))
