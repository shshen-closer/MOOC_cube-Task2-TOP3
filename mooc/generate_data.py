# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:41:54 2020

@author: shshen

数据生成

"""
import numpy as np
import json
from tqdm import tqdm
from sklearn.model_selection import KFold,StratifiedKFold


seq_len = 30

with open('data/student2id', 'r', encoding='utf8') as fo:     #读取 student to id  字典  
    for line in fo:
        student2id = eval(line)
with open('data/problem2id', 'r', encoding='utf8') as fo:   #读取 problem to id   字典
    for line in fo:
        problem2id = eval(line)

p_0 = np.load('data/p_0.npy')   #读取难度值低于0.75的题目

       
test_all = np.load("data/test_data.npy", allow_pickle=True)  #读取线上测试集数据
train_all = np.load("data/train_data.npy", allow_pickle=True) # 读取所有训练数据
#np.random.shuffle(train_all)


#  生成 训练集和验证集
kfold = KFold(n_splits=10, shuffle=False)             #10折交叉
count = 1
for (train_index, valid_index) in kfold.split(train_all):
    print(count)
    train_data = train_all[train_index]
    valid_data = train_all[valid_index]
    train_q = []          #题目
    train_a = []         #答案
    train_length = []    #序列长度
   # fenshu = []      
    for tt in train_data:
        leng = len(tt)
        split = int(leng*0.6)                  # 前60%数据进行模型训练，  后40%数据进行模型验证
        front = tt[:split]
        for ii in tt[split:]:
            if ii['problem_id'] in p_0:       #过滤难度值低于0.75的题目
                continue
        #    fenshu.append(ii['label'])
            a = []
            b = []

            
            for ff in front:
                if ff['problem_id'] not in p_0:
                    a.append(problem2id[ff['problem_id']])
                    b.append(ff['label'])
                    
            a = a[:seq_len - 1]
            b = b[:seq_len - 1]

            while len(a) < seq_len - 1:      #长度不足的序列补齐
                a.append(840)
                b.append(2)
            
            a.append(problem2id[ii['problem_id']])
            b.append(ii['label'])
            
            train_q.append(a)
            train_a.append(b)
            train_length.append(seq_len - 1)

   #print(np.mean(fenshu))
        
            
    np.save("data/data_30/train_q_"+ str(count) + ".npy",np.array(train_q))
    np.save("data/data_30/train_a_"+ str(count) + ".npy",np.array(train_a))
    np.save("data/data_30/train_length_"+ str(count) + ".npy",np.array(train_length))
    
    valid_q = []
    valid_a = []
    valid_length = []
  #  fenshu = []
    for tt in valid_data:
        leng = len(tt)
        split = int(leng*0.6)      # 前60%数据进行模型训练，  后40%数据进行模型验证
        
        for ii in tt[split:]:
            if ii['problem_id'] in p_0:
                continue
           # fenshu.append(ii['label'])
            a = []
            b = []
            
            for ff in tt[:split]:
                if ff['problem_id'] not in p_0:
                    a.append(problem2id[ff['problem_id']])
                    b.append(ff['label'])
      
            a = a[:seq_len - 1]
            b = b[:seq_len - 1]
            while len(a) < seq_len - 1:
                a.append(840)
                b.append(2)
            a.append(problem2id[ii['problem_id']])
            b.append(ii['label'])
            
            valid_length.append(seq_len - 1)
            valid_q.append(a)
            valid_a.append(b)
  #  print(np.mean(fenshu))
    np.save("data/data_30/valid_q_"+ str(count) + ".npy",np.array(valid_q))
    np.save("data/data_30/valid_a_"+ str(count) + ".npy",np.array(valid_a))
    np.save("data/data_30/valid_length_"+ str(count) + ".npy",np.array(valid_length))
    count+=1


#  生成线上测试集数据
test_q = []
test_a = []
test_length = []
test_id = []
for tt in test_all:
    front = [x for x in tt if 'item_id' not in x.keys()]  # 已知答案数据，用于建模
    behind = [x for x in tt if 'item_id' in x.keys()]    #线上测试数据

    for ii in behind:
        a = []
        b = []
        test_id.append(ii['item_id'])
        

        for ff in front[:len(front)]:
            
            a.append(problem2id[ff['problem_id']])
            b.append(ff['label'])
        a = a[:seq_len - 1]
        b = b[:seq_len - 1]
        while len(a) < seq_len - 1:
            a.append(840)
            b.append(2)
        a.append(problem2id[ii['problem_id']])
        b.append(2)
        
        test_q.append(a)
        test_a.append(b)
        test_length.append(seq_len - 1)
np.save("data/data_30/test_q.npy",np.array(test_q))
np.save("data/data_30/test_a.npy",np.array(test_a))
np.save("data/data_30/test_length.npy",np.array(test_length))
np.save("data/data_30/test_id.npy",np.array(test_id))

