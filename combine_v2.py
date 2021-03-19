# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:30:38 2020

@author: shshen

多个预测结果简单融合

"""

import numpy as np
import json
import os


sample_submission = []
with open('Task2_data_0804/problem_act_test_new.json', 'r', encoding='utf8')as fi:
    for line in fi:
        item = json.loads(line)
        lll = {"item_id": item['item_id'], "label": 1}
        sample_submission.append(lll)


file_path = 'ckt_model/result/'
list_file = os.listdir(file_path)
pred_10flod = []
for iii in list_file:
    
    with open(file_path + iii, 'r', encoding='utf8')as fi:
        for line in fi:
            pred = eval(line)
        pred_10flod.append(pred)

sub = []
answer = []
for temp in sample_submission:
    preds = []
    count = 0
    for pp in pred_10flod:
        preds.append(pp[temp['item_id']])
    avg_pred = np.mean(preds)
    if avg_pred > 0.5:
        temp['label'] =1
    else:
        temp['label'] =0
    sub.append(temp)
    answer.append(temp['label'])
print(np.mean(answer))
with open('submission.json', 'w', encoding='utf8') as fo:
    for ss in sub:
        json.dump(ss, fo, ensure_ascii=False)
        fo.write('\n')