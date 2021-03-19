6# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:30:38 2020

@author: shshen

单个预测结果生成

"""

import numpy as np
import json
sample_submission = []
with open('Task2_data_0804/problem_act_test_new.json', 'r', encoding='utf8')as fi:
    for line in fi:
        item = json.loads(line)
        lll = {"item_id": item['item_id'], "label": 1}
        sample_submission.append(lll)

with open('pred_labels_runs1', 'r', encoding='utf8')as fi:
    for line in fi:
        pred = eval(line)

sub = []
answer = []
for temp in sample_submission:
    temp['label'] = pred[temp['item_id']]
    sub.append(temp)
    answer.append(pred[temp['item_id']])
print(np.mean(answer))
with open('submission.json', 'w', encoding='utf8') as fo:
    for ss in sub:
        json.dump(ss, fo, ensure_ascii=False)
        fo.write('\n')