import os
import sys

filerun = sys.argv[1]

list_file = os.listdir(filerun + '/')

for i in range(len(list_file)):
    print(list_file[i])
    os.system("CUDA_VISIBLE_DEVICES=0 python test.py " + str(i+1) +' ' + list_file[i] + ' '+ filerun)
