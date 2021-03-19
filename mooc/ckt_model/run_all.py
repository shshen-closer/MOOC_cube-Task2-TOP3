import os
for i in range(1, 11):
    os.system("CUDA_VISIBLE_DEVICES=0"+" python train.py "+ str(i)+" 128")
