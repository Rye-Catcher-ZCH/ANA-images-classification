# import os
# import csv

# Coarse speckled 0
# Fine spechkled 1
# Nucleolar 2
# Peripheral 3

# 实现数据预处理划分数据集的过程


# import random
# import shutil
# from shutil import copy2
trainfiles = open('all_list.csv', 'r')
lines = trainfiles.readlines()
# num_train = len(trainfiles)
# random.shuffle(index_list)
num = 0
test = open('test.csv', 'w')
train = open('train.csv', 'w')
for i in range(0, 20):

    if i < 20 * 0.5:
        train.write(lines[i])
    else:
        test.write(lines[i])

for i in range(20, 56):
    # fileName = os.path.join('all_list', trainfiles[i])
    if i < 38:
        train.write(lines[i])
    else:
        test.write(lines[i])

for i in range(56, 223):
    # fileName = os.path.join('all_list', trainfiles[i])
    if i < 140:
        train.write(lines[i])
    else:
        test.write(lines[i])

for i in range(223, 260):
    # fileName = os.path.join('all_list', trainfiles[i])
    if i < 241:
        train.write(lines[i])
    else:
        test.write(lines[i])
