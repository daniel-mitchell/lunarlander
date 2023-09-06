#!/usr/bin/env python

import numpy as np
import sys, os

dirname = sys.argv[1]
files=os.listdir(dirname)
# data = []
# for f in files:
#     data.append(np.loadtxt(f, delimiter='\x08', usecols=range(1)))
#     #print(f+","+np.mean(data)+","+np.var(data))

data = np.zeros(len(files))

for i in range(len(files)):
    d = np.loadtxt(dirname+files[i], delimiter='\x08', usecols=range(1))
    data[i] = d

print(np.mean(data))
print(np.std(data))
print(np.min(data),np.max(data))
# if len(sys.argv) > 3:
#     data = None
#     for i in range(int(sys.argv[2]), int(sys.argv[3])+1):
#         dataTemp = np.loadtxt(filename+str(i), delimiter='\x08', usecols=range(1))
#         if data is None:
#             data = dataTemp
#         else:
#             data += dataTemp
#     data /= int(sys.argv[3])+1
# else:
#     data = np.loadtxt(filename, delimiter='\x08', usecols=range(1))
