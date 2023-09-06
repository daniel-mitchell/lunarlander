#!/usr/bin/env python

import scipy.stats
import numpy as np
import sys, os

dirname = sys.argv[1]
files=os.listdir(dirname)
# data = []
# for f in files:
#     data.append(np.loadtxt(f, delimiter='\x08', usecols=range(1)))
#     #print(f+","+np.mean(data)+","+np.var(data))

for i in range(len(files)):
    for j in range(i+1, len(files)):
        d1 = np.loadtxt(dirname+files[i], delimiter='\x08', usecols=range(1))
        d2 = np.loadtxt(dirname+files[j], delimiter='\x08', usecols=range(1))
        norm1 = scipy.stats.normaltest(d1)
        norm2 = scipy.stats.normaltest(d2)
        if norm1[1] < 0.05 or norm2[1] < 0.05:
            #Not normal, use levene's
            stat,pval = scipy.stats.levene(d1, d2)
        else:
            #Normal, use bartlett's
            stat,pval = scipy.stats.bartlett(d1, d2)
        res = scipy.stats.ttest_ind(d1, d2, equal_var=(pval>0.05))
        print(files[i], files[j], res)

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
