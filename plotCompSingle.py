#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys, os
from matplotlib import rcParams
import itertools, math

dirname = sys.argv[1]
paramX = int(sys.argv[2])
nameX = sys.argv[3]

clist = rcParams['axes.color_cycle']
cgen = itertools.cycle(clist)

files=os.listdir(dirname)
if ".DS_Store" in files:
    files.remove(".DS_Store")

data=np.zeros([len(files), 3])

for i in range(len(files)):
    dataRaw = np.loadtxt(dirname+files[i], delimiter='\x08', usecols=range(1))
    params = files[i].split("_")
    data[i] = np.array([np.mean(dataRaw)+1000000, np.std(dataRaw)/math.sqrt(np.size(dataRaw)), params[paramX]])
    # print(files[i])
    # print(data[i])

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
#data = np.loadtxt(filename, delimiter=',', usecols=[0, paramX, paramL])
data=data[data[:, 2].argsort()]
#lines = set(data[:,3])

#for line in lines:
#    d = data[np.where(data[:,3] == line)]
    # plt.plot(d[:,2],d[:,0],marker='o',label=nameL+"="+str(line), clip_on=False, zorder=10)
    # plt.fill_between(d[:,2],d[:,0]-d[:,1],d[:,0]+d[:,1], alpha=0.5, facecolor=cgen.next())
plt.errorbar(data[:,2],data[:,0],yerr=data[:,1],marker='o', clip_on=False, zorder=10,elinewidth=2,capsize=5,capthick=2)
    # print(d[:,2])
    # print(d[:,1])
    # print(-d[:,0])

# l=1000
# data2 = [sum(data[max(i-l, 0):min(i+l, len(data))])/(min(i+l, len(data)) - max(i-l, 0)) for i in range(len(data))]

# plt.plot(data.cumsum())
# plt.plot(data, linestyle='None', marker='o', markersize = 1)
# plt.plot(data2, "r-")
# x=range(len(data))
# z = np.polyfit(x, data, 1)
# p = np.poly1d(z)
# plt.plot(x, p(x), "r--")
plt.xscale("log")
#plt.legend(numpoints=1)
plt.xlabel(nameX)
plt.ylabel("Return")
#axes = plt.gca()
#axes.set_ylim([-1000,0])
if sys.argv[-1] == "--save":
    plt.savefig(dirname[:-1]+".png")
else:
    plt.show()
