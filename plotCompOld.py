#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

filename = sys.argv[1]
paramX = int(sys.argv[2])
paramL = int(sys.argv[3])
nameX = sys.argv[4]
nameL = sys.argv[5]

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
data = np.loadtxt(filename, delimiter=',', usecols=[0, paramX, paramL])
data=data[data[:, 1].argsort()]
lines = set(data[:,2])

for line in lines:
    d = data[np.where(data[:,2] == line)]
    plt.plot(d[:,1],d[:,0]/-30,marker='o',label=nameL+"="+str(line), clip_on=False, zorder=10)

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
plt.legend()
plt.xlabel(nameX)
plt.ylabel("Return")
#axes = plt.gca()
#axes.set_ylim([-1000,0])
if sys.argv[-1] == "--save":
    plt.savefig(filename+".png")
else:
    plt.show()
