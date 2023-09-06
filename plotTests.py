#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

filename = sys.argv[1]

if len(sys.argv) > 3:
    data = None
    for i in range(int(sys.argv[2]), int(sys.argv[3])+1):
        dataTemp = np.loadtxt(filename+str(i))
        if data is None:
            data = dataTemp
        else:
            data += dataTemp
    data /= int(sys.argv[3])+1
else:
    data = np.loadtxt(filename)

# l=1000
# data2 = [sum(data[max(i-l, 0):min(i+l, len(data))])/(min(i+l, len(data)) - max(i-l, 0)) for i in range(len(data))]

# plt.plot(data.cumsum())
plt.plot(data, linestyle='None', marker='o', markersize = 1)
# plt.plot(data2, "r-")
# x=range(len(data))
# z = np.polyfit(x, data, 1)
# p = np.poly1d(z)
# plt.plot(x, p(x), "r--")
axes = plt.gca()
axes.set_ylim([-1000,0])
if sys.argv[-1] == "--save":
    plt.savefig(filename+".png")
else:
    plt.show()
