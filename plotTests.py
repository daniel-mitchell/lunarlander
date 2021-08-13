#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

filename = sys.argv[1]

if len(sys.argv) > 2:
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

plt.plot(data.cumsum())
plt.show()
