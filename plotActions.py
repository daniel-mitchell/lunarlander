#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

# dataMu = []
# dataSigma = []
dataPos = []
dataVel = []
dataAng = []
dataAngVel = []
dataAct = []
splits = []
# dataMuGrad = []
# dataSigmaGrad = []
# dataTdErr = []
index = 0
if (len(sys.argv) == 1) or (len(sys.argv) == 2 and sys.argv[-1] == "--save"):
    for line in sys.stdin:
        dataRaw = line.split(" ")
        if len(dataRaw) > 4:
            index += 1
            # dataMu.append(float(dataRaw[0]))
            # dataSigma.append(float(dataRaw[1]))
            dataPos.append(float(dataRaw[0]))
            dataVel.append(float(dataRaw[1]))
            dataAng.append(float(dataRaw[2]))
            dataAngVel.append(float(dataRaw[3]))
            dataAct.append(float(dataRaw[4]))
            # dataMuGrad.append(float(dataRaw[5]))
            # dataSigmaGrad.append(float(dataRaw[6]))
            # dataTdErr.append(float(dataRaw[7]))
        else:
            splits.append(index)
else:
    with open(sys.argv[1]) as f:
        lines = f.readlines()
        for line in lines:
            dataRaw = line.split(" ")
            if len(dataRaw) > 4:
                index += 1
                # dataMu.append(float(dataRaw[0]))
                # dataSigma.append(float(dataRaw[1]))
                dataPos.append(float(dataRaw[0]))
                dataVel.append(float(dataRaw[1]))
                dataAng.append(float(dataRaw[2]))
                dataAngVel.append(float(dataRaw[3]))
                dataAct.append(float(dataRaw[4]))
                # dataMuGrad.append(float(dataRaw[5]))
                # dataSigmaGrad.append(float(dataRaw[6]))
                # dataTdErr.append(float(dataRaw[7]))
            else:
                splits.append(index)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(18, 10))
# fig, ax = plt.subplots(1, 5, figsize=(18, 10))
ax1.plot(dataPos, range(len(dataPos)), linestyle='None', marker='o', markersize = 1)
ax2.plot(dataVel, range(len(dataVel)), linestyle='None', marker='o', markersize = 1)
ax3.plot(dataAng, range(len(dataAng)), linestyle='None', marker='o', markersize = 1)
ax4.plot(dataAngVel, range(len(dataAngVel)), linestyle='None', marker='o', markersize = 1)
# ax2.errorbar(dataMu, range(len(dataMu)), xerr=dataSigma, linestyle='None', marker='o', markersize = 1)
ax5.plot(dataAct, range(len(dataAct)), linestyle='None', marker='x', markersize = 2, color='y')
# ax3.plot(dataMuGrad, range(len(dataMuGrad)), linestyle='None', marker='o', markersize = 1)
# ax4.plot(dataSigmaGrad, range(len(dataMuGrad)), linestyle='None', marker='o', markersize = 1)
# ax5.plot(dataTdErr[:-1], range(len(dataTdErr)-1), linestyle='None', marker='o', markersize = 1)
for s in splits:
    ax1.axhline(y=s, color='r', linestyle='-')
    ax2.axhline(y=s, color='r', linestyle='-')
    ax3.axhline(y=s, color='r', linestyle='-')
    ax4.axhline(y=s, color='r', linestyle='-')
    ax5.axhline(y=s, color='r', linestyle='-')
# plt.plot(dataAct, range(len(dataAct)), linestyle='None', marker='x', markersize = 2, color='y')
ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax4.invert_yaxis()
ax5.invert_yaxis()
if sys.argv[-1] == "--save":
    plt.savefig(sys.argv[1]+".png")
else:
    plt.show()
