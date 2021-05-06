import numpy as np
from tiles3 import tiles, tileswrap, IHT
import math

class TileCoder:
    def __init__(self):
        #Bounds:
        #X:[-1, 1]
        #Y:
        #V_x:
        #V_y:
        #Theta: [0, 2pi)
        #Omega:
        # minVals = minVals.astype(np.float64)
        # maxVals = maxVals.astype(np.float64)
        # self.numTilings = 10
        # self.numTiles = np.array([10, 10])
        # self.dims = len(minVals)
        # self.prevTiles = [int(np.product(self.numTiles[:i])) for i in range(self.dims + 1)]
        # self.minVals = minVals
        # self.ranges = (maxVals - minVals)/(self.numTiles - 1)
        # self.totalTiles = self.numTilings*self.prevTiles[-1] + 1
        # self.totalTilings = self.numTilings + 1
        minVals = [-10, -10, -10, -10, 0, -10]
        maxVals = [10, 10, 10, 10, 2*math.pi, 10]
        self.scales = [10, 10, 10, 10, 10/(2*math.pi), 10]
        self.wrapWidths = [False, False, False, False, 10, False]
        self.numTilings = [10, 10, 10, 10, 10, 10]
        self.numTiles = [(maxVals[i] - minVals[i])*self.scales[i]*self.numTilings[i] for i in range(len(minVals))]
        self.totalTilings = int(sum(self.numTilings))
        self.totalTiles = int(sum(self.numTiles))
        self.iht = IHT(self.totalTiles)
        #rename for portablility
        self.active_features = self.totalTilings
        self.num_features = self.totalTiles
    
    #rename for portablility
    def indices(selff, observation):
        return self.tilecode(observation)
    
    def tilecode(self, observation):
        indices = list()
        for i in range(6):
            indices += tileswrap(self.iht, self.numTilings[i], [observation[i]*self.scales[i]],\
                        [self.wrapWidths[i]], [i])
        return indices
        # observation = np.hstack(observation)
        # if len(self.numTiles) != len(observation):
        #     raise ValueError("observation does not have the correct length for this coding")
        # try:
        #     if not np.isfinite(observation).any():
        #         print(observation)
        #         raise ValueError("observation has infinite or NaN values")
        # except TypeError as e:
        #     print(observation.dtype)
        #     print(observation)
        #     raise e
        #
        # vals = observation - self.minVals
        # indices = [0]*self.numTilings
        # for i in range(self.numTilings):
        #     index = 0
        #     for j in range(len(vals)):
        #         index += int((vals[j] + i*self.ranges[j]/self.numTilings)//self.ranges[j])*self.prevTiles[j]
        #     indices[i] = index + i*self.prevTiles[-1]
        # indices.append(self.totalTiles - 1)
        # return indices
