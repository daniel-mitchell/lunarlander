#!/usr/local/bin/python3
import sys, random, signal, math
import numpy as np
import scipy.special
from TileCoder import TileCoder, HashingTileCoder
from tilecoder2 import TileCoder as TileCoder2
from enum import Enum
import ctypes
import multiprocessing as mp

# NUM_EPISODES = 100
# MAX_TIMESTEPS_PER_EPISODE = 1000000
# ALPHA_U = 0.05
# ALPHA_V = 1
ALPHA_R = 0.0
# ALPHA_N = 1.0
# TAU = 16
GAMMA = 1
EPSILON = 0#.0001
INAC = False
S = False
EPISODES_TO_PRINT = []
MAX_ALPHA_BETA = 1000
MAX_PARAM = math.log(MAX_ALPHA_BETA)
# SWITCH_THRESHOLD = 1
ALPHA_DECAY_RATE = 0.8 #Higher value is slower decay, 0 is none

def sigIntHandler(sig, frame):
    print(frame.f_lineno)
    env.close()
    exit(1)

signal.signal(signal.SIGINT, sigIntHandler)

class MultiDistAgent:
    # def __init__(self, alphaU, alphaVInit, alphaDecay, tau, epsilon, inac, s, seed, printOutput, numEpisodes):
    def __init__(self, simulator, dt=0.5, Lambda=0.75, alpha_v=0.1, alpha_u=0.1, num_features=2**20,
                    tile_weight_exponent=0.5, trunc_normal=True, subspaces=[1,2,6], seed=0):
        print("Starting Seed",seed)
        self.printOutput = True
        self.simulator = simulator
        self.minAction = (0, -simulator.max_rcs)
        self.maxAction = (simulator.max_thrust, simulator.max_rcs)
        self.dt = max(dt, self.simulator.dt)
        self.alphaDecay = ALPHA_DECAY_RATE
        self.lambdaa = Lambda
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.inac = INAC
        self.s = S
        # self.printOutput = printOutput
        # self.numEpisodes = numEpisodes
        self.rng = np.random.RandomState(seed)
        self.rng2 = np.random.RandomState(seed)
        
        # self.tc = TileCoder2()
        self.tc = self.make_tile_coder(tile_weight_exponent, subspaces)
        self.u = np.zeros(self.tc.num_features*6) #Parameters
        self.v = np.zeros(self.tc.num_features*2) #Weights for critic
        #self.v = np.ones(tc.num_features*2)*0.1 #Weights for critic
        self.w = np.zeros(self.tc.num_features*6) #Weights for actor
        self.maxReturn = 0
        self.timedOut = 0
        #TODO:Divide by 4 and 1 for number active at once, or 6 and 2 for total
        self.alphaU = alpha_u/(4*self.tc.active_features)
        self.alphaVInit = alpha_v/(self.tc.active_features)
        self.maxParam = MAX_PARAM/self.tc.active_features
        self.returns = []
        self.times = []
        self.i_episode = 0

    def make_tile_coder (self, tile_weight_exponent, subspaces):
        #                            xpos   ypos  xvel  yvel        rot     rotvel
        state_signed  = np.array ([ False, False, True, True,      True,      True ])
        state_bounded = np.array ([  True,  True, True, True,     False,      True ])
        tile_size     = np.array ([    5.,    5.,   2.,   2., math.pi/2, math.pi/6 ])
        num_tiles     = np.array ([     6,     4,    4,    4,         2,         3 ])
        num_offsets   = np.array ([     2,     2,    4,    4,         8,         4 ])

        self.max_state = (tile_size * num_tiles) - 1e-8

        self.min_state = -self.max_state
        self.min_state[np.logical_not(state_signed)] = 0.0

        self.max_clip_state = self.max_state.copy()
        self.max_clip_state[np.logical_not(state_bounded)] = float('inf')

        self.min_clip_state = -self.max_clip_state
        self.min_clip_state[np.logical_not(state_signed)] = 0.0

        num_tiles[state_signed] *= 2
        num_tiles[state_bounded] += 1

        return TileCoder (tile_size, num_tiles, num_offsets, subspaces, tile_weight_exponent)

    def initialize(self, state):
        self.i_episode += 1
        print("Starting Episode",self.i_episode)
        self.x = np.array(self.tc.indices(state))
        self.returnAmount = 0
        self.rBar = 0
        self.ev = np.zeros(self.tc.num_features*2) #Weight trace
        self.eu = np.zeros(self.tc.num_features*6) #Parameter trace
        if self.alphaDecay != 0:
            self.alphaV = self.alphaVInit*(math.log(1+self.alphaDecay)/math.log(1+self.alphaDecay*(self.i_episode)))
        else:
            self.alphaV = self.alphaVInit
        return self.computeAction()

    def update(self, state, reward, terminal=False, learn=True):
        self.returnAmount += reward
        xPrime = np.array(self.tc.indices(state))
        self.chosenDistPrime = int(np.sum(self.v[xPrime]) < np.sum(self.v[xPrime + self.tc.num_features]))

        #Update values
        self.delta = reward - self.rBar + self.gamma*(np.sum(self.v[xPrime])+np.sum(self.v[xPrime + \
                    (1+self.chosenDistPrime)*self.tc.num_features])) -\
                    (np.sum(self.v[self.x]) + np.sum(self.v[self.x + (1+self.chosenDist)*self.tc.num_features]))
        self.rBar += ALPHA_R*self.delta
        self.ev *= self.gamma*self.lambdaa
        self.ev[self.x] += 1
        self.ev[self.x + (1+self.chosenDist)*self.tc.num_features] += 1
        self.v += self.alphaV*self.delta*self.ev

        #Compute gradlog
        self.gradLog = self.computeGradLog()
        self.variance = self.alpha*self.beta/\
                        ((self.alpha + self.beta)*(self.alpha+self.beta)*(self.alpha+self.beta+1))

        #Update parameters
        self.eu = self.gamma*self.lambdaa*self.eu + self.gradLog
        if self.inac:
            self.w += self.alphaV*(self.delta*self.eu - np.dot(self.gradLog, self.gradLog)*self.w)
            if self.s:
                self.u += self.alphaU*self.w*self.variance
            else:
                self.u += self.alphaU*self.w
        else:
            if self.s:
                self.u += self.alphaU*self.delta*self.eu*self.variance
            else:
                self.u += self.alphaU*self.delta*self.eu
        self.u = np.minimum(self.u, self.maxParam)
        self.x = xPrime

        if self.printOutput and not np.isfinite(self.u).all():
            print(self.delta)
            print(self.eu)

        return self.computeAction()
        #TODO: Might need to handle terminal case

    def printAgentValues(self, params, weights, episode, tc, env):
        with open("agentValuesMulti_{}_low.csv".format(episode), "w") as f:
            for s0 in range(tc.numTiles[0]*tc.numTilings):
                for s1 in range(tc.numTiles[1]*tc.numTilings):
                    d0 = s0/(tc.numTiles[0]*tc.numTilings)*(env.high_state[0] - env.low_state[0]) + env.low_state[0]
                    d1 = s1/(tc.numTiles[1]*tc.numTilings)*(env.high_state[1] - env.low_state[1]) + env.low_state[1]
                    tiles = np.array(tc.indices(np.array([d0, d1])))
                    alpha = math.exp(np.sum(params[tiles])) + 1
                    beta = math.exp(np.sum(params[tiles + tc.num_features])) + 1
                    mean = alpha/(alpha+beta)
                    var = alpha*beta/((alpha+beta)*(alpha+beta)*(alpha+beta+1))

        with open("agentValuesMulti_{}_high.csv".format(episode), "w") as f:
            for s0 in range(tc.numTiles[0]*tc.numTilings):
                for s1 in range(tc.numTiles[1]*tc.numTilings):
                    d0 = s0/(tc.numTiles[0]*tc.numTilings)*(env.high_state[0] - env.low_state[0]) + env.low_state[0]
                    d1 = s1/(tc.numTiles[1]*tc.numTilings)*(env.high_state[1] - env.low_state[1]) + env.low_state[1]
                    tiles = np.array(tc.indices(np.array([d0, d1])))
                    alpha = math.exp(np.sum(params[tiles + 2*tc.num_features])) + 1
                    beta = math.exp(np.sum(params[tiles + 3*tc.num_features])) + 1
                    mean = alpha/(alpha+beta)
                    var = alpha*beta/((alpha+beta)*(alpha+beta)*(alpha+beta+1))
                    f.write(",".join(map(str, [d0, d1, mean, var])) + "\n")

        with open("agentValuesMulti_{}_comp.csv".format(episode), "w") as f:
            for s0 in range(tc.numTiles[0]*tc.numTilings):
                for s1 in range(tc.numTiles[1]*tc.numTilings):
                    d0 = s0/(tc.numTiles[0]*tc.numTilings)*(env.high_state[0] - env.low_state[0]) + env.low_state[0]
                    d1 = s1/(tc.numTiles[1]*tc.numTilings)*(env.high_state[1] - env.low_state[1]) + env.low_state[1]
                    tiles = np.array(tc.indices(np.array([d0, d1])))
                    if np.sum(weights[tiles]) < np.sum(weights[tiles + tc.num_features]):
                        f.write(",".join(map(str, [d0, d1, 1])) + "\n")
                    else:
                        f.write(",".join(map(str, [d0, d1, 0])) + "\n")

    def computeAction(self):
        self.action, self.chosenDist, self.alpha, self.beta = self.selectAction()
        self.action = np.hstack(np.array([self.action]))
        scaledAction = self.action.copy()
        scaledAction[0] = scaledAction[0]*(self.maxAction[0] - self.minAction[0])\
                            + self.minAction[0]
        if self.chosenDist:
            scaledAction[1] *= self.maxAction[1]
        else:
            scaledAction[1] *= self.minAction[1]
        return scaledAction

    def selectAction(self):
        if self.rng2.random() < self.epsilon:
            distOffset = int(self.rng2.random() < 0.5)
        else:
            distOffset = int(np.sum(self.v[self.x]) < np.sum(self.v[self.x + self.tc.num_features]))
        distOffset *= 2*self.tc.num_features
        alpha0 = math.exp(np.sum(self.u[self.x])) + 1
        beta0 = math.exp(np.sum(self.u[self.x + self.tc.num_features])) + 1
        action0 = self.rng.beta(alpha0, beta0)
        alpha1 = math.exp(np.sum(self.u[self.x +self.tc.num_features*2 + distOffset])) + 1
        beta1 = math.exp(np.sum(self.u[self.x + self.tc.num_features*3 + distOffset])) + 1
        action1 = self.rng.beta(alpha1, beta1)
        return (action0, action1), int(distOffset != 0), np.array([alpha0, alpha1]), np.array([beta0, beta1])

    def computeGradLog(self):
        gradLogAlpha0 = np.zeros(self.tc.num_features)
        gradLogBeta0 = np.zeros(self.tc.num_features)
        gradLogAlpha1 = np.zeros(self.tc.num_features)
        gradLogBeta1 = np.zeros(self.tc.num_features)
        try:
            gradLogAlpha0[self.x] = (math.log(self.action[0]) + scipy.special.digamma(self.alpha[0] + self.beta[0])\
                            - scipy.special.digamma(self.alpha[0])) * (self.alpha[0] - 1)
            gradLogBeta0[self.x] = (math.log(1 - self.action[0]) + scipy.special.digamma(self.alpha[0] + self.beta[0])
                            - scipy.special.digamma(self.beta[0])) * (self.beta[0] - 1)
            gradLogAlpha1[self.x] = (math.log(self.action[1]) + scipy.special.digamma(self.alpha[1] + self.beta[1])\
                            - scipy.special.digamma(self.alpha[1])) * (self.alpha[1] - 1)
            gradLogBeta1[self.x] = (math.log(1 - self.action[1]) + scipy.special.digamma(self.alpha[1] + self.beta[1])\
                            - scipy.special.digamma(self.beta[1])) * (self.beta[1] - 1)
        except ValueError as e:
            if self.printOutput:
                print("Action:", self.action)
                print("Alpha:", self.alpha)
                print("Beta:", self.beta)
                # print("Max parameter:", np.max(np.abs(u)))
                # print("Timestep:", t)
            raise(e)
        if self.chosenDist:
            gradLog = np.concatenate([gradLogAlpha0, gradLogBeta0, np.zeros(2*self.tc.num_features),
                                        gradLogAlpha1, gradLogBeta1])
        else:
            gradLog = np.concatenate([gradLogAlpha0, gradLogBeta0, gradLogAlpha1, gradLogBeta1,
                                        np.zeros(2*self.tc.num_features)])
        return gradLog

    def get_state(self):
        return np.vstack ((self.v.reshape(2, self.tc.num_features), self.u.reshape(6, self.tc.num_features)))

    def set_state(self, state):
        state.shape = (8, self.tc.num_features)
        self.v = state[:2].reshape(2*self.tc.num_features)
        self.u = state[2:].reshape(6*self.tc.num_features)

    def save_state (self, savefile='data/saved_state.npy'):
        np.save (savefile, self.get_state())

    def load_state (self, savefile='data/saved_state.npy', mmap_mode=None):
        state = np.array (np.load (savefile, mmap_mode), copy=False)
        self.set_state(state)

    def persist_state(self, savefile=None, readonly=False):
        if savefile == None:
            state = np.frombuffer(mp.RawArray(ctypes.c_double, int(8*self.tc.num_features)))
            state[:] = self.get_state().flat
            self.set_state(state)
        else:
            if not readonly: self.save_state(savefile)
            self.load_state (savefile, mmap_mode='r' if readonly else 'r+')
