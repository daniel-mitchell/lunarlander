#!/usr/local/bin/python3
import gym, sys, random, signal, math
import numpy as np
import scipy.special
from tilecoder2 import TileCoder
from enum import Enum

NUM_EPISODES = 100
MAX_TIMESTEPS_PER_EPISODE = 1000000
ALPHA_U = 0.05
ALPHA_V = 1
ALPHA_R = 0.0
ALPHA_N = 1.0
TAU = 16
GAMMA = 1
EPSILON = 0#.0001
INAC = False
S = False
EPISODES_TO_PRINT = []
MAX_ALPHA_BETA = 1000
MAX_PARAM = math.log(MAX_ALPHA_BETA)
SWITCH_THRESHOLD = 1
ALPHA_DECAY_RATE = 0.8 #Higher value is slower decay, 0 is none

class Dist(Enum):
    Beta = 1
    Gaussian =2
    Beta0 = 3

DIST = Dist.Beta

def sigIntHandler(sig, frame):
    print(frame.f_lineno)
    env.close()
    exit(1)

signal.signal(signal.SIGINT, sigIntHandler)

def printAgentValues(params, weights, episode, tc, env):
    with open("agentValuesMulti_{}_low.csv".format(episode), "w") as f:
        for s0 in range(tc.numTiles[0]*tc.numTilings):
            for s1 in range(tc.numTiles[1]*tc.numTilings):
                d0 = s0/(tc.numTiles[0]*tc.numTilings)*(env.high_state[0] - env.low_state[0]) + env.low_state[0]
                d1 = s1/(tc.numTiles[1]*tc.numTilings)*(env.high_state[1] - env.low_state[1]) + env.low_state[1]
                tiles = np.array(tc.tilecode(np.array([d0, d1])))
                alpha = math.exp(np.sum(params[tiles])) + 1
                beta = math.exp(np.sum(params[tiles + tc.totalTiles])) + 1
                mean = alpha/(alpha+beta)
                var = alpha*beta/((alpha+beta)*(alpha+beta)*(alpha+beta+1))

    with open("agentValuesMulti_{}_high.csv".format(episode), "w") as f:
        for s0 in range(tc.numTiles[0]*tc.numTilings):
            for s1 in range(tc.numTiles[1]*tc.numTilings):
                d0 = s0/(tc.numTiles[0]*tc.numTilings)*(env.high_state[0] - env.low_state[0]) + env.low_state[0]
                d1 = s1/(tc.numTiles[1]*tc.numTilings)*(env.high_state[1] - env.low_state[1]) + env.low_state[1]
                tiles = np.array(tc.tilecode(np.array([d0, d1])))
                alpha = math.exp(np.sum(params[tiles + 2*tc.totalTiles])) + 1
                beta = math.exp(np.sum(params[tiles + 3*tc.totalTiles])) + 1
                mean = alpha/(alpha+beta)
                var = alpha*beta/((alpha+beta)*(alpha+beta)*(alpha+beta+1))
                f.write(",".join(map(str, [d0, d1, mean, var])) + "\n")

    with open("agentValuesMulti_{}_comp.csv".format(episode), "w") as f:
        for s0 in range(tc.numTiles[0]*tc.numTilings):
            for s1 in range(tc.numTiles[1]*tc.numTilings):
                d0 = s0/(tc.numTiles[0]*tc.numTilings)*(env.high_state[0] - env.low_state[0]) + env.low_state[0]
                d1 = s1/(tc.numTiles[1]*tc.numTilings)*(env.high_state[1] - env.low_state[1]) + env.low_state[1]
                tiles = np.array(tc.tilecode(np.array([d0, d1])))
                if np.sum(weights[tiles]) < np.sum(weights[tiles + tc.totalTiles]):
                    f.write(",".join(map(str, [d0, d1, 1])) + "\n")
                else:
                    f.write(",".join(map(str, [d0, d1, 0])) + "\n")

def selectAction(activeTiles, params, weights, numFeatures, epsilon, rng, rng2):
    if rng2.random() < epsilon:
        distOffset = int(rng2.random() < 0.5)
    else:
        distOffset = int(np.sum(weights[activeTiles]) < np.sum(weights[activeTiles + numFeatures]))
    distOffset *= 2*numFeatures
    alpha0 = math.exp(np.sum(params[activeTiles])) + 1
    beta0 = math.exp(np.sum(params[activeTiles + numFeatures])) + 1
    action0 = rng.beta(alpha0, beta0)
    alpha1 = math.exp(np.sum(params[activeTiles + numFeatures*2 + distOffset])) + 1
    beta1 = math.exp(np.sum(params[activeTiles + numFeatures*3 + distOffset])) + 1
    action1 = rng.beta(alpha1, beta1)
    return (action0, action1), int(distOffset != 0), np.array([alpha0, alpha1]), np.array([beta0, beta1])

# def selectActionBeta0(activeTiles, params, weights, numFeatures, epsilon, rng, rng2):
#     if rng2.random() < epsilon:
#         distOffset = int(rng2.random() < 0.5)
#     else:
#         distOffset = int(np.sum(weights[activeTiles]) < np.sum(weights[activeTiles + numFeatures]))
#     distOffset *= 2*numFeatures
#     alpha = math.exp(np.sum(params[activeTiles + distOffset]))
#     beta = math.exp(np.sum(params[activeTiles + numFeatures + distOffset])) + 1
#     action = rng.beta(alpha, beta)
#     return action, int(distOffset != 0), alpha, beta

def computeGradLog(x, action, alpha, beta, tc, chosenDist, printOutput):
    gradLogAlpha0 = np.zeros(tc.totalTiles)
    gradLogBeta0 = np.zeros(tc.totalTiles)
    gradLogAlpha1 = np.zeros(tc.totalTiles)
    gradLogBeta1 = np.zeros(tc.totalTiles)
    try:
        gradLogAlpha0[x] = (math.log(action[0]) + scipy.special.digamma(alpha[0] + beta[0]) -\
                        scipy.special.digamma(alpha[0])) * (alpha[0] - 1)
        gradLogBeta0[x] = (math.log(1 - action[0]) + scipy.special.digamma(alpha[0] + beta[0]) -\
                        scipy.special.digamma(beta[0])) * (beta[0] - 1)
        gradLogAlpha1[x] = (math.log(action[1]) + scipy.special.digamma(alpha[1] + beta[1]) -\
                        scipy.special.digamma(alpha[1])) * (alpha[1] - 1)
        gradLogBeta1[x] = (math.log(1 - action[1]) + scipy.special.digamma(alpha[1] + beta[1]) -\
                        scipy.special.digamma(beta[1])) * (beta[1] - 1)
    except ValueError as e:
        if printOutput:
            print("Action:", action)
            print("Alpha:", alpha)
            print("Beta:", beta)
            # print("Max parameter:", np.max(np.abs(u)))
            # print("Timestep:", t)
        raise(e)
    if chosenDist:
        gradLog = np.concatenate([gradLogAlpha0, gradLogBeta0, np.zeros(2*tc.totalTiles), gradLogAlpha1, gradLogBeta1])
    else:
        gradLog = np.concatenate([gradLogAlpha0, gradLogBeta0, gradLogAlpha1, gradLogBeta1, np.zeros(2*tc.totalTiles)])
    return gradLog

# def computeGradLogBeta0(x, action, alpha, beta, tc, chosenDist, printOutput):
#     gradLogAlpha = np.zeros(tc.totalTiles)
#     gradLogBeta = np.zeros(tc.totalTiles)
#     try:
#         gradLogAlpha[x] = (math.log(action) + scipy.special.digamma(alpha + beta) -\
#                         scipy.special.digamma(alpha)) * alpha
#         gradLogBeta[x] = (math.log(1 - action) + scipy.special.digamma(alpha + beta) -\
#                         scipy.special.digamma(beta)) * (beta - 1)
#     except ValueError as e:
#         if printOutput:
#             print("Action:", action)
#             print("Alpha:", alpha)
#             print("Beta:", beta)
#             # print("Max parameter:", np.max(np.abs(u)))
#             # print("Timestep:", t)
#         raise(e)
#     if chosenDist:
#         gradLog = np.concatenate([np.zeros(2*tc.totalTiles), gradLogAlpha, gradLogBeta])
#     else:
#         gradLog = np.concatenate([gradLogAlpha, gradLogBeta, np.zeros(2*tc.totalTiles)])
#     return gradLog

def actorcritic(alphaU, alphaVInit, alphaDecay, tau, epsilon, inac, s, seed, printOutput, numEpisodes):
    lambdaa = 1 - 1/tau
    env = gym.make("LunarLanderContinuous-v2").env
    #env = gym.make("MountainCarContinuousModReward-v0")
    #env = gym.make("MountainCarContinuousModReward2-v0")
    env._max_episode_steps = MAX_TIMESTEPS_PER_EPISODE
    env.seed(seed)
    rng = np.random.RandomState(seed)
    rng2 = np.random.RandomState(seed)
    
    #Setup Agent
    tc = TileCoder()
    u = np.zeros(tc.totalTiles*6) #Parameters
    v = np.ones(tc.totalTiles*3)*0.1 #Weights for critic
    #v = np.concatenate([np.ones(tc.totalTiles)*0.1, np.ones(tc.totalTiles)*-1000000]) #Weights for critic
    w = np.zeros(tc.totalTiles*6) #Weights for actor
    maxReturn = 0
    timedOut = 0
    alphaU /= 6*tc.totalTilings
    alphaVInit /= 4*tc.totalTilings
    maxParam = MAX_PARAM/tc.totalTilings
    returns = []
    times = []
    for i_episode in range(numEpisodes):
        #Setup Episode
        observation = env.reset()
        x = np.array(tc.tilecode(observation))
        returnAmount = 0
        rBar = 0
        ev = np.zeros(tc.totalTiles*3) #Weight trace
        eu = np.zeros(tc.totalTiles*6) #Parameter trace
        if alphaDecay != 0:
            alphaV = alphaVInit*(math.log(1+alphaDecay)/math.log(1+alphaDecay*(i_episode+1)))
        else:
            alphaV = alphaVInit
    
        for t in range(MAX_TIMESTEPS_PER_EPISODE):
            #Take action
            #env.render()
            if DIST == Dist.Beta:
                action, chosenDist, alpha, beta =\
                        selectAction(x, u, v,tc.totalTiles, epsilon, rng, rng2)
                action = np.hstack(np.array([action]))
                scaledAction = action.copy()
                scaledAction[0] = scaledAction[0]*(env.action_space.high[0] - env.action_space.low[0])\
                                    + env.action_space.low[0]
                if chosenDist:
                    scaledAction[1] *= env.action_space.high[1]
                else:
                    scaledAction[1] *= env.action_space.low[1]
            else:
                raise ValueError("Invalid Distribution")
            actionOld = action
            if printOutput and not np.isfinite(action).any():
                print(action)
                print(actionOld)
                print(alpha)
                print(beta)
                print(x)
                print(u[x])
                env.close()
                raise ValueError("action has infinite or NaN values")
            #scaledAction = action*(env.action_space.high - env.action_space.low) + env.action_space.low
            observation, reward, done, info = env.step(scaledAction)
            returnAmount += reward
            xPrime = np.array(tc.tilecode(observation))
            chosenDistPrime = int(np.sum(v[xPrime]) < np.sum(v[xPrime + tc.totalTiles]))
        
            #Update values
            #xAll = np.concatenate([x, x + tc.totalTiles])
            #xPrimeAll = np.concatenate([xPrime, xPrime + tc.totalTiles])
            delta = reward - rBar + GAMMA*(np.sum(v[xPrime])+np.sum(v[xPrime + (1+chosenDistPrime)*tc.totalTiles])) -\
                        (np.sum(v[x]) + np.sum(v[x + (1+chosenDist)*tc.totalTiles]))
            rBar += ALPHA_R*delta
            ev *= GAMMA*lambdaa
            ev[x] += 1
            ev[x + (1+chosenDist)*tc.totalTiles] += 1
            v += alphaV*delta*ev
            # gradLogAlpha = np.zeros(tc.totalTiles)
            # gradLogBeta = np.zeros(tc.totalTiles)
            # try:
            #     gradLogAlpha[x] = (math.log(action) + scipy.special.digamma(alpha + beta) -\
            #                     scipy.special.digamma(alpha)) * (alpha - 1)
            #     gradLogBeta[x] = (math.log(1 - action) + scipy.special.digamma(alpha + beta) -\
            #                     scipy.special.digamma(beta)) * (beta - 1)
            # except ValueError as e:
            #     if printOutput:
            #         print("Action:", action)
            #         print("Alpha:", alpha)
            #         print("Beta:", beta)
            #         print("Max parameter:", np.max(np.abs(u)))
            #         print("Timestep:", t)
            #     raise(e)
            # if chosenDist:
            #     gradLog = np.concatenate([np.zeros(2*tc.totalTiles), gradLogAlpha, gradLogBeta])
            # else:
            #     gradLog = np.concatenate([gradLogAlpha, gradLogBeta, np.zeros(2*tc.totalTiles)])
            if DIST == Dist.Beta:
                gradLog = computeGradLog(x, action, alpha, beta, tc, chosenDist, printOutput)
                variance = alpha*beta/\
                                ((alpha + beta)*(alpha+beta)*(alpha+beta+1))
            else:
                raise ValueError("Invalid Distribution")
            eu = GAMMA*lambdaa*eu + gradLog
            if inac:
                w += alphaV*(delta*eu - np.dot(gradLog, gradLog)*w)
                if s:
                    u += alphaU*w*variance
                else:
                    u += alphaU*w
            else:
                if s:
                    u += alphaU*delta*eu*variance
                else:
                    u += alphaU*delta*eu
            u = np.minimum(u, maxParam)
            x = xPrime
            
            if printOutput and not np.isfinite(u).all():
                print(delta)
                print(eu)
            
            #Check if at terminal state
            if done:
                if printOutput:
                    print("Episode {} finished after {} timesteps with return {}"\
                            .format(i_episode + 1, t+1, returnAmount))
                # print("Action:", action)
                # print("Alpha:", alpha)
                # print("Beta:", beta)
                # print("Max parameter:", np.max(np.abs(u)))
                returns.append(returnAmount)
                times.append(t+1)
                maxReturn = max(maxReturn, returnAmount)
                break
        else:
            if printOutput:
                print("Episode {} timed out after {} timesteps with return {}"\
                        .format(i_episode + 1, t + 1, returnAmount))
            returns.append(returnAmount)
            times.append(MAX_TIMESTEPS_PER_EPISODE)
            maxReturn = max(maxReturn, returnAmount)
            #raise Exception("Timed out")
            timedOut += 1
    averageReturn = sum(returns)/len(returns)
    averageTime = sum(times)/len(times)
    averageThresholdReturn = sum(returns[SWITCH_THRESHOLD:])/len(returns[SWITCH_THRESHOLD:])
    if printOutput:
        #printAgentValues(u, v, i_episode + 1, tc, env)
        print("Maximum return was {}".format(maxReturn))
        print("Average return was {}".format(averageReturn))
        print("Average timesteps was {}".format(averageTime))
        print("{} episode(s) timed out".format(timedOut))
    env.close()
    return averageReturn, averageThresholdReturn, averageTime

if __name__ == "__main__":
    actorcritic(ALPHA_U, ALPHA_V, ALPHA_DECAY_RATE, TAU, EPSILON, INAC, S, int(sys.argv[1]), True, NUM_EPISODES)
