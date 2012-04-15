import numpy as np
import multiprocessing as mp
import sys

from Framework import Framework
from Simulator import LunarLanderSimulator
from PolicyGradientAgent import PolicyGradientAgent

simulator = LunarLanderSimulator(dt=1.0/30)
agent = PolicyGradientAgent (simulator, dt=0.5)
framework = Framework (simulator, agent)

def manual_control ():
    import Display
    user_agent = Display.UserAgent(simulator)
    window = Display.LunarLanderWindow (Framework (simulator, user_agent))

def visualize ():
    import Display
    window = Display.LunarLanderWindow (framework)
