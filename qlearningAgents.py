# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        return max(self.getQValue(state, action) for action in legalActions)

    def computeActionFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        maxQValue = max(self.getQValue(state, action) for action in legalActions)
        bestActions = [action for action in legalActions if self.getQValue(state, action) == maxQValue]
        return random.choice(bestActions)  # Break ties randomly

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)  # Exploration
        else:
            return self.computeActionFromQValues(state)  # Exploitation

    def update(self, state, action, nextState, reward):
        # Q-learning update: Q(s, a) = Q(s, a) + alpha * (R + gamma * max(Q(s', a')) - Q(s, a))
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha) * self.qValues[(state, action)] + self.alpha * sample

class PacmanQAgent(QLearningAgent):
    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action

class ApproximateQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        return self.weights * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        # Q-learning update with linear approximation
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for feature in features:
            self.weights[feature] += self.alpha * difference * features[feature]

    def final(self, state):
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            pass  # You can add code to run after training here