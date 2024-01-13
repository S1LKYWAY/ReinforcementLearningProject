# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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

import mdp
import util
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    def __init__(self, mdp, discount=0.9, iterations=100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with a default of 0
        self.runValueIteration()

    def runValueIteration(self):
        for _ in range(self.iterations):
            new_values = util.Counter()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    best_q_value = max(
                        self.computeQValueFromValues(state, action)
                        for action in self.mdp.getPossibleActions(state)
                    )
                    new_values[state] = best_q_value
            self.values = new_values

    def getValue(self, state):
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        q_value = 0.0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])
        return q_value

    def computeActionFromValues(self, state):
        if self.mdp.isTerminal(state):
            return None
        best_action = None
        best_q_value = float('-inf')
        for action in self.mdp.getPossibleActions(state):
            q_value = self.computeQValueFromValues(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    def __init__(self, mdp, discount=0.9, iterations=1000):
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = {}  # Dictionary to store predecessors of states

        for state in self.mdp.getStates():
            predecessors[state] = set()

        # Compute predecessors of all states
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if prob > 0:
                            predecessors[next_state].add(state)

        priority_queue = util.PriorityQueue()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                best_q_value = max(
                    self.computeQValueFromValues(state, action)
                    for action in self.mdp.getPossibleActions(state)
                )
                diff = abs(self.values[state] - best_q_value)
                priority_queue.update(state, -diff)

        for iteration in range(self.iterations):
            if priority_queue.isEmpty():
                break

            state = priority_queue.pop()
            if not self.mdp.isTerminal(state):
                best_q_value = max(
                    self.computeQValueFromValues(state, action)
                    for action in self.mdp.getPossibleActions(state)
                )
                self.values[state] = best_q_value

            for predecessor in predecessors[state]:
                if not self.mdp.isTerminal(predecessor):
                    diff = abs(self.values[predecessor] - max(
                        self.computeQValueFromValues(predecessor, action)
                        for action in self.mdp.getPossibleActions(predecessor)
                    ))
                    if diff > self.theta:
                        priority_queue.update(predecessor, -diff)
