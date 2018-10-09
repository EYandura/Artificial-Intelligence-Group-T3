#############################
#
# Reece Clingenpeel, Eric Yandura
#
# DOCUMENTATION:
#   I used only the course text and notes to complete this assignment.
#
##############################

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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Loop according to the supplied number of iterations.
        for iteration in range(self.iterations):
            # Make a copy of the values to accommodate pass by value.
            val = self.values.copy()
            # Get all the states in the mdp.
            states = mdp.getStates()

            for state in states:
                # Get all the possible actions for the current state.
                actions = mdp.getPossibleActions(state)

                if mdp.isTerminal(state) == False:
                    max = -99999

                    for action in actions:
                        v = 0
                        # Get all the possible transitions for each the state and action.
                        transitions = mdp.getTransitionStatesAndProbs(state, action)

                        for transition in transitions:
                            # Perform value iteration.
                            v = v \
                                + transition[1] \
                                * (mdp.getReward(state, action, transition[0]) + discount * self.values[transition[0]])

                        if v > max:
                            max = v

                        val[state] = max

                else:
                    for action in actions:
                        v = 0
                        # Get all the possible transitions for each the state and action.
                        transitions = mdp.getTransitionStatesAndProbs(state, action)

                        for transition in transitions:
                            # Perform value iteration.
                            v = v \
                                + transition[1] \
                                * (mdp.getReward(state, action, transition[0]) + discount * self.values[transition[0]])

                        val[state] = v

            self.values = val

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        qVal = 0
        # Get all the possible transitions for each the state and action.
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        for transition in transitions:
            # Perform value iteration.
            qVal = qVal \
                + transition[1] \
                * (self.mdp.getReward(state, action, transition[0]) + self.discount * self.values[transition[0]])

        return qVal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        # Get all the possible actions in the mdp.
        actions = self.mdp.getPossibleActions(state)

        if self.mdp.isTerminal(state) == False:
            # The best action is initially the first returned.
            bestAction = actions[0]
            # Calculate the base Q value from the tentative best action.
            bestQ = self.getQValue(state, bestAction)

            for action in actions:
                # Update the bestAction based on Q values.
                if self.getQValue(state, action) > bestQ:
                    bestQ = self.getQValue(state, action)
                    bestAction = action

            return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
