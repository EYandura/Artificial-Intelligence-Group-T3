#############################
#
# Eric Yandura and Reece Clingenpeel
#
# Links used:
# https://pymotw.com/2/collections/counter.html
# http://scikit-learn.org/stable/modules/feature_extraction.html
# https://machinelearningmastery.com/feature-selection-machine-learning-python/
# No other help recieved
#
##############################

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
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # set up the q values
        self.qValues = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # create the q value tuple based on the state and action
        return self.qValues[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE *** "
        # get all of the actions possible for the state and initialize max q to 0
        legalActions = self.getLegalActions(state)
        maximum_q = 0
        # if there are legal actions, conitinue. If there are not, return the initialized max q
        if legalActions:
            # initialize the max value to negative infinity
            maximum_q = -9999999
            # iterate through each possible action and find the highest q of the options
            for each_action in legalActions:
                # get the new q value from the current action
                current_q = self.getQValue(state, each_action)
                # if our current max is less than our new q, replace it.
                if maximum_q < current_q:
                    maximum_q = current_q
            # if there are no legal actions, return initialized value
        return maximum_q


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # get all of the possible actions and initalize the best action
        legalActions = self.getLegalActions(state)
        best_action = None
        # if there are valid actions, continue into if
        if legalActions:
            # initialize the maximum value to negative infinity
            maximum_q = -9999999
            # iterate through each action to check the q values to get the best
            for each_action in legalActions:
                # get the q value of the action you are checking
                current_q_value = self.getQValue(state, each_action)
                # if our current max is less than our new q, replace the action.
                if maximum_q <= current_q_value:
                    maximum_q = current_q_value
                    best_action = each_action
        # return the best action.
        return best_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        # if there are legal actions, choose the action based on the probability self.epsilon
        if legalActions:
            #  With the probability self.epsilon, we should take a random action
            if util.flipCoin(self.epsilon) == True:
                # To pick randomly from a list, use random.choice(list)
                action = random.choice(legalActions)
            # take the best policy action otherwise
            else:
                action = self.getPolicy(state)
            return action
        # if you are at a terminal state, you should return None as the action
        else:
            return None

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        # initialize the legal actions of the next state
        legalActions = self.getLegalActions(nextState)
        # initialize the R(s, a, s')
        Rsas = reward


        # difference = [R + (discount * (maxQ(s', a')))] - Q(s, a)

        # create the maxQ(s',a') and update R if there are actions
        if legalActions:
            # initialize a list of the q's from each of the next state's actions
            q_values_at_next_state = []

            # append the q value of the next state with each action
            for each_action in legalActions:
                q_values_at_next_state.append(self.getQValue(nextState, each_action))

            # get the best action's Q-value
            maxQ = max(q_values_at_next_state)
            # reward is the current reward plus the discount * best Q-Value in the next state
            Rsas = reward + (self.discount * maxQ)


        # Q(s,a) = Q(s,a) + (learning_rate * difference)
        # intialize the learning rate
        learn_rate = self.alpha
        Qsa = self.getQValue(state, action)
        difference = Rsas - Qsa
        self.qValues[(state, action)] = Qsa + (learn_rate * difference)



    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):

    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        # calls the getAction method of QLeaning Agent
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"

        # Q(s,a) = SUM(  fi(s,a) * weight    )

        f = self.featExtractor.getFeatures(state, action)

        Qsa = 0

        for feature in f:
            fi = f[feature]
            weight = self.weights[feature]

            Qsa = Qsa + (fi * weight)
        return Qsa


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        # difference = [Rsa + (discount * (maxQ(s', a')))] - Q(s, a)   ***SAME AS ABOVE***

        Rsa = reward
        f = self.featExtractor.getFeatures(state, action)

        discount = self.discount
        maxQ = self.getValue(nextState)
        Qsa = self.getQValue(state, action)
        a = self.alpha

        diff = (Rsa + discount * maxQ) - Qsa

        # weights = weights + a * difference * f(s, a)
        for each_feat in f.keys():
            self.weights[each_feat] = self.weights[each_feat] + a * diff * f[each_feat]



    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
