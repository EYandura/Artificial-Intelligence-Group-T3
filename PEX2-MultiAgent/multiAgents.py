# ##############################
#
# Reece Clingenpeel, Eric Yandura
#
# DOCUMENTATION:
# ~ The Python Doc was used throughout this file in order to explore the built in Python structures and functionality.
# ~ The class text and Notes were used throughout the assignment.
# I followed along with a similar problem online to better understand how to do number 1 (Eric Yandura) '
# Did not just copy, once I already had my solution, it was used to help find errors
# https://github.com/advaypakhale/Berkeley-AI-Pacman-Projects/blob/master/multiagent/multiAgents.py
#
# ###############################

# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE Question 1***"
        # get the minimum distance to the closest ghost

        ghostDistance = min([manhattanDistance(newPos, each.getPosition()) for each in newGhostStates])

        if ghostDistance:
            ghost_dist = -10 / ghostDistance
        else:
            ghost_dist = -1000000

        list_of_food = newFood.asList()

        # get the distance to the closest food
        if list_of_food:
            closeFood = min([manhattanDistance(newPos, food) for food in list_of_food])
        # there is no food
        else:
            closeFood = 0

        # return the weighted score
        return (-1 * closeFood) + ghost_dist - (100 * len(list_of_food))


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (Question 2)
      Perform a post order traversal of the game tree assuming that the opponent behaves optimally.
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # TODO count to keep track of iteration and agent (each agent needs x iterations)

        agentIndex = self.index
        depth = self.depth

        return self.value(gameState, agentIndex, depth)[1]

    def value(self, gameState, agentIndex, depth):
        """
        If the gameState is a terminal state, return it. Otherwise, find the min/max and return the minimax value.

        :param gameState: The current state of the game.
        :return: The minimax value at the state gameState.
        """
        # If the current depth is 0, we have traversed the tree; return the value of the evaluation function.
        if depth == 0:
            return (self.evaluationFunction(gameState), '')

        # If gameState is a terminal state (win or loss), return the value of the evaluation function.
        if gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), '')

        # If the agent is Pacman (agentIndex == 0) find the max.
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        # Otherwise, the agent is a Ghost (agentIndex >= 1); find the min.
        elif agentIndex >= 1:
            return self.minValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        """
        Return the max value of the successors of gameState.

        :param gameState: The current state of the game.
        :return: The max value of gameState's successors.
        """

        # Initialize v to a min value.
        v = -99999

        # The number of agents in the game.
        numAgents = gameState.getNumAgents()

        # The nextAgentIndex will be agentIndex + 1 (mod numAgents) as we want to iterate each agent
        # once for each level of depth.
        nextAgentIndex = (agentIndex + 1) % numAgents

        # If we are looking at the last agent in the game, decrement the depth by 1.
        if agentIndex == numAgents - 1:
            nextDepth = depth - 1
        # Otherwise, the depth remains the same.
        else:
            nextDepth = depth

        # Return a list of all the legal actions of the agent.
        actions = gameState.getLegalActions(agentIndex)

        # Loop the legal actions and find the max.
        for a in actions:
            successor = gameState.generateSuccessor(agentIndex, a)
            actionVal = self.value(successor, nextAgentIndex, nextDepth)
            if actionVal[0] > v:
                v = actionVal[0]
                currentBestAction = a

        return (v, currentBestAction)

    def minValue(self, gameState, agentIndex, depth):
        """
        Return the min value of the successors of gameState.

        :param gameState: The current state of the game.
        :return: The min value of gameState's successors.
        """

        # Initialize v to a max value.
        v = 99999

        # The number of agents in the game.
        numAgents = gameState.getNumAgents()

        # The nextAgentIndex will be agentIndex + 1 (mod numAgents) as we want to iterate each agent
        # once for each level of depth.
        nextAgentIndex = (agentIndex + 1) % numAgents

        # If we are looking at the last agent in the game, decrement the depth by 1.
        if agentIndex == numAgents - 1:
            nextDepth = depth - 1
        # Otherwise, the depth remains the same.
        else:
            nextDepth = depth

        # Return a list of all the legal actions of the agent.
        actions = gameState.getLegalActions(agentIndex)
        # Loop the legal actions and find the min.
        for a in actions:
            successor = gameState.generateSuccessor(agentIndex, a)
            actionVal = self.value(successor, nextAgentIndex, nextDepth)
            if actionVal[0] < v:
                v = actionVal[0]
                currentBestAction = a

        return (v, currentBestAction)
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 4 - optional)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Question 3)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """


        agentIndex = self.index
        depth = self.depth

        return self.value(gameState, agentIndex, depth)[1]

    def value(self, gameState, agentIndex, depth):
        """
        If the gameState is a terminal state, return it. Otherwise, find the min/max and return the minimax value.

        :param gameState: The current state of the game.
        :return: The minimax value at the state gameState.
        """
        # If the current depth is 0, we have traversed the tree; return the value of the evaluation function.
        if depth == 0:
            return (self.evaluationFunction(gameState), '')

        # If gameState is a terminal state (win or loss), return the value of the evaluation function.
        if gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), '')

        # If the agent is Pacman (agentIndex == 0) find the max.
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        # Otherwise, the agent is a Ghost (agentIndex >= 1); find the expected.
        elif agentIndex >= 1:
            return self.expValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        """
        Return the max value of the successors of gameState.

        :param gameState: The current state of the game.
        :return: The max value of gameState's successors.
        """

        # Initialize v to a min value.
        v = -99999

        # The number of agents in the game.
        numAgents = gameState.getNumAgents()

        # The nextAgentIndex will be agentIndex + 1 (mod numAgents) as we want to iterate each agent
        # once for each level of depth.
        nextAgentIndex = (agentIndex + 1) % numAgents

        # If we are looking at the last agent in the game, decrement the depth by 1.
        if agentIndex == numAgents - 1:
            nextDepth = depth - 1
        # Otherwise, the depth remains the same.
        else:
            nextDepth = depth

        # Return a list of all the legal actions of the agent.
        actions = gameState.getLegalActions(agentIndex)

        # Loop the legal actions and find the max.
        for a in actions:
            successor = gameState.generateSuccessor(agentIndex, a)
            actionVal = self.value(successor, nextAgentIndex, nextDepth)
            if actionVal[0] > v:
                v = actionVal[0]
                currentBestAction = a

        return (v, currentBestAction)

    def expValue(self, gameState, agentIndex, depth):
        """
        Return the probable value of the successors of gameState.

        :param gameState: The current state of the game.
        :return: The min value of gameState's successors.
        """

        # Initialize v to a max value.
        v = 0

        # The number of agents in the game.
        numAgents = gameState.getNumAgents()

        # The nextAgentIndex will be agentIndex + 1 (mod numAgents) as we want to iterate each agent
        # once for each level of depth.
        nextAgentIndex = (agentIndex + 1) % numAgents

        # If we are looking at the last agent in the game, decrement the depth by 1.
        if agentIndex == numAgents - 1:
            nextDepth = depth - 1
        # Otherwise, the depth remains the same.
        else:
            nextDepth = depth

        # Return a list of all the legal actions of the agent.
        actions = gameState.getLegalActions(agentIndex)
        currentProbability = 0
        # Loop the legal actions and find the min.
        for a in actions:
            successor = gameState.generateSuccessor(agentIndex, a)
            actionVal = self.value(successor, nextAgentIndex, nextDepth)

            # The probability of the move is 1/(# of legal actions for the agent).
            p = 1.0 / len(gameState.getLegalActions(agentIndex))
            v += p * actionVal[0]

            if v > currentProbability:
                currentProbability = v

        currentBestAction = a

        return (currentProbability, currentBestAction)



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5 - optional).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
 

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

