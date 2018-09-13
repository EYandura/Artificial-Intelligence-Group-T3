# ------------------------
# CS 471 Fall 2018
# C1C Reece Clingenpeel, C1C Eric Yandura
# DOCUMENTATION:
#   ~ The Python Doc was used throughout this file in order to explore the built in Python structures and functionality.
#   ~ C1C Curran Brandt helped me to conceptually understand the general method for implementing the DFS.
# -----------------------

# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


# ======================================================================================
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


# ======================================================================================
def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


# --------------------------------------------------------------------------------------
def depthFirstSearch(problem):
    """
    CS471 PEX 1 Question 1
    Search the deepest nodes in the search tree first using a LIFO Stack.

    :param problem: The search problem we are solving.
    :return: A list containing the list of actions that reaches the goal.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:" + problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    from game import Directions

    # Create a stack to be used as the fringe.
    fringe = util.Stack()
    # Create a list to store the expanded nodes to avoid infinite trees.
    expanded = []

    # Insert the start state into the fringe as a node in the form (currentState, [path])
    startState = problem.getStartState()
    fringe.push((startState, []))

    # Loop while the fringe is not empty.
    while fringe.isEmpty() == False:
        # Pop the first node from the fringe.
        node = fringe.pop()

        # Variables for ease of reading.
        currentState = node[0]
        path = node[1]

        # If the currentState is a goal node, return the path of that node.
        if problem.isGoalState(currentState):
            return path

        # If the current node is not in expanded. add the currentState to expanded and expand the node.
        if expanded.count(currentState) == 0:
            # Add the currentState to expanded and expand the node into children.
            expanded.append(currentState)
            children = problem.getSuccessors(currentState)

            # Push each child into the fringe as a node.
            # The path is the current path with the second element of the child tuple appended to it.
            for child in children:
                newState = child[0]
                newPath = path + [child[1]]

                fringe.push((newState, newPath))

    # Return an empty list if the fringe is empty.
    return []


# --------------------------------------------------------------------------------------
def breadthFirstSearch(problem):
    """
    CS 471 PEX 1 Question 2
    Search the shallowest nodes in the search tree first using a FIFO Queue.

    :param problem: The search problem we are solving.
    :return: A list containing the list of actions that reaches the goal.
    """

    from game import Directions

    # Create a queue to be used as the fringe.
    fringe = util.Queue()
    # Create a list to store the expanded nodes to avoid infinite trees.
    expanded = []

    # Insert the start state into the fringe as a node in the form (currentState, [path])
    startState = problem.getStartState()
    fringe.push((startState, []))

    # Loop while the fringe is not empty.
    while fringe.isEmpty() == False:
        # Pop the first node from the fringe.
        node = fringe.pop()

        # Variables for ease of reading.
        currentState = node[0]
        path = node[1]

        # If the currentState is a goal node, return the path of that node.
        if problem.isGoalState(currentState):
            return path

        # If the current node is not in expanded. add the currentState to expanded and expand the node.
        if expanded.count(currentState) == 0:
            # Add the currentState to expanded and expand the node into children.
            expanded.append(currentState)
            children = problem.getSuccessors(currentState)

            # Push each child into the fringe as a node.
            # The path is the current path with the second element of the child tuple appended to it.
            for child in children:
                newState = child[0]
                newPath = path + [child[1]]

                fringe.push((newState, newPath))

    # Return an empty list if the fringe is empty.
    return []


# --------------------------------------------------------------------------------------
def uniformCostSearch(problem):
    """
    CS 471 PEX 1 Question 3
    Search the lowest cost nodes in the search tree first using a Priority Queue.

    :param problem: The search problem we are solving.
    :return: A list containing the list of actions that reaches the goal.
    """

    # Create a priority queue to be used as the fringe.
    fringe = util.PriorityQueue()
    # Create a list to store the expanded nodes to avoid infinite trees.
    expanded = []

    # Insert the start state into the fringe as a node in the form (currentState, [path])
    startState = problem.getStartState()
    fringe.push((startState, [], 0), 0)  # The totalCost of the start state is 0.

    # Loop while the fringe is not empty.
    while fringe.isEmpty() == False:
        # Pop the first node from the fringe.
        node = fringe.pop()

        # Variables for ease of reading.
        currentState = node[0]
        path = node[1]

        # If the currentState is a goal node, return the path of that node.
        if problem.isGoalState(currentState):
            return path

        # If the current node is not in expanded. add the currentState to expanded and expand the node.
        if expanded.count(currentState) == 0:
            # Add the currentState to expanded and expand the node into children.
            expanded.append(currentState)
            children = problem.getSuccessors(currentState)

            # Push each child into the fringe as a node.
            # The path is the current path with the second element of the child tuple appended to it.
            for child in children:
                newState = child[0]
                newPath = path + [child[1]]
                cost = node[2] + child[2]

                fringe.push((newState, newPath, cost), cost) # backwards plus heuristic

    # Return an empty list if the fringe is empty.
    return []


# --------------------------------------------------------------------------------------
def nullHeuristic(state, problem=None):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    return 0


# --------------------------------------------------------------------------------------
def aStarSearch(problem, heuristic=nullHeuristic):
    """
    CS 471 PEX 1 Question 4
    Search the lowest estimated total cost nodes in the search tree first using a Priority Queue.

    :param problem: The search problem we are solving.
    :return: A list containing the list of actions that reaches the goal.
    """
    # Create a priority queue to be used as the fringe.
    fringe = util.PriorityQueue()
    # Create a list to store the expanded nodes to avoid infinite trees.
    expanded = []

    # Insert the start state into the fringe as a node in the form (currentState, [path])
    startState = problem.getStartState()
    fringe.push((startState, [], 0), 0)  # The totalCost of the start state is 0.

    # Loop while the fringe is not empty.
    while fringe.isEmpty() == False:
        # Pop the first node from the fringe.
        node = fringe.pop()

        # Variables for ease of reading.
        currentState = node[0]
        path = node[1]

        # If the currentState is a goal node, return the path of that node.
        if problem.isGoalState(currentState):
            return path

        # If the current node is not in expanded. add the currentState to expanded and expand the node.
        if expanded.count(currentState) == 0:
            # Add the currentState to expanded and expand the node into children.
            expanded.append(currentState)
            children = problem.getSuccessors(currentState)

            # Push each child into the fringe as a node.
            # The path is the current path with the second element of the child tuple appended to it.
            for child in children:
                newState = child[0]
                newPath = path + [child[1]]
                cost = node[2] + child[2]
                heuristic_cost = heuristic(child[0], problem) + cost

                fringe.push((newState, newPath, cost), heuristic_cost)  # backwards plus heuristic

    return []



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
