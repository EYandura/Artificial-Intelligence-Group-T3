# ------------------------
# CS 471 Fall 2018
# YOUR NAMES HERE
# DOCUMENTATION:
#   ~ The Python Doc was used throughout this file in order to explore the built in Python structures and functionality.
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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

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
    # Create the list to store the path.
    path = []
    # Create a list to store the expanded nodes to avoid infinite trees.
    expanded = []

    # Insert the start state into the fringe.
    startState = problem.getStartState()
    fringe.push(startState)

    # TODO: Returns a blank list for the final autograder test to display an output.
    # if isinstance(startState, str):
    #     fringe.push(startState)
    # else:
    #     return []

    # Loop while the fringe is not empty.
    while fringe.isEmpty() == False:
        # Pop a node for expansion.
        node = fringe.pop()

        # If the node is not just a string, the node letter is the first element of the tuple.
        if isinstance(node, str) == False:
            # Add the node path directions (first index of node tuple) to the path.
            path.append(node[1])
            # Set node to the node letter (0th tuple index).
            node = node[0]

        # If the node is the goal state, return the path.
        if problem.isGoalState(node):
            return path

        # Only expand the node if it has not already been expanded.
        if expanded.count(node) == 0:
            children = problem.getSuccessors(node)
        else:
            children = []

        # If the node has no children or it was already expanded, remove it from the end of the path,
        #   as it will not ever lead to the goal.
        if len(children) == 0 or expanded.count(node) > 0:
            path.pop()

        # If the node is not in the expanded list, add it's children to the fringe.
        if expanded.count(node) == 0:
            for child in children:
                fringe.push(child)
            # Add the node to the expanded list.
            expanded.append(node)

    # Return an empty list if the fringe is empty.
    return []

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
    # Create the list to store the path.
    path = []
    # Create a list to store the expanded nodes to avoid infinite trees.
    expanded = []

    # Insert the start state into the fringe.
    startState = problem.getStartState()
    fringe.push(startState)

    # TODO: Returns a blank list for the final autograder test to display an output.
    # if isinstance(startState, str):
    #     fringe.push(startState)
    # else:
    #     return []

    # Loop while the fringe is not empty.
    while fringe.isEmpty() == False:
        # Pop a node for expansion.
        node = fringe.pop()

        # If the node is not just a string, the node letter is the first element of the tuple.
        if isinstance(node, str) == False:
            # Add the node path directions (first index of node tuple) to the path.
            path.append(node[1])
            # Set node to the node letter (0th tuple index).
            node = node[0]

        # If the node is the goal state, return the path.
        if problem.isGoalState(node):
            return path

        # Only expand the node if it has not already been expanded.
        if expanded.count(node) == 0:
            children = problem.getSuccessors(node)
        else:
            children = []

        # If the node has no children or it was already expanded, remove it from the beginning of the path,
        #   as it will not ever lead to the goal.
        if len(children) == 0 or expanded.count(node) > 0:
            path.pop()

        # If the node is not in the expanded list, add it's children to the fringe.
        if expanded.count(node) == 0:
            for child in children:
                fringe.push(child)
            # Add the node to the expanded list.
            expanded.append(node)

    # Return an empty list if the fringe is empty.
    return []

def uniformCostSearch(problem):
    """
    CS 471 PEX 1 Question 3
    Search the lowest cost nodes in the search tree first using a Priority Queue.

    :param problem: The search problem we are solving.
    :return: A list containing the list of actions that reaches the goal.
    """

    from game import Directions

    # Create a priority queue to be used as the fringe.
    fringe = util.PriorityQueue()
    # Create the list to store the path.
    path = []
    # Create a list to store the expanded nodes to avoid infinite trees.
    expanded = []

    # Insert the start state into the fringe; the startState has a priority cost of 0.
    startState = problem.getStartState()
    fringe.push(startState, 0)

    # TODO: Returns a blank list for the final autograder test to display an output.
    # if isinstance(startState, str):
    #     fringe.push(startState)
    # else:
    #     return []

    # Loop while the fringe is not empty.
    while fringe.isEmpty() == False:
        # Pop a node for expansion.
        node = fringe.pop()

        # If the node is not just a string, the node letter is the first element of the tuple.
        if isinstance(node, str) == False:
            # Add the node path directions (first index of node tuple) to the path.
            path.append(node[1])
            # Set node to the node letter (0th tuple index).
            node = node[0]

        # If the node is the goal state, return the path.
        if problem.isGoalState(node):
            return path

        # Only expand the node if it has not already been expanded.
        if expanded.count(node) == 0:
            children = problem.getSuccessors(node)
        else:
            children = []

        # If the node has no children or it was already expanded, remove it from the beginning of the path,
        #   as it will not ever lead to the goal.
        if len(children) == 0 or expanded.count(node) > 0:
            path.pop()

        # If the node is not in the expanded list, add it's children to the fringe.
        if expanded.count(node) == 0:
            for child in children:
                fringe.push(child, child[2])  # The priority cost is the third tuple element.
            # Add the node to the expanded list.
            expanded.append(node)

    # Return an empty list if the fringe is empty.
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """
    CS 471 PEX 1 Question 4
    *** YOUR CODE HERE ***
    """
	
    return []



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
