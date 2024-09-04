'''
data.py
----------------------
This file is meant to hold all utility functions that help build this model in python

09/03/2024 > Checked and validated
'''

import re

colors = {0: 'red', 1: 'blue', 2: 'orange', 3: 'green'}
'''
A storage variable where an integer key returns a corresponding color
0 - red
1 - blue
2 - orange
3 - green
'''

pointsByLength = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}
'''
A storage variable where an integer key of length corresponds to a point value for claiming that route
1 - 1 point
2 - 2 point
3 - 4 point
4 - 7 point
5 - 10 point
6 - 15 point
'''

indexByColor = {'PINK': 0, 'WHITE': 1, 'BLUE': 2, 'YELLOW': 3, 'ORANGE': 4, 'BLACK': 5, 'RED': 6, 'GREEN': 7, 'WILD': 8}
'''
A storage variable where a string key of a card color corresponds to an integer value
'''

def getPaths(map: str) -> list[list[str]]:
    """
    Takes a map name and returns a list of paths between cities where each item is an array [city1, length, color, city2] as strings
    """
    lines = open(f"engine/{map}_paths.txt").readlines()
    paths = []
    i = 0
    for path in lines:
        data = re.search('(^\D+)(\d)\W+(\w+)\W+(.+)', path)
        paths.append([data.group(1).strip(), data.group(2).strip(), data.group(3).strip(), data.group(4).strip(), i])
        i += 1
    return paths

def getPathsAM(map: str) -> list[list[str]]:
    """
    Takes a map name and returns a list of paths between cities where each item is an array [city1, length, color, city2] as strings including the edge data (for action map)
    """
    lines = open(f"engine/{map}_paths.txt").readlines()
    paths = []
    i = 0
    for path in lines:
        data = re.search('(^\D+)(\d)\W+(\w+)\W+(.+)', path)
        weight = int(data.group(2).strip())
        paths.append([data.group(1).strip(), data.group(4).strip(), 0, {'weight': weight, 'color': data.group(3).strip(), 'owner': '', 'index': i}])
        i += 1
    return paths

def getDestinationCards(map: str) -> list[list[str]]:
    """
    Takes a map name and returns a list of paths between cities where each item is an array [city1, points, city2] as strings
    """
    lines = open(f"engine/{map}_destinations.txt").readlines()
    cards = []
    i = 0
    for card in lines:
        data = re.search('(^\D+)(\d+)\s(.+)', card)
        cards.append([data.group(1).strip(), data.group(2).strip(), data.group(3).strip(), i])
        i += 1
    
    return cards

def listColors() -> list[str]:
    '''
    A utility function that returns all possible colors of a train car card in order of the engine indexing
    '''
    return ['PINK', 'WHITE', 'BLUE', 'YELLOW', 'ORANGE', 'BLACK', 'RED', 'GREEN', 'WILD']

def listDestTakes() -> list[list[int]]:
    '''
    A utility function meant to display all possibilities of destination card takes when a new draw is requested
    
    It returns an array of arrays, where each array is a list of integers that represents the index of which card to pick.
    [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    '''
    return [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]

def product(*args: int) -> float:
    '''
    Takes an arbitrary number of integers and returns the product.
    '''
    p = 1
    for arg in args:
        p *= arg
    return p