'''
objects.py
-----------
All objects required to simulate a game of ticket to ride is stored here for use in game.py
'''

from random import Random, randint
from collections import deque

class Route:

    def __init__(self, city1: str, city2: str, weight: int, color: str, index: int) -> None:
        self.city1: str = city1
        self.city2: str = city2
        self.weight: int = weight
        self.color: str = color
        self.id: int = index
    
    def __str__(self) -> str:
        return f"({self.id}) {self.city1} --{self.weight}-- {self.city2} ({self.color})"

class Deck:

    def __init__(self, items: list = [], seed: int = None) -> None:
        if seed == None:
            seed = randint(0, 99999999)
        self.rand = Random(seed)
        self.rand.shuffle(items)
        self.cards = deque(items)
        
    def __str__(self) -> str:
        strings: list[str] = []
        if len(self.cards) == 0:
            return "EMPTY"
        else:
            for card in self.cards:
                strings.append(str(card))
            return '\n'.join(strings)
    
    def shuffle(self, seed) -> None:
        """Shuffle the deck in place"""
        if len(self.cards) == 0:
            return
        shuffled = list(self.cards)
        self.rand.shuffle(shuffled)
        self.cards = deque(shuffled)
    
    def draw(self, num: int) -> list:
        '''Draw a number of cards from the deck'''
        assert len(self.cards) >= num, f"Attempted to draw {num} cards from a deck of {len(self.cards)} cards"
        cardsDrawn: list = []
        for _ in range(num):
            cardsDrawn.append(self.cards.pop())
        return cardsDrawn
    
    def insert(self, cards: list) -> None:
        """
        Place the list of cards into the back of the deck
        """
        assert type(cards) == list, "When inserting cards into Deck, cards must be in a list."
        for card in cards:
            self.cards.appendleft(card)

class Action:
    """
    The object representing an action (to) take/taken in the game
    """
    def __init__(self, action: int, route: Route = None, colorsUsed: list[str] = None, colorToDraw: str = None, askingForDeal: bool = None, takeDests: list[int] = None, faceUpCards = None) -> None:
        """
        Create an action object\n
        @parameters\n
        action = 0 [PLACE ROUTE]\n
            needs route and needs colorsUsed\n
        action = 1 [DRAW FACE UP]\n
            needs colorToDraw\n
        action = 2 [DRAW FACE DOWN]\n
        action = 3 [DRAW DESTINATIONS]\n
            needs askingForDeal or needs takeDests
        """
        self.action = action
        if self.action == 0:
            assert route != None, "Action object: route placement but route object not given"
            assert type(route) == Route, "Action object: route not supplied as type Route"
            assert colorsUsed != None, "Action object: route placement but colors used not given"
            assert len(colorsUsed) > 0, "Action object: route placement but colors used is empty"
            self.route = route
            self.colorsUsed = colorsUsed
        elif self.action == 1:
            assert colorToDraw != None, "Action object: drawing from face up but which index from face up is not supplied"
            assert type(colorToDraw) == str, f"Action object: colorToDraw must be supplied as string but given {colorToDraw}"
            self.colorToDraw = colorToDraw
        elif self.action == 3:
            assert askingForDeal != None, "Action object: destination deal understood but object does not know if deal was given or asked"
            self.askingForDeal = askingForDeal
            if askingForDeal == False:
                assert takeDests != None, "Action object: indicated that destinations have been dealt but no takeDest of indexes is supplied"
                assert type(takeDests) == list, "Action object: takeDests not supplied correctly, must be list of indexes"
                self.takeDests = takeDests
        self.faceUpCards: list[str] = faceUpCards
        assert self.faceUpCards != None
        """A list of the face up cards BEFORE the action described in this object was taken"""
    
    def __str__(self) -> str:
        if self.action == 0:
            return f"({self.faceUpCards}) {self.route} using {self.colorsUsed}"
        elif self.action == 1:
            return f"({self.faceUpCards}) {self.colorToDraw} (DRAW FACE UP)"
        elif self.action == 2:
            return f"({self.faceUpCards}) (DRAW FACE DOWN)"
        elif self.action == 3:
            if self.askingForDeal:
                return f"({self.faceUpCards}) (ASK FOR DEST DEAL)"
            else:
                return f"({self.faceUpCards}) {self.takeDests} (TAKE DESTS)"
    
    def __hash__(self):
        value = str(self.action)
        if self.action == 0:
            value = value + str(self.route.id)
            for color in self.colorsUsed:
                value = value + str(color_indexing[color])
        if self.action == 1:
            value = value + str(color_indexing[self.colorToDraw])
        if self.action == 3:
            if self.askingForDeal:
                value = value + "9"
            else:
                for index in self.takeDests:
                    value = value + str(index)
        return int(value)
    
    def __eq__(self, value):
        if value == None:
            return False
        assert type(value) == Action
        assert 4 > self.action > -1
        assert 4 > value.action > -1
        if self.action == 0 and value.action == 0:
            if self.route.id != value.route.id:
                return False
            if self.colorsUsed != value.colorsUsed:
                return False
            #print(f"Action 0 equal! {self} == {value}")
            return self.checkFaceUps(value)
        elif self.action == 1 and value.action == 1:
            if self.colorToDraw != value.colorToDraw:
                return False
            #print(f"Action 1 equal! {self} == {value}")
            return self.checkFaceUps(value)
        elif self.action == 2 and value.action == 2:
            #print(f"Action 2 equal! {self} == {value}")
            return self.checkFaceUps(value)
        elif self.action == 3 and value.action == 3:
            if self.askingForDeal == True and value.askingForDeal == True:
                return self.checkFaceUps(value)
            if self.askingForDeal == False and value.askingForDeal == False:
                if self.takeDests != value.takeDests:
                    return False
                #print(f"Action 3 equal! {self} == {value}")
                return self.checkFaceUps(value)
            else:
                return False
        else:
            TypeError(f"Action equality function reached an unspecified state when given actions {self} and {value}")
        
    def checkFaceUps(self, value):
        for color in value.faceUpCards:
            if self.faceUpCards.count(color) == value.faceUpCards.count(color):
                return True
            else:
                return False


class Player:

    def __init__(self, name: str) -> None:
        self.points: int = 0
        self.name: str = name
        self.turnsTaken: int = 0
        self.trainsLeft: int = 45
        self.turnOrder: int = None
        self.trainCardHand: list[str] = []
        self.destinationCardHand: list[Destination] = []
        self.colorCounts: list[list[str]] = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        """The known cards for each other player in the game (each color its own index, each index represents the turn order of that player)"""
        self.gotLongestRoute: bool = False

    def __str__(self) -> str:
        destString = []
        for x in self.destinationCardHand:
            destString.append(str(x))
        return f"(PLAYER {self.turnOrder}) {self.name}: {self.points} points, {self.trainsLeft} trains left\nDESTS: [{', '.join(destString)}]\nCOLRS: {self.trainCardHand}\nCOLCNT: {self.colorCounts}"

    

class Destination:

    def __init__(self, city1: str, city2: str, points: int, index: int) -> None:
        self.city1: str = city1
        self.city2: str = city2
        self.points: int = points
        self.id: int = index
    
    def __str__(self) -> str:
        return f"({self.id}) {self.city1} --{self.points}-- {self.city2}"

def actionsAreEqual(action1: Action, action2: Action) -> bool:
    """Returns whether the two actions given in the parameters are equal"""
    assert type(action1) == Action
    assert type(action2) == Action
    assert -1 > action1.action > 4
    assert -1 > action2.action > 4
    if action1.action == 0 and action2.action == 0:
        if action1.route.id != action2.route.id:
            return False
        if action1.colorsUsed != action2.colorsUsed:
            return False
        print(f"Action 0 equal! {action1} == {action2}")
        return True
    elif action1.action == 1 and action2.action == 1:
        if action1.colorToDraw != action2.colorToDraw:
            return False
        print(f"Action 1 equal! {action1} == {action2}")
        return True
    elif action1.action == 2 and action2.action == 2:
        print(f"Action 2 equal! {action1} == {action2}")
        return True
    elif action1.action == 3 and action2.action == 3:
        if action1.askingForDeal == True and action2.askingForDeal == True:
            return True
        if action1.askingForDeal == False and action2.askingForDeal == False:
            if action1.takeDests != action2.takeDests:
                return False
            print(f"Action 3 equal! {action1} == {action2}")
            return True
        else:
            return False
    else:
        TypeError(f"Action equality function reached an unspecified state when given actions {action1} and {action2}")

color_indexing: dict[str, int] = {'PINK': 0, 'WHITE': 1, 'BLUE': 2, 'YELLOW': 3, 'ORANGE': 4, 'BLACK': 5, 'RED': 6, 'GREEN': 7, 'WILD': 8}
"""A dictionary that maps string names to their index values (standardization)"""

pointsByLength: dict[int, int] = {1:1, 2:2, 3:4, 4:7, 5:10, 6:15}
"""A dictionary that maps route length to the points gained for placing it"""

graphColors: dict[int, str] = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange'}
"""A dictionary that maps a player turn order to the color denoting their routes on the graph drawn at the end of the game"""

