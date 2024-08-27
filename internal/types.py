import re
import sys
import networkx as nx
from copy import deepcopy
from random import shuffle
from collections import deque
from matplotlib import pyplot

# ENDED: Creating labels for the neural network in network.py
# TODO: draw() player routes with separate colors
# TODO: any empty variables in the game object must be set to none when not in use.
# TODO: always update the current player object!!
# TODO: verify that game translation to network input is valid
# TODO: test network output to logit conversion!
# TODO: getting logits for an action (is multiplying the best way?)
# TODO: (self.turn - 1) % len(self.players) the way to get the right person to go from self.players?
# TODO: reshuffle train cards when 
# TODO: color counting for each player
# TODO: in MCTS, do terminal states in the simulation have different things to backprop? consult AGZero paper
# TODO: More specific logs? Terminal state logs and who caused the end of the game and such
# TODO: Make sure the player who initiated the last turn actually gets a last turn
# TODO: Learning rate schedule - how should it be adjusted for this implementation? (training.py)
# TODO: Do typecasting and object creating for the bucket in training.py
# TODO: Test gameWinner() function

"""
CLASSES
--------------------------------------------
"""

class Deck:
    """
    The object to represent a deck of cards in Ticket to Ride
    """
    def __init__(self, items: list) -> None:
        """
        Create a new, shuffled deck of cards from a list of items (no typecast for items within)
        """
        shuffle(items)
        self.cards = deque(items)

    def __str__(self) -> str:
        strings: list[str] = []
        for card in self.cards:
            strings.append(str(card))
        return '\n'.join(strings)
    
    def count(self) -> int:
        """
        The number of cards in this deck
        """
        return len(self.cards)
    
    def shuffle(self) -> None:
        """
        Shuffle the deck in place
        """
        assert len(self.cards) > 0, "Can't shuffle an empty deck"
        reShuffled = shuffle(list(self.cards))
        self.cards = deque(reShuffled)
    
    def draw(self, number: int) -> list:
        """
        Draw a number of cards from the deck
        """
        assert len(self.cards) >= number, f"Attempted to draw {number} cards from a deck of {len(self.cards)} cards"
        cardsDrawn: list = []
        for _ in range(number):
            cardsDrawn.append(self.cards.pop())
        return cardsDrawn
    
    def insert(self, cards: list) -> None:
        """
        Place the list of cards into the back of the deck
        """
        assert type(cards) == list, "When inserting cards into Deck, cards must be in a list."
        for card in cards:
            self.cards.appendleft(card)

class Destination:
    """
    The object to represent a destination card in Ticket to Ride
    """
    def __init__(self, 
                 city1: str, 
                 city2: str, 
                 points: int, 
                 index: int
                 ) -> None:
        """
        Create a destination card (index denotes a constant value for this card)
        """
        self.city1: str = city1
        self.city2: str = city2
        self.points: int = points
        self.index: int = index
    
    def __str__(self) -> str:
        return f"({self.index}) {self.city1} --{self.points}-- {self.city2}"

class Route:
    """
    The object to represent a single route between two cities in Ticket to Ride
    """
    def __init__(self, 
                 city1: str, 
                 city2: str, 
                 weight: int,
                 color: str,
                 index: int
                 ) -> None:
        """
        Create a Route object (index denotes unique value for Route)
        """
        self.city1: str = city1
        self.city2: str = city2
        self.weight: int = weight
        self.color: str = color
        self.index: int = index
    
    def __str__(self) -> str:
        return f"({self.index}) {self.city1} --{self.weight}-- {self.city2} ({self.color})"

class Action:
    """
    The object representing an action (to) take/taken in the game
    """
    def __init__(self, action: int, route: Route = None, colorsUsed: list[str] = None, colorToDraw: str = None, askingForDeal: bool = None, takeDests: list[int] = None) -> None:
        """
        Create an action object (0 - Place Route, 1 - Draw Face Up, 2 - Draw Face Down, 3 - Draw Destination Card)

        On initialization, relevant parameters must be supplied depending on which action is being described.
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
            assert type(colorToDraw) == str, "Action object: colorToDraw must be supplied as integer"
            self.colorToDraw = colorToDraw
        elif self.action == 3:
            assert askingForDeal != None, "Action object: destination deal understood but object does not know if deal was given or asked"
            self.askingForDeal = askingForDeal
            if askingForDeal == False:
                assert takeDests != None, "Action object: indicated that destinations have been dealt but no takeDest of indexes is supplied"
                assert type(takeDests) == list, "Action object: takeDests not supplied correctly, must be list of indexes"
                self.takeDests = takeDests
    
    def __str__(self) -> str:
        if self.action == 0:
            return f"{self.route} using {self.colorsUsed}"
        elif self.action == 1:
            return f"{self.colorToDraw} (DRAW FACE UP)"
        elif self.action == 2:
            return f"(DRAW FACE DOWN)"
        elif self.action == 3:
            if self.askingForDeal:
                return f"(ASK FOR DEST DEAL)"
            else:
                return f"{self.takeDests} (TAKE DESTS)"

class Agent:
    """
    The agent object that plays Ticket to Ride
    """
    def __init__(self, name: str) -> None:
        """
        Initialize an agent (give it a cool name!)
        """
        self.points: int = 0
        self.name: str = name
        self.trainsLeft: int = 45
        self.turnOrder: int = None
        self.trainCards: list[str] = []
        self.destinationCards: list[Destination] = []
        self.colorCounts: list[list[str]] = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        """The known cards for each other player in the game (each color its own index)"""
    
    def __str__(self) -> str:
        return f"(PLAYER {self.turnOrder}) {self.name}: {self.points} points, {self.trainsLeft} trains left\n{self.destinationCards}\n{self.trainCards}"

class Game:
    """
    The game & state representation object for Ticket to Ride
    """
    def __init__(self, 
                 map: str, 
                 players: list[Agent],
                 logs: bool,
                 draw: bool
                 ) -> None:
        """
        Initialize a game of Ticket to Ride that is ready to play.
        """
        self.map: str = map
        self.moves: int = 0
        self.doLogs: bool = logs
        self.drawGame: bool = draw
        self.gameOver: bool = False
        self.lastTurn: bool = False
        self.endedGame: Agent = None
        self.colorPicked: str = None
        self.turn = 1 - len(players)
        self.wildPicked: bool = False
        self.gameLogs: list[str] = []
        self.discardPile: Deck = Deck([])
        self.trainCarDeck: Deck = trainCarDeck()
        self.destinationDeal: list[Destination] = None
        self.movePerforming: Action = Action(3, askingForDeal=True)
        self.board: nx.MultiGraph = initBoard(self.map, len(players))
        self.faceUpCards: list[str] = self.trainCarDeck.draw(5)
        self.destinationCards: list[Destination] = getDestinationCards(map)
        self.destinationsDeck: Deck[Destination] = Deck(self.destinationCards)
        self.players: list[Agent] = players if ((2 <= len(players) <= 4)) else sys.exit("ERROR: Game must have 2-4 players")
        for i, player in enumerate(self.players):
            player.turnOrder = i
            player.trainCards = self.trainCarDeck.draw(4)
        self.makingNextMove: Agent = self.players[0]

    def __str__(self) -> str:
        info = f"Turn: {self.turn}\nPlayers: {len(self.players)}\nGame Over: {self.gameOver}\nFinal Turns: {self.lastTurn}\nDestinations Left: {self.destinationsDeck.count()}\nTrain Cards Left: {self.trainCarDeck.count()}\n{self.faceUpCards}\n--------------------------------------------------\nNEXT TO GO: {self.makingNextMove}\n"
        return info
    
    def clone(self):
        """
        Creates a deep copy of the current Game
        """
        new = Game(self.map, deepcopy(self.players), False, False)
        new.turn = deepcopy(self.turn)
        new.moves = deepcopy(self.moves)
        new.board = deepcopy(self.board)
        new.players = deepcopy(self.players)
        new.gameLogs = deepcopy(self.gameLogs)
        new.gameOver = deepcopy(self.gameOver)
        new.lastTurn = deepcopy(self.lastTurn)
        new.endedGame = deepcopy(self.endedGame)
        new.wildPicked = deepcopy(self.wildPicked)
        new.faceUpCards = deepcopy(self.faceUpCards)
        new.colorPicked = deepcopy(self.colorPicked)
        new.trainCarDeck = deepcopy(self.trainCarDeck)
        new.makingNextMove = deepcopy(self.makingNextMove)
        new.movePerforming = deepcopy(self.movePerforming)
        new.destinationDeal = deepcopy(self.destinationDeal)
        new.destinationCards = deepcopy(self.destinationCards)
        new.destinationsDeck = deepcopy(self.destinationsDeck)
        return new
    
    def play(self, action: Action):
        """
        Takes the current game object and applies a given, valid game action to it
        """
        self.moves += 1
        initLastTurn = None
        if self.turn < 1:
            assert action.action == 3, f"TURN {self.turn} Game starts by dealing destination cards, action given: '{action.action}' is invalid for this turn"
            assert action.takeDests != None, f"TURN {self.turn} no destination card indexes (takeDests) specified for pickup"
            destDeal: list[Destination] = self.destinationsDeck.draw(3)
            for index in action.takeDests:
                self.makingNextMove.destinationCards.append(destDeal[index])
                self.makingNextMove.points -= destDeal[index].points
            if len(destDeal) > 0: self.destinationsDeck.insert(destDeal)

            # Logging
            if self.doLogs: 
                self.gameLogs = self.gameLogs + [f"TURN {self.turn} || PLAYER {self.makingNextMove.turnOrder} ({self.makingNextMove.name}) picking {action.takeDests} from [{destDeal[0]}, {destDeal[1]}, {destDeal[2]}]\n", f"         Points: {self.makingNextMove.points}\n", f"         Trains Left: {self.makingNextMove.trainsLeft}\n", f"         Colors Held: {self.makingNextMove.trainCards}\n", "         Dests Held: "]
                for dest in self.makingNextMove.destinationCards:
                    self.gameLogs = self.gameLogs + [f"{dest}, "]
                self.gameLogs = self.gameLogs + ["\n"]
            
            self.turn += 1
            self.makingNextMove = self.players[(self.turn - 1) % len(self.players)]
            if self.turn == 1:
                self.movePerforming = None
        else:
            # NO ACTION - GAME IS OVER
            if action == None:
                self.gameOver = True
                self.endedGame = self.makingNextMove.turnOrder
            # ROUTE PLACING LOGIC
            elif action.action == 0:
                
                # See if the route is already taken, otherwise update game board to reflect new owner of route
                isTaken: bool = None
                placing = None
                for path in self.board.get_edge_data(action.route.city1, action.route.city2).values():
                    if path['owner'] == None: 
                        isTaken = False
                        placing = path
                        path['owner'] = self.makingNextMove.turnOrder
                        break
                if isTaken == None: raise TypeError(f"TURN {self.turn} Placing route that is already owned by player {self.makingNextMove.turnOrder}")

                # Find placement makeup of card colors
                using = []
                colorsGiven = 0
                weight = placing['weight']
                for color in action.colorsUsed:
                    while color in self.makingNextMove.trainCards and colorsGiven < weight:
                        self.makingNextMove.trainCards.remove(color)
                        using.append(color)
                        colorsGiven += 1
                
                # Update player information
                self.makingNextMove.trainsLeft -= weight
                self.makingNextMove.points += pointsByLength[weight]

                # Logging
                if self.doLogs: 
                    self.gameLogs = self.gameLogs + [f"TURN {self.turn} || PLAYER {self.makingNextMove.turnOrder} ({self.makingNextMove.name}) placing {action.route} using {using} action {action.colorsUsed}\n", f"         Points: {self.makingNextMove.points}\n", f"         Trains Left: {self.makingNextMove.trainsLeft}\n", f"         Colors Held: {self.makingNextMove.trainCards}\n", "         Dests Held: "]
                    for dest in self.makingNextMove.destinationCards:
                        self.gameLogs = self.gameLogs + [f"{dest}, "]
                    self.gameLogs = self.gameLogs + ["\n"]

                # Update game variables
                self.discardPile.insert(using)
                if self.makingNextMove.trainsLeft < 3:
                    self.lastTurn = True
                    self.endedGame = self.makingNextMove
                    initLastTurn = True
                self.turn += 1
                self.movePerforming = None
                self.makingNextMove = self.players[(self.turn - 1) % len(self.players)]
            # CARD PICKING LOGIC (FACE UP)
            elif action.action == 1:
                
                if self.movePerforming == None:

                    # Update game state
                    self.faceUpCards.remove(action.colorToDraw)
                    if self.trainCarDeck.count() > 0: 
                        replacement = self.trainCarDeck.draw(1)
                        self.faceUpCards.append(replacement[0])

                    # Update player information
                    self.makingNextMove.trainCards.append(action.colorToDraw)

                    # Logging
                    if self.doLogs: 
                        self.gameLogs = self.gameLogs + [f"TURN {self.turn} || PLAYER {self.makingNextMove.turnOrder} ({self.makingNextMove.name}) picked face up {action.colorToDraw}\n", f"         Points: {self.makingNextMove.points}\n", f"         Trains Left: {self.makingNextMove.trainsLeft}\n", f"         Colors Held: {self.makingNextMove.trainCards}\n", "         Dests Held: "]
                        for dest in self.makingNextMove.destinationCards:
                            self.gameLogs = self.gameLogs + [f"{dest}, "]
                        self.gameLogs = self.gameLogs + ["\n"]

                    # Update game variables
                    if action.colorToDraw == 'WILD':
                        self.turn += 1
                        self.movePerforming = None
                        self.colorPicked = None
                        self.wildPicked = False
                        self.makingNextMove = self.players[(self.turn - 1) % len(self.players)]
                    self.movePerforming = action
                    self.colorPicked = action.colorToDraw
                    self.wildPicked = False
                
                elif self.movePerforming.action == 1 or self.movePerforming.action == 2:
                    assert action.colorToDraw != 'WILD', f"TURN {self.turn} Player {self.makingNextMove.turnOrder} {self.makingNextMove.name} tried to pick up a wild on the second draw"
                    
                    # Update game state
                    self.faceUpCards.remove(action.colorToDraw)
                    if self.trainCarDeck.count() > 0: 
                        replacement = self.trainCarDeck.draw(1)
                        self.faceUpCards.append(replacement[0])

                    # Update player information
                    self.makingNextMove.trainCards.append(action.colorToDraw)

                    # Logging
                    if self.doLogs: 
                        self.gameLogs = self.gameLogs + [f"TURN {self.turn} || PLAYER {self.makingNextMove.turnOrder} ({self.makingNextMove.name}) picked face up {action.colorToDraw}\n", f"         Points: {self.makingNextMove.points}\n", f"         Trains Left: {self.makingNextMove.trainsLeft}\n", f"         Colors Held: {self.makingNextMove.trainCards}\n", "         Dests Held: "]
                        for dest in self.makingNextMove.destinationCards:
                            self.gameLogs = self.gameLogs + [f"{dest}, "]
                        self.gameLogs = self.gameLogs + ["\n"]

                    # Update game variables
                    self.turn += 1
                    self.movePerforming = None
                    self.makingNextMove = self.players[(self.turn - 1) % len(self.players)]
                    self.colorPicked = None
                    self.wildPicked = False

                else:
                    raise TypeError(f"TURN {self.turn} Player {self.makingNextMove.turnOrder} {self.makingNextMove.name} tried to draw face up from invalid state")
            # CARD PICKING LOGIC (FACE DOWN)
            elif action.action == 2:
                
                if self.movePerforming == None:

                    # Update game state
                    drawn = self.trainCarDeck.draw(1)

                    # Update player information
                    self.makingNextMove.trainCards.append(drawn[0])

                    # Logging
                    if self.doLogs: 
                        self.gameLogs = self.gameLogs + [f"TURN {self.turn} || PLAYER {self.makingNextMove.turnOrder} ({self.makingNextMove.name}) picked face down {drawn[0]}\n", f"         Points: {self.makingNextMove.points}\n", f"         Trains Left: {self.makingNextMove.trainsLeft}\n", f"         Colors Held: {self.makingNextMove.trainCards}\n", "         Dests Held: "]
                        for dest in self.makingNextMove.destinationCards:
                            self.gameLogs = self.gameLogs + [f"{dest}, "]
                        self.gameLogs = self.gameLogs + ["\n"]

                    # Update game variables
                    self.movePerforming = action
                    self.colorPicked = drawn[0]

                elif self.movePerforming.action == 1 or self.movePerforming.action == 2:
                    assert self.wildPicked == False, f"TURN {self.turn} Player {self.makingNextMove.turnOrder} {self.makingNextMove.name} tried to pick from face down deck after picking WILD"

                    # Update game state
                    drawn = self.trainCarDeck.draw(1)

                    # Update player information
                    self.makingNextMove.trainCards.append(drawn[0])

                    # Logging
                    if self.doLogs: 
                        self.gameLogs = self.gameLogs + [f"TURN {self.turn} || PLAYER {self.makingNextMove.turnOrder} ({self.makingNextMove.name}) picked face down {drawn[0]}\n", f"         Points: {self.makingNextMove.points}\n", f"         Trains Left: {self.makingNextMove.trainsLeft}\n", f"         Colors Held: {self.makingNextMove.trainCards}\n", "         Dests Held: "]
                        for dest in self.makingNextMove.destinationCards:
                            self.gameLogs = self.gameLogs + [f"{dest}, "]
                        self.gameLogs = self.gameLogs + ["\n"]

                    # Update game variables
                    self.turn += 1
                    self.colorPicked = None
                    self.wildPicked = False
                    self.movePerforming = None
                    self.makingNextMove = self.players[(self.turn - 1) % len(self.players)]
            # DESTINATION PICKING LOGIC
            elif action.action == 3:
                if action.askingForDeal == True:
                    self.destinationDeal: list[Destination] = self.destinationsDeck.draw(3)
                    self.movePerforming = action

                    # Logging
                    if self.doLogs: 
                        self.gameLogs = self.gameLogs + [f"TURN {self.turn} || PLAYER {self.makingNextMove.turnOrder} ({self.makingNextMove.name}) asked for destination deal: [{self.destinationDeal[0]}, {self.destinationDeal[1]}, {self.destinationDeal[2]}]\n", f"         Points: {self.makingNextMove.points}\n", f"         Trains Left: {self.makingNextMove.trainsLeft}\n", f"         Colors Held: {self.makingNextMove.trainCards}\n", "         Dests Held: "]
                        for dest in self.makingNextMove.destinationCards:
                            self.gameLogs = self.gameLogs + [f"{dest}, "]
                        self.gameLogs = self.gameLogs + ["\n"]

                else:
                    # Update game state
                    for index in range(0, 3):
                        if index not in action.takeDests:
                            notTaking = self.destinationDeal[index]
                            self.destinationsDeck.insert([notTaking])
                    
                    # Update player information
                    for index in action.takeDests:
                        taking = self.destinationDeal[index]
                        self.makingNextMove.destinationCards.append(taking)
                        self.makingNextMove.points -= taking.points

                    # Logging
                    if self.doLogs: 
                        self.gameLogs = self.gameLogs + [f"TURN {self.turn} || PLAYER {self.makingNextMove.turnOrder} ({self.makingNextMove.name}) chose {action.takeDests}\n", f"         Points: {self.makingNextMove.points}\n", f"         Trains Left: {self.makingNextMove.trainsLeft}\n", f"         Colors Held: {self.makingNextMove.trainCards}\n", "         Dests Held: "]
                        for dest in self.makingNextMove.destinationCards:
                            self.gameLogs = self.gameLogs + [f"{dest}, "]
                        self.gameLogs = self.gameLogs + ["\n"]

                    # Update game variables
                    self.turn += 1
                    self.movePerforming = None
                    self.destinationDeal = None
                    self.makingNextMove = self.players[(self.turn - 1) % len(self.players)]
        
        # Logging
        if self.doLogs and self.gameOver == False:
            self.gameLogs = self.gameLogs + [f"         Game train deck length = {self.trainCarDeck.count()}\n         Game dests deck length = {self.destinationsDeck.count()}\n\n"]

        if self.lastTurn and self.endedGame.turnOrder == self.makingNextMove.turnOrder and initLastTurn == None:
            self.gameOver = True
            print(f"Game is over turn {self.turn} __ {self.endedGame.turnOrder} ended game and its currently {self.makingNextMove.turnOrder}?")

    def draw(self) -> None:
        """
        Draws the graph representation of the current game using matplotlib
        """
        pos = nx.spectral_layout(self.board)
        nx.draw_networkx_nodes(self.board, pos)
        nx.draw_networkx_labels(self.board, pos, font_size=6)
        for player in self.players:
            edges = [edge for edge in self.board.edges(data=True) if edge[2]['owner'] == player.turnOrder]
            nx.draw_networkx_edges(self.board, pos, edges, edge_color=graphColors[player.turnOrder], connectionstyle=f"arc3, rad = 0.{player.turnOrder}", arrows=True)
        pyplot.show()
    
    def log(self) -> None:
        """
        Logs the game in a log.txt (MAKE SURE LOGS ARE SET TO TRUE AT GAME OBJECT CREATION)
        """
        file = open("log.txt", "w")
        file.writelines(self.gameLogs)

    def validMoves(self) -> list[Action]:
        """
        Returns a list of valid Action objects that denote the actions that can be taken form the current game state. Returns an empty list if there are no valid actions to take (i.e. game is over)
        """
        actionList: list[Action] = []
        
        if self.gameOver: return actionList
        else:
            if self.movePerforming == None:
                numWilds: int = self.makingNextMove.trainCards.count("WILD")
                for route in self.board.edges(data=True):
                    
                    if route[2]['owner'] != None: continue
                    weight: int = int(route[2]['weight'])
                    color: str = route[2]['color']
                    routeType: Route = Route(route[0], route[1], route[2]['weight'], route[2]['color'], route[2]['index'])
                    if self.makingNextMove.trainsLeft < weight: continue
                    
                    if color != "GRAY":

                        numColor: int = self.makingNextMove.trainCards.count(color)
                        if numColor == 0: continue
                        elif numColor < weight:
                            if numWilds > 0:
                                if numWilds + numColor == weight: actionList.append(Action(0, routeType, [color, 'WILD']))
                                elif numWilds + numColor > weight:
                                    actionList.append(Action(0, routeType, [color, 'WILD']))
                                    if numWilds < weight:
                                        actionList.append(Action(0, routeType, ['WILD', color]))
                        elif numColor >= weight:
                            actionList.append(Action(0, routeType, [color]))
                            if 0 < numWilds < weight:
                                actionList.append(Action(0, routeType, ['WILD', color]))
                    else:
                        for color in color_indexing.keys():
                            if color == "WILD": continue
                            numColor: int = self.makingNextMove.trainCards.count(color)
                            if numColor == 0: continue
                            elif numColor < weight:
                                if numWilds > 0:
                                    if numWilds + numColor == weight: actionList.append(Action(0, routeType, [color, 'WILD']))
                                    if numWilds + numColor > weight:
                                        actionList.append(Action(0, routeType, [color, 'WILD']))
                                        if numWilds < weight:
                                            actionList.append(Action(0, routeType, ['WILD', color]))
                                else: continue
                            elif numColor >= weight:
                                actionList.append(Action(0, routeType, [color]))
                                if 0 < numWilds < weight:
                                    actionList.append(Action(0, routeType, ['WILD', color]))
                        if numWilds >= weight:
                            actionList.append(Action(0, routeType, ['WILD']))
                if len(self.faceUpCards) > 1:
                    for card in self.faceUpCards:
                        actionList.append(Action(1, colorToDraw=card))
                if self.trainCarDeck.count() > 0:
                    actionList.append(Action(2))
                if self.destinationsDeck.count() >= 3:
                    actionList.append(Action(3, askingForDeal=True))
            elif self.movePerforming.action == 1:
                for card in self.faceUpCards:
                    if card == "WILD": continue
                    else: actionList.append(Action(1, colorToDraw=card))
                if self.trainCarDeck.count() >= 1: actionList.append(Action(2))
            elif self.movePerforming.action == 2:
                for card in self.faceUpCards:
                    if card == "WILD": continue
                    actionList.append(Action(1, colorToDraw=card))
                if self.trainCarDeck.count() >= 1: actionList.append(Action(2))
            elif self.movePerforming.action == 3:
                for take in listDestTakes():
                    actionList.append(Action(3, askingForDeal=False, takeDests=take))

        if len(actionList) == 0:
            self.gameOver = True
            self.endedGame = self.makingNextMove.turnOrder
        
        return actionList
    
    def getWinner(self) -> Agent:
        """
        Returns the current leading player of the game
        """
        return self.players[max([player.turnOrder for player in self.players])]

class Node:
    """
    An object representing a node storing a game state in MCTS
    """
    def __init__(self, priorProb: float) -> None:
        self.visits: int = 0                    # N(s,a)
        self.totalWinProb: float = 0            # W(s,a)
        self.priorProb: float = priorProb       # P(s,a)
        self.children: dict[Action, Node] = {}  
        self.toPlay: int = None
        self.terminal = False
    
    def isExpanded(self) -> bool:
        """
        Returns true if a node has been expanded already (has children)
        """
        return len(self.children) > 0
    
    def value(self) -> float:
        """
        Get the mean action value - Q(s,a) - of the node
        """
        if self.visits == 0:
            return 0
        return self.totalWinProb / self.visits

"""
FUNCTIONS
--------------------------------------------
"""

def getDestinationCards(map: str) -> list[Destination]:
    """
    Takes a map name and returns a list of paths between cities where each item is a Destination object
    """
    lines = open(f"internal/{map}_destinations.txt").readlines()
    cards: list[Destination] = []
    index = 0
    for card in lines:
        data = re.search('(^\D+)(\d+)\s(.+)', card)
        cards.append(Destination(data.group(1).strip(), data.group(3).strip(), int(data.group(2).strip()), index))
        index += 1
    return cards

def getRoutes(map: str) -> list[Route]:
    """
    Takes a map name and returns a list of paths between cities where each item is a Route object
    """
    lines = open(f"internal/{map}_paths.txt").readlines()
    paths = []
    index = 0
    for path in lines:
        data = re.search('(^\D+)(\d)\W+(\w+)\W+(.+)', path)
        paths.append(Route(data.group(1).strip(), data.group(4).strip(), int(data.group(2).strip()), data.group(3).strip(), index))
        index += 1
    return paths

def trainCarDeck() -> Deck:
    """
    Builds the standard train car deck of 110 cards
    """
    deck = ['PINK']*12+['WHITE']*12+['BLUE']*12+['YELLOW']*12+['ORANGE']*12+['BLACK']*12+['RED']*12+['GREEN']*12+['WILD']*14
    return Deck(deck)

def initBoard(map: str, players: int) -> nx.MultiGraph:
    """
    Creates a networkx MultiGraph representation of the game board given a map name
    """
    board = nx.MultiGraph()
    if players == 4:
        board.add_edges_from((route.city1, route.city2, {'weight': route.weight, 'color': route.color, 'owner': None, 'index': route.index}) for route in getRoutes(map))
    else:
        board.add_edges_from((route.city1, route.city2, {'weight': route.weight, 'color': route.color, 'owner': None, 'index': route.index}) for route in getRoutes(map) if board.has_edge(route.city1, route.city2) == False)
    return board

def listDestTakes() -> list[list[int]]:
    return [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]

"""
VARIABLES
--------------------------------------------
"""

color_indexing: dict[str, int] = {'PINK': 0, 'WHITE': 1, 'BLUE': 2, 'YELLOW': 3, 'ORANGE': 4, 'BLACK': 5, 'RED': 6, 'GREEN': 7, 'WILD': 8}
"""A dictionary that maps string names to their index values (standardization)"""

pointsByLength: dict[int, int] = {1:1, 2:2, 3:4, 4:7, 5:10, 6:15}
"""A dictionary that maps route length to the points gained for placing it"""

graphColors: dict[int, str] = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}
"""A dictionary that maps a player turn order to the color denoting their routes on the graph drawn at the end of the game"""