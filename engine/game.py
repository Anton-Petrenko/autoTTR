'''
game.py
---------
Game engine is located here
'''

import re, sys
import networkx as nx
import matplotlib.pyplot
from random import Random, randint
from copy import deepcopy
from itertools import combinations
from engine.objects import Deck, Action, Player, Destination, Route, color_indexing, pointsByLength, graphColors
import numpy as np

class Game:
    '''
    The game engine for Ticket to Ride in Python
    '''
    def __init__(self, players: list[Player] = [], map: str = 'USA', logs: bool = False, visualize: bool = False, copy: bool = False, reshuffleLimit: int = 10, seed: int = None) -> None:
        '''
        Initialize a game of Ticket to Ride & deal destination card choices to each player
        '''

        # Create random instance for this game object
        self.seed = seed
        if seed == None: self.seed = randint(0, 99999999)

        # Game variables
        self.turn = 0
        self.states = []
        self.map: str = map
        self.copy: bool = copy
        self.noValidMovesInARow = 0
        self.lastRoundTurn: int = 0
        self.lastRound: bool = False
        self.firstRound: bool = True
        self.recordLogs: bool = logs
        self.gameIsOver: bool = False
        self.gameLogs: list[str] = []
        self.history: list[Action] = []
        self.totalActionsTaken: int = 0
        self.reshuffleLimit = reshuffleLimit
        self.visualizeGame: bool = visualize
        self.lastAction: Action | None = None
        self.finalStandings: list[Player] = []
        self.currentAction: Action | None = None
        self.causedLastTurn: Player | None = None
        self.destinationCardsDealt: list[Destination] | None = None
        self.childVisits = []
        self.players = players if ((2 <= len(players) <= 4)) else sys.exit("ERROR: Game must have 2-4 players")

        # Build game objects
        self.discardPile: Deck = Deck()
        self.trainCarDeck: Deck = self.makeTrainCarDeck() if not copy else Deck([])
        self.faceUpCards: list[str] = self.trainCarDeck.draw(5) if not copy else []
        self.board: nx.MultiGraph = self.makeBoard(self.map, len(players)) if not copy else None
        self.destinationCards: list[Destination] = self.makeDestinationCards(map) if not copy else []
        self.destinationsDeck: Deck[Destination] = Deck(self.destinationCards) if not copy else Deck([])
        if not copy:
            for i, player in enumerate(self.players):
                player.turnOrder = i

        # Save init state
        self.init_trainCarDeck: Deck | None = self.trainCarDeck if not copy else None
        self.init_faceUpCards: list[str] | None = self.faceUpCards if not copy else None
        self.init_destinationsDeck: Deck[Destination] | None = self.destinationsDeck if not copy else None

        # Begin the game
        if not copy:
            self.checkFaceUpCards()
        self.destinationCardsDealt = self.destinationsDeck.draw(3) if not copy else []
        self.init_destinationCardsDealt = self.destinationCardsDealt if not copy else None

    def __str__(self) -> str:
        dests = []
        for x in self.destinationsDeck.cards:
            dests.append(str(x))
        return f"FIRSTROUND: {self.firstRound}\nLASTROUND: {self.lastRound}\nDESTS ({len(self.destinationsDeck.cards)}): {', '.join(dests)}\nCLRDECK ({len(self.trainCarDeck.cards)}): {', '.join(self.trainCarDeck.cards)}\nUPCOLOR: {self.faceUpCards}\nDISCARD: {len(self.discardPile.cards)}\n------------------------------------------------"

    def clone(self):
        """Creates an entirely new copy of the current game"""
        playersNew = deepcopy(self.players)
        new = Game(playersNew, self.map, True, False, True, self.reshuffleLimit, seed=self.seed)
        new.turn = deepcopy(self.turn)
        new.lastRound = deepcopy(self.lastRound)
        new.firstRound = deepcopy(self.firstRound)
        new.recordLogs = deepcopy(self.recordLogs)
        new.gameIsOver = deepcopy(self.gameIsOver)
        new.gameLogs = deepcopy(self.gameLogs) if self.recordLogs else []
        new.totalActionsTaken = deepcopy(self.totalActionsTaken)
        new.visualizeGame = False
        new.lastAction = deepcopy(self.lastAction)
        new.currentAction = deepcopy(self.currentAction)
        new.causedLastTurn = deepcopy(self.causedLastTurn)
        new.destinationCardsDealt = deepcopy(self.destinationCardsDealt)
        new.discardPile = deepcopy(self.discardPile)
        new.trainCarDeck = deepcopy(self.trainCarDeck)
        new.faceUpCards = deepcopy(self.faceUpCards)
        new.board = deepcopy(self.board)
        new.destinationCards = deepcopy(self.destinationCards)
        new.destinationsDeck = deepcopy(self.destinationsDeck)
        return new

    def getRoutes(self, map: str) -> list[Route]:
        '''
        Get a list of all available & unavailable routes for the game
        '''
        lines = open(f"datafiles/{map}_paths.txt").readlines()
        paths = []
        index = 0
        for path in lines:
            data = re.search('(^\D+)(\d)\W+(\w+)\W+(.+)', path)
            paths.append(Route(data.group(1).strip(), data.group(4).strip(), int(data.group(2).strip()), data.group(3).strip(), index))
            index += 1
        return paths

    def makeBoard(self, map: str, numPlayers: int) -> nx.MultiGraph:
        '''
        Build the game board for the given map
        '''
        board = nx.MultiGraph()
        board.add_edges_from((route.city1, route.city2, {'weight': route.weight, 'color': route.color, 'owner': None, 'index': route.id}) for route in self.getRoutes(map))
        return board
        
    def makeTrainCarDeck(self) -> Deck:
        '''
        Build the train car deck
        '''
        deck = ['PINK']*12+['WHITE']*12+['BLUE']*12+['YELLOW']*12+['ORANGE']*12+['BLACK']*12+['RED']*12+['GREEN']*12+['WILD']*14
        return Deck(deck)

    def makeDestinationCards(self, map: str) -> list[Destination]:
        '''
        Build the destination card deck for the given map
        '''
        lines = open(f"datafiles/{map}_destinations.txt").readlines()
        cards: list[Destination] = []
        index = 0
        for card in lines:
            data = re.search('(^\D+)(\d+)\s(.+)', card)
            cards.append(Destination(data.group(1).strip(), data.group(3).strip(), int(data.group(2).strip()), index))
            index += 1
        return cards
    
    def getValidMoves(self) -> list[Action]:
        '''Returns a list of Action objects given the current state the parent game object is in'''

        validMoves: list[Action] = []
        
        if self.gameIsOver:
            return validMoves
        
        if self.firstRound:
            x = len(self.destinationCardsDealt)
            assert 0 < x < 4, f"Destination cards were not dealt on turn {self.turn}"
            helper = list(range(0, x))
            for y in range(2, x+1):
                for z in combinations(helper, y):
                    take = list(z)
                    validMoves.append(Action(3, askingForDeal=False, takeDests=take))

        elif self.lastAction == None:

            nextPlayer: int = self.turn % len(self.players)
            numWilds: int = self.players[nextPlayer].trainCardHand.count("WILD")

            # All route placements
            for route in self.board.edges(data=True):
                if route[2]['owner'] != None:
                    continue
                
                claimable = True
                for edge in self.board.get_edge_data(route[0], route[1]).values():
                    if edge['owner'] == nextPlayer:
                        claimable = False
                        continue
                if not claimable:
                    continue

                if len(self.players) < 4:
                    flag = False
                    for edge in self.board.get_edge_data(route[0], route[1]).values():
                        if edge['owner'] != None:
                            flag = True
                    if flag:
                        continue
                
                weight = int(route[2]['weight'])
                if self.players[nextPlayer].trainsLeft <= weight: 
                    continue

                color: str = route[2]['color']
                route = Route(route[0], route[1], route[2]['weight'], route[2]['color'], route[2]['index'])
        
                # Handle [WILD]
                #if route.city1 == "RALEIGH" and route.city2 == "ATLANTA":
                    #print(f"\n{route}")
                if numWilds >= weight:
                    validMoves.append(Action(0, route, ['WILD']))
                
                # Handle [COLOR]
                if self.players[nextPlayer].trainCardHand.count(color) >= weight:
                    validMoves.append(Action(0, route, [color]))
                
                if color == "GRAY":
                    for loopColor in color_indexing.keys():

                        if loopColor == "WILD":
                            continue

                        numLoopColor = self.players[nextPlayer].trainCardHand.count(loopColor)
                        if numLoopColor + numWilds < weight:
                            continue
                        if numLoopColor >= weight:
                            validMoves.append(Action(0, route, [loopColor]))                       
                        if 0 < numLoopColor < weight and numLoopColor + numWilds > weight:
                            validMoves.append(Action(0, route, [loopColor, 'WILD']))                       
                        if 0 < numWilds < weight and numWilds + numLoopColor > weight:
                            validMoves.append(Action(0, route, ['WILD', loopColor]))                       
                        if numWilds + numLoopColor == weight and numWilds > 0:
                            validMoves.append(Action(0, route, [loopColor, 'WILD']))
                
                else:

                    numColor = self.players[nextPlayer].trainCardHand.count(color)
                    if numColor + numWilds < weight:
                        continue
                    if numColor >= weight:
                        validMoves.append(Action(0, route, [color]))                   
                    if 0 < numColor < weight and numColor + numWilds > weight:
                        validMoves.append(Action(0, route, [color, 'WILD']))                        
                    if 0 < numWilds < weight and numWilds + numColor > weight:
                        validMoves.append(Action(0, route, ['WILD', color]))                        
                    if numWilds + numColor == weight:
                        validMoves.append(Action(0, route, [color, 'WILD']))

            if len(self.faceUpCards) > 0:

                for card in self.faceUpCards:
                    validMoves.append(Action(1, colorToDraw=card))
            
            if len(self.trainCarDeck.cards) > 0:
                validMoves.append(Action(2))
            
            if len(self.destinationsDeck.cards) > 0:
                validMoves.append(Action(3, askingForDeal=True))

            #print()

        elif self.lastAction.action == 1 or self.lastAction.action == 2:

            for card in self.faceUpCards:
                if card != 'WILD':
                    validMoves.append(Action(1, colorToDraw=card))

            if len(self.trainCarDeck.cards) > 0:
                validMoves.append(Action(2))
        
        elif self.lastAction.action == 3:

            x = len(self.destinationCardsDealt)
            helper = list(range(0, x))
            for y in range(1, x+1):
                for z in combinations(helper, y):
                    take = list(z)
                    validMoves.append(Action(3, askingForDeal=False, takeDests=take))
        if len(validMoves) == 0:
            self.noValidMovesInARow += 1
            if self.recordLogs:
                #self.gameLogs = self.gameLogs + [f"\n Game is over... no valid moves remaining"]
                self.gameLogs = self.gameLogs + [f"\n > No valid moves remaining for this player\n"]
            if len(self.players) == self.noValidMovesInARow:
                self.gameLogs = self.gameLogs + [f"\n Game is over... no valid moves remaining\n"]
                self.endGame()
            else:
                self.turn += 1
                self.lastAction = None
            #self.endGame()
        else:
            self.noValidMovesInARow = 0
        
        return validMoves
    
    def play(self, action: Action | None) -> None:
        """
        Takes the current game object and applies a given, valid game action to it
        """

        if self.lastRound == True and self.turn == self.lastRoundTurn + 5:
            self.endGame()
            return

        if self.recordLogs and not self.lastAction:
                self.gameLogs = self.gameLogs + [f"-------------------- TURN {self.turn} --------------------\n{self}\n{self.players[self.turn % len(self.players)]}\n\n"]

        if not self.copy:
            self.states.append(self.stateToInput())

        if self.firstRound:
            assert action.action == 3, f"TURN {self.turn} Game starts by dealing destination cards, action given: '{action.action}' is invalid for this turn"
            assert action.takeDests != None, f"TURN {self.turn} no destination card indexes (takeDests) specified for pickup"
            assert len(self.destinationCardsDealt) >= len(action.takeDests), f"TURN {self.turn} wanted to take cards {action.takeDests} in deal {self.destinationCardsDealt}"

            if self.recordLogs:
                self.gameLogs = self.gameLogs + [f"> dealt [{self.destinationCardsDealt[0]}, {self.destinationCardsDealt[1]}, {self.destinationCardsDealt[2]}]\n> taking {action.takeDests}\n"]

            for index in action.takeDests:
                toTake = self.destinationCardsDealt[index]
                self.players[self.turn % len(self.players)].destinationCardHand.append(toTake)
                self.players[self.turn % len(self.players)].points -= toTake.points
            
            i = 0
            for index in action.takeDests:
                self.destinationCardsDealt.remove(self.destinationCardsDealt[index-i])
                i += 1
            
            self.destinationsDeck.insert(self.destinationCardsDealt)

            if self.turn < (len(self.players) - 1):
                self.destinationCardsDealt = self.destinationsDeck.draw(3)
            else:
                self.destinationCardsDealt = None
                self.firstRound = False
            
            if self.recordLogs:
                self.gameLogs = self.gameLogs + [f"\n{self.players[self.turn % len(self.players)]}\n\n"]

            self.lastAction = None
            self.turn += 1

        elif action == None:
            self.log()
            raise TypeError("Action of value None was given to play")  
        elif action.action == 0:
            self.placeRoute(action.route.city1, action.route.city2, action.colorsUsed)
        elif action.action == 1:
            self.pickFaceUp(action)
        elif action.action == 2:
            self.pickFaceDown(action)
        elif action.action == 3:
            if action.askingForDeal == True:
                assert self.lastAction == None
                self.dealDestinations(action)
            else:
                assert self.lastAction != None
                self.pickDestinations(action.takeDests)

        self.history.append(action)
        self.totalActionsTaken += 1
        self.checkFaceUpCards()
    
    def pickDestinations(self, takeIndexes: list[int]) -> None:
        """
        The function to handle the taking of all destination cards of the player's choice
        """

        if self.recordLogs:
            deal = []
            for x in self.destinationCardsDealt:
                deal.append(str(x))
            self.gameLogs = self.gameLogs + [f"> deal is {', '.join(deal)}\n"]
            self.gameLogs = self.gameLogs + [f"> taking {takeIndexes}\n"]

        for index in takeIndexes:
            toTake = self.destinationCardsDealt[index]
            self.players[self.turn % len(self.players)].destinationCardHand.append(toTake)
            self.players[self.turn % len(self.players)].points -= toTake.points
        
        i = 0
        for index in takeIndexes:
            self.destinationCardsDealt.remove(self.destinationCardsDealt[index-i])
            i += 1

        if self.recordLogs:
            self.gameLogs = self.gameLogs + [f"\n{self.players[self.turn % len(self.players)]}\n\n"]
        
        self.lastAction = None
        self.turn += 1
        
        self.destinationsDeck.insert(self.destinationCardsDealt)

    def dealDestinations(self, action: Action) -> None:
        """
        The function to handle dealing out the destination cards to the next player for choosing
        """
        self.destinationCardsDealt: list[Destination] = []
        if self.recordLogs:
            self.gameLogs = self.gameLogs + [f"> asking for deal...\n"]
        while len(self.destinationCardsDealt) < 3 and len(self.destinationsDeck.cards) > 0:
            draw = self.destinationsDeck.draw(1)[0]
            self.destinationCardsDealt.append(draw)
        if len(self.destinationCardsDealt) == 0:
            print("Tried dealing destinations but no more left...code should NEVER REACH THIS")
        self.lastAction = action

    def pickFaceDown(self, action: Action) -> None:
        """
        The function for changing the game state to reflect a take of a face down card
        """
        drawn = self.trainCarDeck.draw(1)[0]
        self.players[self.turn % len(self.players)].trainCardHand.append(drawn)

        if self.recordLogs:
            self.gameLogs = self.gameLogs + [f"> picking face down\n> it's a {drawn}!\n"]

        
        
        if self.lastAction != None:
            self.lastAction = None
            self.turn += 1
            if self.recordLogs:
                self.gameLogs = self.gameLogs + [f"\n{self.players[self.turn % len(self.players)]}\n\n"]
        else:
            self.lastAction = action
    
    def checkFaceUpCards(self) -> None:

        if len(self.trainCarDeck.cards) == 0 and len(self.discardPile.cards) > 0:
            if self.recordLogs:
                self.gameLogs = self.gameLogs + ["\n> SHUFFLING DISCARDS INTO FACE DOWN DECK\n"]
            self.discardPile.shuffle(self.seed)
            self.trainCarDeck = self.discardPile
            self.discardPile = Deck([])

        elif self.faceUpCards.count("WILD") >= 3:
            if self.recordLogs:
                self.gameLogs = self.gameLogs + ["\n> RESHUFFLE OF FACE UP CARDS TRIGGERED... \n"]
            self.discardPile.insert(self.faceUpCards)
            self.faceUpCards = []
            i = self.reshuffleLimit
            while len(self.faceUpCards) <= 5 and i != 0:
                try:
                    self.faceUpCards.append(self.trainCarDeck.draw(1)[0])
                except:
                    self.discardPile.shuffle(self.seed)
                    self.trainCarDeck.insert(list(self.discardPile.cards))
                    self.discardPile = Deck([])
                    i -= 1

        elif len(self.faceUpCards) < 5 and len(self.discardPile.cards) > 0:
            if self.recordLogs:
                self.gameLogs = self.gameLogs + ["\n> SHUFFLING DISCARDS BACK INTO PLAY\n"]
            self.discardPile.shuffle(self.seed)
            while len(self.faceUpCards) < 5 and len(self.discardPile.cards) > 0:
                self.faceUpCards.append(self.discardPile.draw(1)[0])

    def pickFaceUp(self, action: Action) -> None:
        """
        The function used to pick up a face up card and edit the game details to reflect that change
        """
        if self.lastAction == None:

            if action.colorToDraw not in self.faceUpCards:
                print(f"Attempted to remove {action.colorToDraw} from {self.faceUpCards}")
            self.faceUpCards.remove(action.colorToDraw)
            if len(self.trainCarDeck.cards) > 0:
                replacementCard = self.trainCarDeck.draw(1)
                self.faceUpCards.append(replacementCard[0])
            self.players[self.turn % len(self.players)].trainCardHand.append(action.colorToDraw)

            if self.recordLogs:
                self.gameLogs = self.gameLogs + [f"> pick up {action.colorToDraw}\n"]
            if action.colorToDraw == "WILD":
                self.turn += 1
                self.lastAction = None
            else:
                self.lastAction = action

            if len(self.faceUpCards) == 0:
                self.turn += 1
                self.lastAction = None
                return

        elif self.lastAction.action == 1 or self.lastAction.action == 2:

            assert action.colorToDraw != "WILD"
            self.faceUpCards.remove(action.colorToDraw)
            if len(self.trainCarDeck.cards) > 0:
                replacement = self.trainCarDeck.draw(1)
                self.faceUpCards.append(replacement[0])
            self.players[self.turn % len(self.players)].trainCardHand.append(action.colorToDraw)

            if self.recordLogs:
                self.gameLogs = self.gameLogs + [f"> pick up {action.colorToDraw}\n"]
                self.gameLogs = self.gameLogs + [f"\n{self.players[self.turn % len(self.players)]}\n\n"]

            self.turn += 1
            self.lastAction = None
            

    def placeRoute(self, city1: str, city2: str, colorsUsed: list[str]) -> None:
        """
        The function used to place a route and edit the game details to reflect that change
        """
        info = []
        openPath = None
        atLeastOneOpen: bool | None = None
        atLeastOneTaken: bool | None = None
        for path in self.board.get_edge_data(city1, city2).values():
            info.append(path)
            if path['owner'] == None:
                atLeastOneOpen = True
                openPath = path
            else:
                atLeastOneTaken = True
        
        if len(self.players) < 4 and atLeastOneTaken:
            raise TypeError(f"TURN {self.turn} Attempted to take second connection in a 2 or 3 player game: {', '.join(info)}")
        
        if atLeastOneOpen == None:
            raise TypeError(f"TURN {self.turn} Route requested to be placed is simply not open {', '.join(info)}")
        
        if openPath['owner'] != None:
            print("?")

        for path in self.board.get_edge_data(city1, city2).values():
            if openPath['index'] == path['index']:
                path['owner'] = self.players[self.turn % len(self.players)].turnOrder

        using = []
        colorsGiven = 0
        weight = openPath['weight']
        for color in colorsUsed:
            while color in self.players[self.turn % len(self.players)].trainCardHand and colorsGiven < weight:
                self.players[self.turn % len(self.players)].trainCardHand.remove(color)
                using.append(color)
                colorsGiven += 1
        
        assert weight == colorsGiven
        self.players[self.turn % len(self.players)].trainsLeft -= weight
        self.players[self.turn % len(self.players)].points += pointsByLength[weight]
        assert self.players[self.turn % len(self.players)].trainsLeft >= 0
        
        if self.recordLogs:
            self.gameLogs = self.gameLogs + [f"> placed {city1} --{weight}-- {city2} {openPath} using {using}\n"]
            self.gameLogs = self.gameLogs + [f"\n{self.players[self.turn % len(self.players)]}\n\n"]

        self.discardPile.insert(using)
        if self.players[self.turn % len(self.players)].trainsLeft < 3:
            if self.lastRound != True:
                self.lastRound = True
                self.causedLastTurn = self.players[self.turn % len(self.players)].turnOrder
                self.lastRoundTurn = self.turn

        self.lastAction = None
        self.turn += 1

    def findMaxWeightForNode(self, playerBoard: nx.MultiGraph, source, visitedEdges):
        tempEdges = [e for e in playerBoard.edges() if e not in visitedEdges and source in e]
        if len(tempEdges) == 0:
            return 0
        else:
            result = []
            result.extend([(self.findMaxWeightForNode(playerBoard, x, visitedEdges+[(x, y)]) + playerBoard[x][y][0]['weight']) for (x, y) in tempEdges if source == y])
            result.extend([(self.findMaxWeightForNode(playerBoard, x, visitedEdges+[(x, y)]) + playerBoard[y][x][0]['weight']) for (x, y) in tempEdges if source == x])
            return max(result)

    def endGame(self) -> None:

        self.gameIsOver = True
        longestRouteValue = None
        longestRoutePlayer = []
        for player in self.players:
            if self.recordLogs:
                self.gameLogs = self.gameLogs + [f"\n{player.name} points calculation\n"]
            playerBoard = nx.MultiGraph()
            playerBoard.add_edges_from(edge for edge in self.board.edges(data=True) if edge[2]['owner'] == player.turnOrder)
            if self.recordLogs:
                for claimed in playerBoard.edges(data=True):
                    self.gameLogs = self.gameLogs + [f"\t(+{pointsByLength[claimed[2]['weight']]}) placed {claimed[0]} to {claimed[1]}\n"]
            for destination in player.destinationCardHand:
                if playerBoard.has_node(destination.city1) and playerBoard.has_node(destination.city2):
                    if nx.has_path(playerBoard, destination.city1, destination.city2):
                        player.points += destination.points
                        if self.recordLogs:
                            self.gameLogs = self.gameLogs + [f"\t(+{destination.points}) {destination} completed\n"]
                    else:
                        if self.recordLogs:
                            self.gameLogs = self.gameLogs + [f"\t(-{destination.points}) {destination} not completed\n"]
                else:
                    if self.recordLogs:
                        self.gameLogs = self.gameLogs + [f"\t(-{destination.points}) {destination} not completed\n"]

            temparr = [self.findMaxWeightForNode(playerBoard, node, []) for node in playerBoard.nodes()]
            temp = 0
            if len(temparr) > 0:
                temp = max(temparr)
            if longestRouteValue == None or temp >= longestRouteValue:
                if longestRouteValue and temp > longestRouteValue:
                    longestRoutePlayer = [player.turnOrder]
                else:
                    longestRoutePlayer.append(player.turnOrder)
                longestRouteValue = temp
            
        for playerIndex in longestRoutePlayer:
            self.players[playerIndex].points += 10
            if self.recordLogs:
                self.gameLogs = self.gameLogs + [f"\n{self.players[playerIndex].name} gets the longest route!\n\n"]

        winners = sorted([player for player in self.players], key=lambda p: p.points, reverse=True)
        if winners[0].points == winners[1].points:
            if winners[1].gotLongestRoute:
                self.finalStandings.append(winners[1])
                self.finalStandings.append(winners[0])
                self.finalStandings = self.finalStandings + winners[2:]
            else:
                self.finalStandings = winners
        else:
            self.finalStandings = winners

        if self.recordLogs:
            self.log()

    def stateToInput(self) -> np.ndarray:
        '''
        Given a game object, converts it to a neural network friendly input of length 511

        1. Available Destinations [30] - every destination possible is given an index and value, value 
        a 1 if it is available to pick up and a 0 if it is not

        2. Destinations held by player [30] - same as #1 but for which destinations the player is holding

        3. Destinations by opponent [3] - array of length 4 to count the raw number of destinations picked
        up by each player where the current player = index 0, next player is index 1... etc

        4. Routes taken by player [400] - An array of length 400, 100 spaces for each player. Each index
        corresponds to one route and contains a value 1 if the player has taken that route and 0 if not

        5. Available colors [9] - An index for each color available to have as a card. The value corresponds
        to how many of that color is currently showing up on the board available to take

        6. Color counting [39] - 9 color spaces for each player, where the first nine is the next player to go,
        the next nine is the next... and so on. Value is 1 for that index if that player has that color. +3 for an unknown slot for opponents but not next player to go
        
        Returns an ndarray of shape (1, 511) which represents the current game state
        '''

        # Generate a list of the order of the next turns in the game before looping back (helper)
        numPlayers = len(self.players)
        playerOrderIndexes = [self.turn % numPlayers]
        for _ in range(numPlayers-1):
            playerOrderIndexes.append((playerOrderIndexes[len(playerOrderIndexes)-1] + 1) % numPlayers)

        destAvail = [0] * 30
        if self.destinationCardsDealt:
            for destination in self.destinationCardsDealt:
                destAvail[destination.id] = 1
        
        destHeld = [0] * 30
        for destination in self.players[playerOrderIndexes[0]].destinationCardHand:
            destHeld[destination.id] = 1

        x = 0
        destCount = [0] * 3
        for turnOrder in playerOrderIndexes[1:]:
            destCount[x] = len(self.players[turnOrder].destinationCardHand)
            x += 1
        
        x = 0
        routesTaken = []
        for turnOrder in playerOrderIndexes:
            taken = [0] * 100
            edges = [edge for edge in self.board.edges(data=True) if edge[2]['owner'] == turnOrder]
            for edge in edges:
                taken[edge[2]['index']] = 1
            routesTaken += taken
        while len(routesTaken) != 400:
            routesTaken.append(0)
        
        colorAvail = [self.faceUpCards.count(color) for color in color_indexing.keys()]

        colorCount = [self.players[playerOrderIndexes[0]].trainCardHand.count(color) for color in color_indexing.keys()]
        for turnOrder in playerOrderIndexes[1:]:
            colorCount += self.players[playerOrderIndexes[0]].colorCounts[turnOrder]
        
        inputArray = destAvail + destHeld + destCount + routesTaken + colorAvail + colorCount
        assert len(inputArray) == 511
        
        input = np.array(inputArray).reshape((1, 1, 511))

        return input

    def draw(self) -> None:
        pos = nx.spectral_layout(self.board)
        nx.draw_networkx_nodes(self.board, pos)
        nx.draw_networkx_labels(self.board, pos, font_size=6)
        for player in self.players:
            edges = [edge for edge in self.board.edges(data=True) if edge[2]['owner'] == player.turnOrder]
            nx.draw_networkx_edges(self.board, pos, edges, edge_color=graphColors[player.turnOrder], connectionstyle=f"arc3, rad = 0.{player.turnOrder}", arrows=True)
        matplotlib.pyplot.title(f"AutoTTR Engine Game\n{self.turn} turns")
        matplotlib.pyplot.show()

    def log(self, path: str = "log.txt") -> None:
        for player in self.finalStandings:
            self.gameLogs = self.gameLogs + [f"{player.name}: {player.points}\n"]
        file = open(path, "w")
        file.writelines(self.gameLogs)