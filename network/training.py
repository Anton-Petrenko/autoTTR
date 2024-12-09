'''
training.py
-----------
The file at top-level that will train the network, MCTS
'''

import math
from numpy import random as nprand
from numpy import ndarray
import numpy as np
from keras import optimizers
import keras
from tensorflow import nn, stop_gradient
import tensorflow as tf
from engine.game import Game
from engine.objects import Player, Action, color_indexing, actionsAreEqual
from network.network import AutoTTR, NetworkOutput
from random import randint, choices

class Datapoint:
    """
    An object to represent one datapoint that includes the network input and labels for all output heads
    """
    def __init__(self, game: Game, index: int):
        self.inputState = game.states[index]
        self.label_a = np.array(self.getLabel_a(game, index))
        self.label_Dc = np.array(self.getLabel_Dc(game, index))
        self.label_Dd = np.array(self.getLabel_Dd(game, index))
        self.label_Dr = np.array(self.getLabel_Dr(game, index))
        self.label_w = np.array(self.getLabel_w(game, index))

    def __str__(self):
        return f"a: {self.label_a}\nDc: {self.label_Dc}\nDd: {self.label_Dd}\nDr: {self.label_Dr}\nw: {self.label_w}"
    
    def getLabel_w(self, game: Game, index: int) -> int:
        """
        Returns if the player taking the action at the index of the game given is the player that won the game.
        
        Returns 0 if lost, 1 if won
        """
        if index < 4:
            return [0.0] if game.finalStandings[0].turnOrder != game.turn % len(game.players) else [1.0]
        
        destsHeld = game.states[index][0][0][30:60]

        for dests in game.finalStandings[0].destinationCardHand:
            if destsHeld[dests.id] == 1:
                return [1.0]
        
        return [0.0]
    
    def getLabel_Dr(self, game: Game, index: int) -> list[int]:
        """
        Returns a list, indexed by route ID's from the game, that reflects the action given by the parameters
        """
        Dr = [0.0]*100

        if game.history[index].action != 0:
            return Dr
        
        Dr[game.history[index].route.id] = 1.0

        return Dr
    
    def getLabel_Dd(self, game: Game, index: int) -> list[int]:
        """
        Returns a list of desired routes to take where the index of the number is whether to take that index of the deal
        """
        Dd = [0.0]*3

        if game.history[index].action != 3:
            return Dd
        if game.history[index].askingForDeal == True:
            return Dd
        
        for i in game.history[index].takeDests:
            Dd[i] = 1.0
        
        return Dd
    
    def getLabel_Dc(self, game: Game, index: int) -> list[int]:
        """
        Returns a list of probabilities for each color to take by color_index
        """
        Dc = [0.5]*9
        if game.history[index].action == 0:
            Dc = [1.0]*9
            for i, color in enumerate(game.history[index].colorsUsed):
                Dc[color_indexing[color]] = 0 + (i * 0.05)
        elif game.history[index].action == 1:
            Dc = [0.0]*9
            Dc[color_indexing[game.history[index].colorToDraw]] = 1.0
        return Dc

    def getLabel_a(self, game: Game, index: int) -> list[int]:
        """
        The label for probabilities of which move to make (int)
        """
        a = [0.0]*4
        a[game.history[index].action] = 1.0
        return a

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

    def __str__(self):
        return f"NODE // visits: {self.visits} winprob: {self.totalWinProb} prior: {self.priorProb} children: {len(self.children)} toPlay: {self.toPlay} terminal: {self.terminal}"

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

class NetworkTrainer:
    
    def __init__(self, 
                 map: str = "USA", 
                 numPlayers: int | None = None, 
                 gameSimsPerBatch: int = 1, 
                 mctsSimsPerMove: int = 1, 
                 logs: bool = False, 
                 rootDirichletAlpha: float = 0.2, 
                 rootExploreFraction: float = 0.25, 
                 pb_cBase: float = 19652, 
                 pb_cInit: float = 1.25, 
                 numSamplingMoves: int = 10, 
                 momentum: float = 0.9, 
                 learningRate: float = 0.001,
                 windowSize: int = 100, 
                 networkSaveInterval: int = 1, 
                 trainingSteps: int = 1, 
                 batchSize: int = 1,
                 weightDecay: int = 0.0001,
                 loadFrom: int = -1) -> None:
        '''
        Set up the training object (default values used are from AlphaZero)

        NOTE: if numPlayers is None, it will train on games with random numbers of players on each simulation
        '''
        self.logs = logs

        # Network to train
        if self.logs:
            print("[NetworkTrainer] Creating a new model...")
        self.network = AutoTTR(learningRate, momentum, weightDecay, loadFrom)
        """The latest network to use - in AlphaGoZero, this is fluid and done in parallel"""
        
        # Storage
        self.windowSize = windowSize
        self.networks = []
        self.gamesPlayed: list[Game] = []

        # Simulation variables
        self.map = map
        self.numPlayers = numPlayers
        self.gameSimsPerBatch = gameSimsPerBatch
        self.mctsSimsPerMove = mctsSimsPerMove

        # Exploration noise
        self.rootDirichletAlpha = rootDirichletAlpha
        self.rootExploreFraction = rootExploreFraction

        # UCB formula
        self.pb_cBase = pb_cBase
        self.pb_cInit = pb_cInit

        # Training
        self.momentum = momentum
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.trainingSteps = trainingSteps
        self.numSamplingMoves = numSamplingMoves
        self.networkSaveInterval = networkSaveInterval
    
    def run(self):
        """
        This function will perform the training of the neural network.

        NOTE: In the AlphaGoZero paper, training and self-play is done in parallel. However, this function takes care of both,
        meaning it is done sequentially.
        """
        steps = 1
        while True:
        #for i in range(self.trainingSteps):
            
            for _ in range(self.gameSimsPerBatch):
                game = self.playGame()
                self.saveGame(game)
            
            batch = self.sampleBatch() # Create data to train on
            self.updateWeights(batch)

            if steps == self.networkSaveInterval:
                self.network.save()
                steps = 1
            else:
                steps += 1

    def updateWeights(self, batch: list[Datapoint]):
        for data in batch:
            label = {
                "action": data.label_a.reshape((1, 1, 4)),
                "colordesire": data.label_Dc.reshape((1, 1, 9)),
                "destinationdesire": data.label_Dd.reshape((1, 1, 3)),
                "routedesire": data.label_Dr.reshape((1, 1, 100)),
                "value": data.label_w
            }
            out = self.network.model.train_on_batch(data.inputState, label, return_dict=True)

    def sampleBatch(self) -> list[Datapoint]:
        totalMoves = sum([len(game.history) for game in self.gamesPlayed])
        gamesForTraining: ndarray = nprand.choice(
            self.gamesPlayed, 
            size=self.batchSize, 
            p=[len(game.history) / totalMoves for game in self.gamesPlayed])
        gamePos = [(game, nprand.randint(len(game.history))) for game in gamesForTraining]
        return [Datapoint(game, index) for (game, index) in gamePos]

    def playGame(self) -> Game:
        """
        Play one full game of Ticket to Ride using self.network, simulating MCTS for each state
        """

        # Create the players list
        players: list[Player] = list()
        if self.numPlayers:
            for i in range(self.numPlayers):
                newPlayer = Player(f"Player{i}")
                players.append(newPlayer)
        else:
            for i in range(randint(2,4)):
                newPlayer = Player(f"Player{i}")
                players.append(newPlayer)

        game = Game(players, logs=True)

        print("simulating new game...")
        while not game.gameIsOver:
            print(f"\t{game.turn}")
            action, root = self.mcts(game)
            if action != None:
                game.play(action)
            self.storeSearchStats(game, root)
        return game

    def storeSearchStats(self, game: Game, root: Node) -> None:
        totalVisits = sum([child.visits for child in root.children.values()])
        game.childVisits.append({action: root.children[action].visits / totalVisits for action in root.children.keys()})

    def mcts(self, game: Game) -> tuple[Action, Node]:
        
        root = Node(0)
        self.evaluate(root, game)
        self.addNoise(root)

        for _ in range(self.mctsSimsPerMove):
            node = root
            path: list[Node] = []
            gameClone: Game = game.clone()

            while node.isExpanded():
                action, node = self.selectChild(node)
                try:
                    gameClone.play(action)
                except:
                    # DO NOT FORGET TO DISABLE LOGS FOR game AND gameClone AFTER DELETING THIS CODE!!
                    print("Printing log of parent game to parent_log.txt")
                    game.log("parent_log.txt")
                    print("Printing log of game that caused error to error_log.txt")
                    gameClone.log("error_log.txt")
                    print("Printing path of the node")
                    for thing in path:
                        print(thing)
                    quit()
                path.append(node)
            
            value = self.evaluate(node, gameClone)
            self.backpropagate(path, value, root, gameClone.players[gameClone.turn % len(gameClone.players)])

        return (self.selectAction(root, game), root)
    
    def selectAction(self, root: Node, game: Game) -> Action:
        """
        Returns the action to take after completing the MCTS search
        """
        visitCounts = [(child.visits, action) for action, child in root.children.items()]
        action: Action = None
        if game.totalActionsTaken < self.numSamplingMoves:
            action = self.softmaxSample(visitCounts)
        else:
            vc = -1
            for tup in visitCounts:
                if tup[0] > vc:
                    action = tup[1]
                    vc = tup[0]

        return action
    
    def softmaxSample(self, counts: list[tuple[int, Action]]) -> Action:
        """
        Given a list of (visits, Action) pairs, returns a selection of an action that is chosen probabilistically according to the softmax of the visit counts
        """
        sumVisits = 0
        actions: list[Action] = []
        for tup in counts:
            sumVisits += tup[0]
            actions.append(tup[1])
        weights = []
        for tup in counts:
            weights.append(tup[0] / sumVisits)
        assert len(counts) == len(weights)
        chosen = choices(actions, weights)
        return chosen[0]

    def backpropagate(self, path: list[Node], value: float, root: Node, currentPlayer: Player) -> None:
        """
        Backpropogates the leaf node information up to the given root based on the search path, winning probability, and current player to move
        """
        for node in path:
            if node.toPlay == currentPlayer.turnOrder:
                node.totalWinProb += value
            else:
                node.totalWinProb = 1 - value
            node.visits += 1


    def selectChild(self, node: Node):
        """
        Selects a child of a given node
        """
        maxChild: Node | None = None
        maxScore: float | None = None
        maxAction: Action | None = "guh"
        for action, child in node.children.items():
            score = self.ucbScore(node, child)
            if maxScore == None:
                maxChild = child
                maxScore = score
                maxAction = action
            elif maxScore < score:
                maxChild = child
                maxScore = score
                maxAction = action
        return maxAction, maxChild

    def ucbScore(self, parent: Node, child: Node) -> float:
        pb_c = math.log((parent.visits + self.pb_cBase + 1) / self.pb_cBase) + self.pb_cInit
        pb_c *= math.sqrt(parent.visits) / (child.visits + 1)
        p_score = pb_c * child.priorProb
        v_score = child.value()
        return p_score + v_score

    def evaluate(self, node: Node, game: Game) -> float:
        """
        The MCTS function for evaluating nodes and producing their children - returns the prediction for next player to go winning the game
        """
        
        node.toPlay = game.turn % 4
        output = self.network.think(game)
        validMoves: list[Action] = game.getValidMoves()
        while len(validMoves) == 0:
            print("hit the loop!")
            if game.gameIsOver:
                return output.w[0]
            else:
                validMoves = game.getValidMoves()
        policy = {action: math.exp(self.getLogitMove(output, action)) for action in validMoves}
        policySum = sum(policy.values())
        for action, value in policy.items():
            node.children[action] = Node(value/policySum)
        return output.w[0]
    
    def getLogitMove(self, output: NetworkOutput, action: Action) -> float:
        """
        Takes a whole network output and an Action and returns a single float that represents the probability of the model selecting that move
        """
        a_p = output.a[action.action]
        Dc_p = 1
        Dd_p = 1
        Dr_p = 1

        if action.action == 0:
            for color in action.colorsUsed:
                Dc_p *= output.Dc[color_indexing[color]]
            Dr_p = output.Dr[action.route.id]
            return a_p * Dc_p * Dr_p
        
        elif action.action == 1:
            Dc_p = output.Dc[color_indexing[action.colorToDraw]]
            return a_p * Dc_p
        
        elif action.action == 2:
            return a_p
        
        elif action.action == 3:
            if action.askingForDeal: 
                return a_p
            else:
                for indexToTake in action.takeDests:
                    Dd_p *= output.Dd[indexToTake]
                return a_p * Dd_p
        
    def addNoise(self, node: Node) -> None:
        """
        Adds noise to the beginning of the MCTS search
        """
        actions = node.children.keys()
        noises = nprand.gamma(self.rootDirichletAlpha, 1, len(actions))
        fraction = self.rootExploreFraction
        for action, noise in zip(actions, noises):
            node.children[action].priorProb = (node.children[action].priorProb * (1 - fraction)) + (noise * fraction)
        
    def saveGame(self, game) -> None:
        if len(self.gamesPlayed) > self.windowSize:
            self.gamesPlayed.pop(0)
        self.gamesPlayed.append(game)

def getLogitMoveExternal(output: NetworkOutput, action: Action) -> float:
        """
        Takes a whole network output and an Action and returns a single float that represents the probability of the model selecting that move
        """
        a_p = output.a[action.action]
        Dc_p = 1
        Dd_p = 1
        Dr_p = 1

        if action.action == 0:
            for color in action.colorsUsed:
                Dc_p *= output.Dc[color_indexing[color]]
            Dr_p = output.Dr[action.route.id]
            return a_p * Dc_p * Dr_p
        
        elif action.action == 1:
            Dc_p = output.Dc[color_indexing[action.colorToDraw]]
            return a_p * Dc_p
        
        elif action.action == 2:
            return a_p
        
        elif action.action == 3:
            if action.askingForDeal: 
                return a_p
            else:
                for indexToTake in action.takeDests:
                    Dd_p *= output.Dd[indexToTake]
                return a_p * Dd_p