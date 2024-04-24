import math
import numpy
from random import choices
from agents.network import Network, NetworkOutput
from internal.types import Game, Agent, Node, Action, color_indexing

class Training:
    """
    An object responsible for training the network
    """
    def __init__(self, 
                 map: str,
                 players: list[Agent],
                 network: Network,
                 gameSimulations: int = 1,
                 simulationsPerState: int = 1,
                 rootDirichletAlpha: float = 0.2,
                 rootExploreFraction: float = 0.25,
                 pb_cBase: float = 19652,
                 pb_cInit: float = 1.25,
                 numSamplingMoves: int = 10
                 ) -> None:
        """
        Initialize the training, with adjustable parameters. Creates a brand new game with four players.
        """
        self.buffer: list[Game] = []
        self.network: Network = network
        self.simulations = simulationsPerState
        self.gameSimulations = gameSimulations
        self.networks: dict[float, Network] = {}
        self.game: Game = Game(map, players, True, False)
        assert type(network) == Network, "Can only train network"

        # Exploration noise
        self.rootDirichletAlpha = rootDirichletAlpha
        self.rootExploreFraction = rootExploreFraction

        # UCB Formula
        self.pb_cBase = pb_cBase
        self.pb_cInit = pb_cInit

        self.numSamplingMoves = numSamplingMoves
    
    def train(self):

        # 1. Self-Play Data Generation
        for _ in range(self.gameSimulations):
            # Do training for a specified number of game simulations - gets network from shared storage
            pass

        # 2. Network Training
        pass

    def playGame(self):
        """
        Tell the network to play a single game of Ticket to Ride
        """
        while self.game.gameOver == False:
            action, root = self.mcts()
            self.game.play(action)
        self.game.log()
    
    def mcts(self) -> tuple[Action, Node]:
        """
        The MCTS function called for every state played in the original game simulation. It returns a tuple of the (final action decided, root state)
        """
        root = Node(0)
        self.evaluate(root, self.game)
        self.addNoise(root)

        for _ in range(self.simulations):

            node: Node = root
            path: list[Node] = []
            gameClone: Game = self.game.clone()

            while node.isExpanded():
                action, node = self.selectChild(node)
                gameClone.play(action)
                path.append(node)

            value = self.evaluate(node, gameClone)
            self.backpropagate(path, value, root, gameClone.makingNextMove)
        return (self.selectAction(root), root)
    
    def selectAction(self, root: Node) -> Action:
        """
        Returns the action to take after completing the MCTS search
        """
        visitCounts = [(child.visits, action) for action, child in root.children.items()]
        action: Action = None
        if self.game.turn < self.numSamplingMoves:
            action = self.softmaxSample(visitCounts)
        else:
            vc = 0
            for tup in visitCounts:
                if tup[0] > vc:
                    action = tup[1]
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
        assert len(counts) == len(weights), "Counts and weights are not the same size"
        chosen = choices(actions, weights)
        return chosen[0]
    
    def backpropagate(self, path: list[Node], value: float, root: Node, currentPlayer: Agent) -> None:
        """
        Backpropogates the leaf node information up to the given root based on the search path, winning probability, and current player to move
        """
        for node in path:
            if node.toPlay == currentPlayer.turnOrder: node.totalWinProb += value
            else: node.totalWinProb = 1 - value
            node.visits += 1
    
    def selectChild(self, node: Node):
        """
        Selects a child of a given node
        """
        maxChild: Node = None
        maxScore: float = None
        maxAction: Action = None
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

    def addNoise(self, node: Node) -> None:
        """
        Adds noise at the beginning of the MCTS search
        """
        actions = node.children.keys()
        noises = numpy.random.gamma(self.rootDirichletAlpha, 1, len(actions))
        fraction = self.rootExploreFraction
        for action, noise in zip(actions, noises):
            node.children[action].priorProb = node.children[action].priorProb * (1 - fraction) + noise * fraction
    
    def evaluate(self, node: Node, game: Game) -> float:
        """
        The MCTS function for evaluating nodes and producing their children - returns the neural network prediction for the winner of the game
        """
        node.toPlay = game.makingNextMove.turnOrder
        output: NetworkOutput = self.network.think(game)
        validMoves: list[Action] = game.validMoves()
        if len(validMoves) == 0: 
            node.terminal = True
            return output.w[0]
        policy = {action: math.exp(self.getLogitMove(output, action)) for action in validMoves}
        policySum = sum(policy.values())
        for action, value in policy.items():
            node.children[action] = Node(value / policySum)
        return output.w[0]

    def getLogitMove(self, output: NetworkOutput, action: Action) -> float:
        """
        Takes a whole network output and an Action and returns a single float that represents the probability of the model selecting that move
        """
        a_p: float = output.a[action.action]
        Dc_p: float = 1
        Dd_p: float = 1
        Dr_p: float = 1

        if action.action == 0:
            for color in action.colorsUsed:
                Dc_p *= output.Dc[color_indexing[color]]
            Dr_p = output.Dr[action.route.index]
            return a_p * Dc_p * Dr_p
        
        elif action.action == 1:
            Dc_p = output.Dc[color_indexing[action.colorToDraw]]
            return a_p * Dc_p
        
        elif action.action == 2: return a_p
        
        elif action.action == 3:
            if action.askingForDeal: return a_p
            else:
                for indexToTake in action.takeDests:
                    Dd_p *= output.Dd[indexToTake]
                return a_p * Dd_p
        