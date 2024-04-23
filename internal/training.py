import math
from agents.network import Network, NetworkOutput
from internal.types import Game, Agent, Node, Action, color_indexing

class Training:
    """
    An object responsible for training the network
    """
    def __init__(self, 
                 map: str,
                 players: list[Agent],
                 network: Network
                 ) -> None:
        """
        Initialize the training, with adjustable parameters. Creates a brand new game with four players.
        """
        assert type(network) == Network, "Can only train network"
        self.network: Network = network
        self.game: Game = Game(map, players, False, False)
    
    def train(self):
        """
        Start training the network
        """
        # while self.game.gameOver == False:
        action = self.mcts()
            # self.game.move(action)
    
    def mcts(self):
        """
        The MCTS function called for every state played in the original game simulation
        """
        root = Node(0)
        self.evaluate(root)
    
    def evaluate(self, root: Node):
        """
        The MCTS function for evaluating nodes - producing it's children
        """
        gameImage = self.game.clone()
        validMoves: list[Action] = gameImage.validMoves()
        output: NetworkOutput = self.network.think(gameImage)
        policy = {action: math.exp(self.getLogitMove(output, action)) for action in validMoves}

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
        