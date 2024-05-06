import keras
import numpy as np
from numpy import ndarray
import tensorflow as tf
from internal.types import Agent, Game, Action, getDestinationCards, getRoutes, color_indexing

class NetworkOutput:
    """
    The object for typecasting neural network outputs
    """
    def __init__(self, a: ndarray, Dc: ndarray, Dd: ndarray, Dr: ndarray, w: ndarray) -> None:
        """
        Create a new neural net output object
        """
        assert a.size == 4, "Network 'a' output is not of length 4"
        assert Dc.size == 9, "Network 'Dc' output is not of length 9"
        assert Dd.size == 3, "Network 'Dd' output is not of length 30"
        assert Dr.size == 100, "Network 'Dr' output is not of length 100"
        assert w.size == 1, "Network 'w' output is not of length 1"

        self.a: ndarray = a[0]
        """The networks desired action to take, represented as a list of 4 probabilities"""
        self.Dc: ndarray = Dc[0]
        """The networks desired colors, represented as a list of 9 (indexed by standard index mapping)"""
        self.Dd: ndarray = Dd[0]
        """The networks desired destination cards, represented as a list of 3 (one spot for each route to pick up by index)"""
        self.Dr: ndarray = Dr[0]
        """The networks desired routes to place, represented as a list of 100 (indexed as in the metadata for each Route)"""
        self.w: ndarray = w[0]

class Network(Agent):
    """
    The neural network implementation for the AI agent.
    """
    def __init__(self, name: str, desireSteps: float = 0.05) -> None:
        """Initializes the neural network architecture"""
        super().__init__(name)

        # Extra Parameter
        self.desireSteps = desireSteps

        # Top half
        inputLayer = keras.layers.Input(shape=(511,))
        lstm = keras.layers.Dense(200, name="brain", activation="relu")(inputLayer)

        # Output Heads
        a_Out = keras.layers.Dense(30, name="action30", activation="relu")(lstm)
        a_Out = keras.layers.Dense(10, name="action10", activation="relu")(a_Out)
        a_Out = keras.layers.Dense(4, name="action", activation=None)(a_Out)
        a_Out = keras.layers.Activation('softmax')(a_Out)

        Dc_Out = keras.layers.Dense(30, name="colordesire30", activation="relu")(lstm)
        Dc_Out = keras.layers.Dense(9, name="colordesire", activation="relu")(Dc_Out)
        Dc_Out = keras.layers.Activation('softmax')(Dc_Out)

        Dd_Out = keras.layers.Dense(30, name="destinationdesire30", activation="relu")(lstm)
        Dd_Out = keras.layers.Dense(15, name="destinationdesire30_2", activation="relu")(Dd_Out)
        Dd_Out = keras.layers.Dense(3, name="destinationdesire", activation="relu")(Dd_Out)
        Dd_Out = keras.layers.Activation('softmax')(Dd_Out)

        Dr_Out = keras.layers.Dense(30, name="routedesire30", activation="relu")(lstm)
        Dr_Out = keras.layers.Dense(65, name="routedesire65", activation="relu")(Dr_Out)
        Dr_Out = keras.layers.Dense(100, name="routedesire", activation="relu")(Dr_Out)
        Dr_Out = keras.layers.Activation('softmax')(Dr_Out)

        w = keras.layers.Dense(30, name="winprob30", activation="relu")(lstm)
        w = keras.layers.Dense(15, name="winprob15", activation="relu")(w)
        w = keras.layers.Dense(1, name="winprob", activation="relu")(w)
        w = keras.layers.Activation(keras.activations.sigmoid)(w)

        self.model = keras.Model(inputs=inputLayer, outputs=[a_Out, Dc_Out, Dd_Out, Dr_Out, w])
        # input= np.random.rand(1, 511)
        # outputs = self.model.predict(input)
        # for i, output in enumerate(outputs):
        #     print(f"Output {i+1} shape:", output.shape)
        
    def __str__(self) -> None:
        self.model.summary()

    def stateToInput(self, game: Game) -> ndarray:
        """
        Given a game state, creates an input array with the required shape for input to the neural network.
        """
        # 1. Available Destinations
        destAvail = [0]*len(getDestinationCards(game.map))
        if game.destinationDeal != None:
            for destination in game.destinationDeal:
                destAvail[destination.index] = 1
        
        # 2. Destinations held by current player
        destsHeld = [0]*len(getDestinationCards(game.map))
        for destinationCard in game.makingNextMove.destinationCards:
            destsHeld[destinationCard.index] = 1
        
        # 3. Destination counts per opponent (integer encoding)
        destCount = []
        for i in range(1, len(game.players)):
            index = ((game.turn + i) - 1) % len(game.players)
            destCount.append(len(game.players[index].destinationCards))
        while len(destCount) != 3:
            destCount.append(0)

        # 4. Routes taken per player
        routesTaken = []
        for i in range(len(game.players)):
            taken = [0] * 100
            index = ((game.turn + i) - 1) % len(game.players)
            edges = [edge for edge in game.board.edges(data=True) if edge[2]['owner'] == game.players[index].turnOrder]
            for edge in edges:
                taken[edge[2]['index']] = 1
            routesTaken += taken
        while len(routesTaken) != 400:
            routesTaken.append(0)
        
        # 5. Available colors
        colorAvail = [game.faceUpCards.count(color) for color in color_indexing.keys()]
        
        # 6. Color counting
        colorCount = [game.makingNextMove.trainCards.count(color) for color in color_indexing.keys()]
        for i in range(1, len(game.players)):
            index = ((game.turn + i) - 1) % len(game.players) - 1
            colorCount += game.makingNextMove.colorCounts[index]
        while len(colorCount) != 39:
            colorCount.append(0)
        
        inputArray = destAvail + destsHeld + destCount + routesTaken + colorAvail + colorCount
        input = np.array(inputArray).reshape((1, 511))

        return input

    def think(self, game: Game) -> NetworkOutput:
        """
        Pass a game state through the network. Player to move is relevant player for the network.
        """
        input = self.stateToInput(game)
        outputs = self.model.predict(input, verbose=0)
        return NetworkOutput(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])

    def learn(self, game: Game, action: Action, winner: Agent, optimizer: keras.optimizers.SGD, weightDecay: float) -> None:
        """
        Perform a weight update for the network given a game state and the action (serving as a label). 
        Includes the keras optimizer and a weight decay as well.
        """
        networkInput = self.stateToInput(game)

        aLabel = [1 if x == action.action else 0 for x in range(4)]

        DcLabel = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if action.colorToDraw != None:
            DcLabel[color_indexing[action.colorToDraw]] = 1
        if action.colorsUsed != None:
            factor = 1
            for color in action.colorsUsed:
                DcLabel[color_indexing[color]] = factor
                factor -= self.desireSteps
        
        DdLabel = [0, 0, 0]
        if action.takeDests != None:
            for index in action.takeDests:
                DdLabel[index] = 1
        
        DrLabel = [0] * 100
        if action.route != None: DrLabel[action.route.index] = 1

        wLabel = [1 if game.makingNextMove.turnOrder == winner.turnOrder else 0]

        loss = 0
        output = self.think(game)
        loss += (keras.losses.mean_squared_error(output.w, wLabel) + tf.nn.softmax_cross_entropy_with_logits())

        