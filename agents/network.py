import keras
import numpy as np
from numpy import ndarray
import tensorflow as tf
from internal.types import Agent, Game, getDestinationCards, getRoutes, color_indexing

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

        self.a: ndarray = a
        """The networks desired action to take, represented as a list of 4 probabilities"""
        self.Dc: ndarray = Dc
        """The networks desired colors, represented as a list of 9 (indexed by standard index mapping)"""
        self.Dd: ndarray = Dd
        """The networks desired destination cards, represented as a list of 3 (one spot for each route to pick up by index)"""
        self.Dr: ndarray = Dr
        """The networks desired routes to place, represented as a list of 100 (indexed as in the metadata for each Route)"""

class Network(Agent):
    """
    The neural network implementation for the AI agent.
    """
    def __init__(self, name: str) -> None:
        """Initializes the neural network architecture"""
        super().__init__(name)

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

    def think(self, game: Game) -> NetworkOutput:
        """
        Pass the current (cloned) state through the neural network and get output from all output heads
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
        outputs = self.model.predict(input)

        return NetworkOutput(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
