'''
network.py
-----------
The code for the autoTTR network
'''

from keras import layers, activations, Model, optimizers
from engine.game import Game
from engine.objects import color_indexing
import numpy as np

class NetworkOutput:
    """
    The object for typecasting neural network outputs
    """
    def __init__(self, a: np.ndarray, Dc: np.ndarray, Dd: np.ndarray, Dr: np.ndarray, w: np.ndarray) -> None:
        """
        Create a new neural net output object
        """
        assert a.size == 4, "Network 'a' output is not of length 4"
        assert Dc.size == 9, "Network 'Dc' output is not of length 9"
        assert Dd.size == 3, "Network 'Dd' output is not of length 30"
        assert Dr.size == 100, "Network 'Dr' output is not of length 100"
        assert w.size == 1, "Network 'w' output is not of length 1"

        self.a: np.ndarray = a[0]
        """The networks desired action to take, represented as a list of 4 probabilities"""
        self.Dc: np.ndarray = Dc[0]
        """The networks desired colors, represented as a list of 9 (indexed by standard index mapping)"""
        self.Dd: np.ndarray = Dd[0]
        """The networks desired destination cards, represented as a list of 3 (one spot for each route to pick up by index)"""
        self.Dr: np.ndarray = Dr[0]
        """The networks desired routes to place, represented as a list of 100 (indexed as in the metadata for each Route)"""
        self.w: np.ndarray = w[0]
        """The chance of winning for the next player to take the turn in the game"""
    
    def __str__(self):
        return f"a: {self.a}\nDc: {self.Dc}\nDd: {self.Dd}\nDr: {self.Dr}\nw: {self.w}"

class AutoTTR:
    '''
    This is the initializer for the AutoTTR model
    '''
    def __init__(self) -> None:

        self.saves: int = 0

        # Top half
        inputLayer = layers.Input(shape=(511,))
        lstm = layers.Dense(200, name="brain", activation="relu")(inputLayer)

        # Output Heads
        a_Out = layers.Dense(30, name="action30", activation="relu")(lstm)
        a_Out = layers.Dense(10, name="action10", activation="relu")(a_Out)
        a_Out = layers.Dense(4, name="action", activation=None)(a_Out)
        a_Out = layers.Activation('softmax')(a_Out)

        Dc_Out = layers.Dense(30, name="colordesire30", activation="relu")(lstm)
        Dc_Out = layers.Dense(9, name="colordesire", activation="relu")(Dc_Out)
        Dc_Out = layers.Activation('softmax')(Dc_Out)

        Dd_Out = layers.Dense(30, name="destinationdesire30", activation="relu")(lstm)
        Dd_Out = layers.Dense(15, name="destinationdesire30_2", activation="relu")(Dd_Out)
        Dd_Out = layers.Dense(3, name="destinationdesire", activation="relu")(Dd_Out)
        Dd_Out = layers.Activation('softmax')(Dd_Out)

        Dr_Out = layers.Dense(30, name="routedesire30", activation="relu")(lstm)
        Dr_Out = layers.Dense(65, name="routedesire65", activation="relu")(Dr_Out)
        Dr_Out = layers.Dense(100, name="routedesire", activation="relu")(Dr_Out)
        Dr_Out = layers.Activation('softmax')(Dr_Out)

        w = layers.Dense(30, name="winprob30", activation="relu")(lstm)
        w = layers.Dense(15, name="winprob15", activation="relu")(w)
        w = layers.Dense(1, name="winprob", activation="relu")(w)
        w = layers.Activation(activations.sigmoid)(w)

        self.model = Model(inputs=inputLayer, outputs=[a_Out, Dc_Out, Dd_Out, Dr_Out, w])

        # test the model input/output
        # input= np.random.rand(1, 511)
        # outputs = self.model.predict(input)
        # for i, output in enumerate(outputs):
        #     print(f"Output {i+1} shape:", output.shape)
        # print(outputs)
    
    def __str__(self) -> None:
        self.model.summary()

    def think(self, game: Game) -> NetworkOutput:
        '''
        Passing in a game, the network will evaluate the state and calculate its output
        '''
        input = game.stateToInput()
        outputs = self.model.predict(input, verbose=False)
        return NetworkOutput(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
    
    def thinkRaw(self, raw) -> NetworkOutput:
        outputs = self.model.predict(raw, verbose=False)
        return NetworkOutput(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
    
    def save(self):
        self.model.save(f"saved/model{self.saves}.keras")
        self.saves += 1