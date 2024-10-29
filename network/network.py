'''
network.py
-----------
The code for the autoTTR network
'''

import keras
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

        # Residual Tower
        inputLayer = keras.Input((1, 511,))
        resTower_1 = keras.layers.Conv1D(256, 1)(inputLayer)
        resTower_2 = keras.layers.BatchNormalization()(resTower_1)
        resTower_3 = keras.layers.ReLU()(resTower_2)

        resBlock_1 = keras.layers.Conv1D(256, 1)(resTower_3)
        resBlock_2 = keras.layers.BatchNormalization()(resBlock_1)
        resBlock_3 = keras.layers.ReLU()(resBlock_2)
        resBlock_4 = keras.layers.Conv1D(256, 1)(resBlock_3)
        resBlock_5 = keras.layers.BatchNormalization()(resBlock_4)
        resBlock_6 = keras.layers.Add()([resTower_3, resBlock_5])
        resBlock_7 = keras.layers.ReLU()(resBlock_6)

        # Output Heads // do not use softmax for outputs in the model arch (see updateWeights in training.py)
        a_Out = keras.layers.Conv1D(2, 1)(resBlock_7)
        a_Out = keras.layers.BatchNormalization()(a_Out)
        a_Out = keras.layers.ReLU()(a_Out)
        a_Out = keras.layers.Dense(4, name="action", activation="softmax")(a_Out)
        #a_Out = layers.Activation('softmax')(a_Out)

        Dc_Out = keras.layers.Conv1D(2, 1)(resBlock_7)
        Dc_Out = keras.layers.BatchNormalization()(Dc_Out)
        Dc_Out = keras.layers.ReLU()(Dc_Out)
        Dc_Out = keras.layers.Dense(9, name="colordesire")(Dc_Out)
        #Dc_Out = layers.Activation('softmax')(Dc_Out)

        Dd_Out = keras.layers.Conv1D(2, 1)(resBlock_7)
        Dd_Out = keras.layers.BatchNormalization()(Dd_Out)
        Dd_Out = keras.layers.ReLU()(Dd_Out)
        Dd_Out = keras.layers.Dense(3, name="destinationdesire", activation="softmax")(Dd_Out)
        #Dd_Out = layers.Activation('softmax')(Dd_Out)

        Dr_Out = keras.layers.Conv1D(2, 1)(resBlock_7)
        Dr_Out = keras.layers.BatchNormalization()(Dr_Out)
        Dr_Out = keras.layers.ReLU()(Dd_Out)
        Dr_Out = keras.layers.Dense(100, name="routedesire", activation="softmax")(Dr_Out)
        #Dr_Out = layers.Activation('softmax')(Dr_Out)

        w = keras.layers.Conv1D(1, 1)(resBlock_7)
        w = keras.layers.BatchNormalization()(w)
        w = keras.layers.ReLU()(w)
        w = keras.layers.Dense(256)(w)
        w = keras.layers.ReLU()(w)
        w = keras.layers.Dense(1)(w)
        w = keras.layers.Activation(keras.activations.tanh)(w)

        self.model = keras.Model(inputs=inputLayer, outputs=[a_Out, Dc_Out, Dd_Out, Dr_Out, w])

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
        return NetworkOutput(outputs[0][0], outputs[1][0], outputs[2][0], outputs[3][0], outputs[4][0])
    
    def thinkRaw(self, raw) -> NetworkOutput:
        outputs = self.model.predict(raw, verbose=False)
        return NetworkOutput(outputs[0][0], outputs[1][0], outputs[2][0], outputs[3][0], outputs[4][0])
    
    def save(self):
        self.model.save(f"saved/model{self.saves}.keras")
        self.saves += 1