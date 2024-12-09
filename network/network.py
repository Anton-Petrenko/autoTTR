'''
network.py
-----------
The code for the autoTTR network
'''

import keras
from engine.game import Game, Action
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
    def __init__(self, learningRate, momentum, weightDecay, loadFrom: int = -1) -> None:

        self.saves: int = None

        if loadFrom == -1:
            self.saves = 0
        else:
            self.saves = loadFrom + 1
        self.model = None

        if loadFrom == -1:
 
            # Residual Tower
            inputLayer = keras.Input((1, 511,))
            resTower_1 = keras.layers.Conv1D(256, 1, kernel_regularizer=keras.regularizers.l2(weightDecay))(inputLayer)
            resTower_2 = keras.layers.BatchNormalization()(resTower_1)
            resTower_3 = keras.layers.ReLU()(resTower_2)

            resBlock_1 = keras.layers.Conv1D(256, 1, kernel_regularizer=keras.regularizers.l2(weightDecay))(resTower_3)
            resBlock_2 = keras.layers.BatchNormalization()(resBlock_1)
            resBlock_3 = keras.layers.ReLU()(resBlock_2)
            resBlock_4 = keras.layers.Conv1D(256, 1, kernel_regularizer=keras.regularizers.l2(weightDecay))(resBlock_3)
            resBlock_5 = keras.layers.BatchNormalization()(resBlock_4)
            resBlock_6 = keras.layers.Add()([resTower_3, resBlock_5])
            resBlock_7 = keras.layers.ReLU()(resBlock_6)

            # Output Heads
            a_Out = keras.layers.Conv1D(2, 1, kernel_regularizer=keras.regularizers.l2(weightDecay))(resBlock_7)
            a_Out = keras.layers.BatchNormalization()(a_Out)
            a_Out = keras.layers.ReLU()(a_Out)
            a_Out = keras.layers.Dense(4, name="action", activation="softmax")(a_Out)

            Dc_Out = keras.layers.Conv1D(2, 1, kernel_regularizer=keras.regularizers.l2(weightDecay))(resBlock_7)
            Dc_Out = keras.layers.BatchNormalization()(Dc_Out)
            Dc_Out = keras.layers.ReLU()(Dc_Out)
            Dc_Out = keras.layers.Dense(9, name="colordesire")(Dc_Out)

            Dd_Out = keras.layers.Conv1D(2, 1, kernel_regularizer=keras.regularizers.l2(weightDecay))(resBlock_7)
            Dd_Out = keras.layers.BatchNormalization()(Dd_Out)
            Dd_Out = keras.layers.ReLU()(Dd_Out)
            Dd_Out = keras.layers.Dense(3, name="destinationdesire", activation="softmax")(Dd_Out)

            Dr_Out = keras.layers.Conv1D(2, 1, kernel_regularizer=keras.regularizers.l2(weightDecay))(resBlock_7)
            Dr_Out = keras.layers.BatchNormalization()(Dr_Out)
            Dr_Out = keras.layers.ReLU()(Dd_Out)
            Dr_Out = keras.layers.Dense(100, name="routedesire", activation="softmax")(Dr_Out)

            w = keras.layers.Conv1D(1, 1, kernel_regularizer=keras.regularizers.l2(weightDecay))(resBlock_7)
            w = keras.layers.BatchNormalization()(w)
            w = keras.layers.ReLU()(w)
            w = keras.layers.Dense(256)(w)
            w = keras.layers.ReLU()(w)
            w = keras.layers.Dense(1, name="value", activation=keras.activations.tanh)(w)

            self.model = keras.Model(inputs=inputLayer, outputs=[a_Out, Dc_Out, Dd_Out, Dr_Out, w])
        
        else:

            self.model = keras.models.load_model(f"latest.keras")

        losses = {
            "action": keras.losses.BinaryCrossentropy(),
            "colordesire": keras.losses.CategoricalCrossentropy(from_logits=True),
            "destinationdesire": keras.losses.BinaryCrossentropy(),
            "routedesire": keras.losses.BinaryCrossentropy(),
            "value": keras.losses.MeanSquaredError()
        }

        self.model.compile(keras.optimizers.Adam(learning_rate=learningRate, use_ema=True, ema_momentum=momentum), loss=losses)

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
        self.model.save(f"latest.keras")
        self.saves += 1