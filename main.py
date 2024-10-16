'''
main.py
----------
All scripting should be done here
'''

# TODO: configure default players list when intializing the game object
# TODO: can a terminal state in Ticket to Ride be determined by how many moves have been made? ex. chess is 512
# TODO: Verify network.stateToInput
# TODO: make sure lastAction in game object is only for sequential moves by the same player. (not literally the last move made)
# TODO: currentAction variable in game object - needed?
# TODO: sub len(self.players) for static variable?
# TODO: change checkFaceUp to checkDecks or something - discards may be populated with all other decks exhausted
# NOTE: route checking for 2, 3 player games is done in placeRoute function in Game object
# TODO: check backpropagation 1 - value for nodes that are not the current players nodes

from network.training import NetworkTrainer
from random import randint
trainer = NetworkTrainer(numPlayers=4, logs=True)
trainer.run()