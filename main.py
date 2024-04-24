import random
from agents.network import Network
from internal.training import Training
from internal.types import Agent, Game

training = Training("USA", [Agent("Spongebob"), Agent("Patrick"), Agent("Sandy"), Agent("Squidward")], Network("AutoTTR"), simulations=1)
training.train()

# Testing game move engine
# game = Game("USA", [Agent("Spongebob"), Agent("Patrick"), Agent("Sandy"), Agent("Squidward")], True, False)
# while game.gameOver == False:
#     validMove = game.validMoves()
#     if len(validMove) == 0: break
#     game.play(random.choice(validMove))
# game.log()
# game.draw()